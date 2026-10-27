"""Optional local parity of archived search with shared standalone math adapters."""
import ast
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.geometry import warp_check
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.outputs import digest, file_hash
from paper_reconstruction_stage4.search import legacy_forest as port
from tests.fixtures import build_fixture

ARCHIVE = Path(__file__).resolve().parents[2] / "paper-reconstruction-stage3" / "research_archive" / "v0.9.31" / "src" / "paper_reconstruction_stage3" / "pose_solver.py"


@unittest.skipUnless(ARCHIVE.is_file(), "Optional local archived source is unavailable")
class ArchivedSearchParity(unittest.TestCase):
    def test_archived_search_and_port_agree_on_identical_contract_edges(self):
        origin = json.loads((Path(__file__).resolve().parents[1] / "vendor" / "ORIGIN.json").read_text())
        self.assertEqual(file_hash(ARCHIVE), origin["legacy_reference_files"]["pose_solver.py"])
        class StandaloneImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.level and node.module in {"alignment", "contracts", "projective"} else node

        tree = ast.fix_missing_locations(StandaloneImports().visit(ast.parse(ARCHIVE.read_text())))
        module = types.ModuleType("stage4_archived_search_parity")
        module.apply_transform = port.map_points
        module.mask_samples = port.mask_samples
        module.collision_fraction = lambda a, b, h, grid, prepared: port.collision_fraction(a, b, h, prepared)
        module.digest = digest
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        module.require = require
        module.warp_check = lambda h, source, target, policy: warp_check(h, source, target, replace(SolverConfig(), **policy), grid=3)
        sys.modules[module.__name__] = module
        try:
            exec(compile(tree, str(ARCHIVE), "exec"), module.__dict__)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for cols, rows in ((3, 1), (2, 2)):
                    bundle = root / f"{cols}_{rows}"
                    build_fixture(bundle, cols=cols, rows=rows, alternatives=True, isolated=True)
                    config = SolverConfig(solver="legacy_forest_control_v1", beam_width=16)
                    problem = load_problem(bundle, config)
                    entries = {p: {"mask_path": str(bundle / problem.manifest["pieces"][p]["mask"])} for p in problem.pieces}
                    edges = [{"id": e.id, "a": e.a, "b": e.b, "weight": e.weight, "transform": e.transform.tolist()} for e in problem.edges]
                    old_config = {"beam_width": config.beam_width, "collision_grid": config.collision_grid,
                                  "maximum_assembly_collision": config.maximum_assembly_collision, "cycle_tolerance_px": config.cycle_tolerance_px}
                    expected = module.solve(entries, edges, old_config)
                    actual = port.solve(problem, config).states[0]
                    self.assertEqual(actual.components(), expected["components"])
                    self.assertEqual(actual.selected, tuple(e["id"] for e in expected["tree_edges"]))
                    self.assertAlmostEqual(actual.score, expected["observable_score"], places=10)
                    for piece in problem.pieces:
                        np.testing.assert_allclose(actual.poses[piece], expected["poses"][piece], atol=1e-10)
        finally:
            sys.modules.pop(module.__name__, None)
