from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw
from paper_reconstruction_contracts import ContractError
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import Piece, State, singleton_state
from paper_reconstruction_stage4.outputs import atomic_json, digest, file_hash, request_identity, build_result, publish, output_session, read_result
from paper_reconstruction_stage4.solver import solve
from paper_reconstruction_stage4.support import SupportIndex
from paper_reconstruction_stage4.structural import StructuralScorer, image_profile
from paper_reconstruction_stage4.search.structural_beam import StructuralSearch
from paper_reconstruction_stage4.search.common import Budget
from tests.fixtures import build_fixture, translation

ROOT = Path(__file__).resolve().parents[1]


def structural_config(**kwargs):
    return SolverConfig(solver="structural_beam_v2", objective_version="observable-structure.v2", max_seconds=10., **kwargs)


class StructuralTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def text_piece(self, angle=0., blank=False):
        image = Image.new("L", (200, 160), 245)
        draw = ImageDraw.Draw(image)
        if not blank:
            for y in range(15, 150, 12):
                for x in range(12, 185, 14):
                    draw.rectangle([x, y, x+8, y+3], fill=20)
        image = image.rotate(angle, expand=True, fillcolor=245)
        path = self.root / (str(angle)+str(blank)+".png")
        image.save(path)
        return Piece("text", "text", path, np.ones((image.height, image.width), bool), {})

    def test_native_projection_axis_tracks_rotation_and_blank_abstains(self):
        horizontal = image_profile(self.text_piece())
        rotated = image_profile(self.text_piece(30.))
        blank = image_profile(self.text_piece(blank=True))
        self.assertGreater(horizontal["axis_confidence"], .2)
        self.assertGreater(rotated["axis_confidence"], .2)
        delta = .5*np.angle(np.exp(2j*(rotated["angle_radians"]-horizontal["angle_radians"])))
        self.assertAlmostEqual(abs(np.rad2deg(delta)), 30., delta=4.)
        self.assertEqual(blank["axis_confidence"], 0.)

    def fixture(self, **kwargs):
        build_fixture(self.root / "input", **kwargs)
        return load_problem(self.root / "input", structural_config())

    def test_global_axis_can_reject_a_higher_scored_wrong_orientation(self):
        problem = self.fixture(rows=1, cols=2)
        config = structural_config(structural_compactness_weight=0., structural_boundary_weight=0., structural_appearance_weight=0.)
        scorer = StructuralScorer(problem, config, SupportIndex(problem, config))
        for profile in scorer.profiles.values():
            profile.update(angle_radians=0., axis_confidence=1., line_pitch_native_px=None)
        edge = problem.edges[0]
        good = State({"piece_00": np.eye(3), "piece_01": translation(27)}, {"piece_00": "piece_00", "piece_01": "piece_00"}, (edge.id,))
        bad = State({"piece_00": np.eye(3), "piece_01": np.array([[.5, -.866, 70], [.866, .5, 0], [0., 0., 1.]])}, dict(good.groups), (edge.id,))
        good_score = scorer.evaluate(good)["score"]
        edge.weight += .1
        self.assertGreater(good_score, scorer.evaluate(bad)["score"])

    def test_observable_refinement_corrects_tree_gap_without_redundant_edges(self):
        problem = self.fixture(rows=1, cols=2, drift=4.)
        config = structural_config(refine=True, refine_translation_px=16., refine_regularization=.001,
            structural_boundary_weight=1., structural_appearance_weight=0., structural_compactness_weight=0.)
        result = solve(problem, config)
        report = result.diagnostics["refinement"][0]
        self.assertTrue(report["attempted"])
        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["tree_layout_supported"])
        self.assertLess(abs(result.states[0].poses["piece_01"][0, 2]-27), 2.)
        np.testing.assert_array_equal(result.states[0].poses["piece_00"], np.eye(3))
        self.assertTrue(result.diagnostics["final_feasibility"][0]["valid"])

    def test_line_pitch_prior_treats_double_spacing_as_ambiguous(self):
        problem = self.fixture(rows=1, cols=2)
        config = structural_config(structural_compactness_weight=0., structural_boundary_weight=0., structural_appearance_weight=0.)
        scorer = StructuralScorer(problem, config, SupportIndex(problem, config))
        for profile in scorer.profiles.values():
            profile.update(angle_radians=0., axis_confidence=1., line_pitch_native_px=12., pitch_confidence=1.)
        edge = problem.edges[0]
        state = State({"piece_00":np.eye(3),"piece_01":translation(27)}, {"piece_00":"piece_00","piece_01":"piece_00"}, (edge.id,))
        score = scorer.evaluate(state)["score"]
        scorer.profiles["piece_01"]["line_pitch_native_px"] = 24.
        scorer.cache.clear()
        self.assertAlmostEqual(score, scorer.evaluate(state)["score"], places=10)

    def omit_good_pool_pose(self):
        problem = self.fixture(rows=1, cols=2, alternatives=True)
        root = problem.root
        pool = json.loads((root / "working_pool.json").read_text())
        removed = pool["edges"].pop(0)
        pool["pairs"][0]["working_geometries"] -= 1
        atomic_json(root / "working_pool.json", pool)
        manifest = problem.manifest
        manifest["counts"]["working_edges"] -= 1
        manifest["working_pool"].update(size=(root / "working_pool.json").stat().st_size, sha256=file_hash(root / "working_pool.json"))
        manifest["manifest_sha256"] = digest({k:v for k,v in manifest.items() if k != "manifest_sha256"})
        atomic_json(root / "bundle.json", manifest)
        return root, removed["hypothesis_id"]

    def test_bounded_archive_expansion_recovers_omitted_geometry_without_mutation(self):
        root, omitted = self.omit_good_pool_pose()
        before = {p.name:file_hash(p) for p in root.iterdir() if p.is_file()}
        config = structural_config(evidence_policy="working_plus_diverse_archive.v1")
        expanded = load_problem(root, config)
        self.assertIn(omitted, {e.id for e in expanded.edges})
        self.assertEqual(expanded.manifest["counts"]["working_edges"], 1)
        self.assertEqual(expanded.input_summary["consumer_edges"], 2)
        self.assertEqual(before, {p.name:file_hash(p) for p in root.iterdir() if p.is_file()})
        with self.assertRaisesRegex(ContractError, "Expanded occurrence"):
            load_problem(root, replace(config, max_working_occurrences=1))
        result = solve(expanded, config)
        request = request_identity(expanded, config)
        value = build_result(expanded, config, result, request)
        output = self.root / "published"
        with output_session(output, request) as work:
            publish(output, work, value, expanded, result.states)
        sealed = read_result(output)
        self.assertEqual(sealed["solver"], "structural_beam_v2")
        self.assertIn(omitted, {j["geometry_id"] for j in sealed["layouts"][0]["selected_joins"]})

    def test_coordinated_repair_can_cross_a_lower_scoring_intermediate(self):
        config = structural_config(repair_rounds=1, max_repair_checks=100, portfolio_size=1)
        search = object.__new__(StructuralSearch)
        search.config, search.trace, search.priorities = config, [], {"bad1": .9, "bad2": .9}
        from collections import Counter
        search.rejections = Counter()
        def state(selected, score):
            return State({"a":np.eye(3)}, {"a":"a"}, tuple(selected), score=score)
        original = state(["bad1", "bad2"], 1.)
        search.rebuild = lambda kept, budget: state(kept, .2*len(kept))
        search.candidates_for = lambda s: [] if any(e.startswith("bad") for e in s.selected) else [type("Choice", (), {"id":"good1" if not s.selected else "good2"})()]
        search.attempt = lambda s,e: state([*s.selected,e.id], .4 if not s.selected else 1.5)
        improved, report = search.coordinated_repair([original], Budget(config))
        self.assertEqual(improved[0].selected, ("good1", "good2"))
        self.assertEqual(report["tasks_visited"], report["tasks_planned"])
        self.assertTrue(any(len(t["removed_geometry_ids"]) == 2 and t["best_score_after"] > 1. for t in report["trace"]))

    def test_config_requires_explicit_matching_versions_and_bounded_resources(self):
        for changes in ({"solver":"structural_beam_v2"}, {"objective_version":"observable-structure.v2"},
                        {"evidence_policy":"working_plus_diverse_archive.v1"}, {"structural_repair_removals":3}):
            with self.assertRaises(ValueError):
                SolverConfig.from_dict(changes)

    def test_thirty_piece_fixture_obeys_budget_and_preserves_inventory(self):
        problem = self.fixture(rows=5, cols=6)
        result = solve(problem, structural_config(beam_width=4, portfolio_size=2, max_proposals=50))
        self.assertEqual(len(result.states[0].poses),30)
        self.assertLessEqual(result.diagnostics["proposals"],50)
        self.assertEqual(result.termination,"proposal_budget")
        self.assertTrue(result.diagnostics["final_feasibility"][0]["valid"])

    def test_structural_path_retains_collision_reflection_and_pole_guards(self):
        problem = self.fixture(rows=1, cols=2)
        edge = problem.edges[0]
        for matrix in (np.eye(3),np.diag([-1.,1.,1.]),np.array([[1.,0.,0.],[0.,1.,0.],[-1/31.,0.,1.]])):
            edge.transform = matrix
            result = solve(problem,structural_config())
            self.assertFalse(result.states[0].selected)
            self.assertEqual(len(result.states[0].poses),2)
            self.assertTrue(result.diagnostics["final_feasibility"][0]["valid"])

    def test_v1_fixture_choices_match_frozen_executable_control(self):
        problem = self.fixture(alternatives=True, drift=3.)
        config = SolverConfig(max_proposals=5000, max_seconds=10.)
        current = solve(problem, config).states[0]
        code = "import sys,json;sys.path.insert(0,sys.argv[1]);from paper_reconstruction_stage4.inputs import load_problem;from paper_reconstruction_stage4.config import SolverConfig;from paper_reconstruction_stage4.solver import solve;c=SolverConfig(max_proposals=5000,max_seconds=10.);s=solve(load_problem(sys.argv[2],c),c).states[0];print(json.dumps({'selected':s.selected,'poses':{p:h.tolist() for p,h in s.poses.items()}}))"
        done = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT/"controls/stage4_alpha_0_1_0_review_baseline/src"), str(problem.root)], capture_output=True, text=True, check=True)
        frozen = json.loads(done.stdout)
        self.assertEqual(list(current.selected), frozen["selected"])
        for p,h in current.poses.items():
            np.testing.assert_array_equal(h, frozen["poses"][p])
