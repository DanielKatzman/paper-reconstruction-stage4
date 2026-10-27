from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from benchmark.joint_diagnostics import audit_candidates
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.outputs import build_result, digest, file_hash, request_identity
from paper_reconstruction_stage4.solver import solve
from tests.fixtures import build_fixture


class CandidateDiagnosticPreviewTests(unittest.TestCase):
    def test_reference_selected_preview_is_labeled_and_does_not_change_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reference = build_fixture(root/"input", rows=1, cols=2)
            config = SolverConfig(solver="joint_beam_v3", objective_version="observable-joint.v4", joint_trials=2,
                                  max_seconds=2., repair_rounds=0, refine=False)
            problem = load_problem(root/"input", config)
            search = solve(problem, config)
            result = build_result(problem, config, search, request_identity(problem, config))
            before = digest(result)
            path = root/"diagnostic.png"
            report = audit_candidates(result, problem, config, reference, maximum=2, seconds=2., preview_path=path)
            self.assertEqual(digest(result), before)
            self.assertGreater(report["measured_valid"], 0)
            self.assertEqual(report["unique_records_available"], 1)
            self.assertGreater(report["exact_duplicate_records_skipped"], 0)
            self.assertIn("Minimum common-gauge", report["diagnostic_preview"]["selection"])
            self.assertEqual(report["diagnostic_preview"]["sha256"], file_hash(path))
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0,0)), (255,255,255))
                self.assertGreater(image.height, 48)
                self.assertGreater(len(np.unique(np.asarray(image.crop((0,0,image.width,48))).reshape(-1,3), axis=0)), 1)
