import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from benchmark.validation_cohort import ANCHOR, choose
from benchmark.validation import plan, safe_output, check_profile
from benchmark.common import atomic_json
from benchmark.consumer_replay import diagnose
from benchmark.diagnostics import stage3_diagnosis, reference_geometry_feasibility
from tests.fixtures import build_fixture
from benchmark.common import read_json, digest
import numpy as np
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem


class ValidationBenchmarkTests(unittest.TestCase):
    def test_reference_constraint_diagnostic_keeps_support_claim_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"bundle"
            reference = build_fixture(root, rows=2, cols=2)
            config = SolverConfig()
            problem = load_problem(root, config)
            report = reference_geometry_feasibility(problem, config, reference)
            self.assertTrue(report["current_rules"]["valid"])
            self.assertIsNone(report["three_pixel_erosion_counterfactual"])
            self.assertIn("no selected-support compatibility", report["scope"])

    def test_replay_diagnosis_respects_prediction_barrier(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(root/"request.json", {})
            atomic_json(root/"progress.json", {"phase": "predictions"})
            with self.assertRaisesRegex(ValueError, "barrier"):
                diagnose(["--run", str(root)])

    def test_pool_replay_diagnosis_recovers_original_search_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"bundle"
            reference = build_fixture(root, rows=1, cols=2)
            for c in reference["contacts"]:
                h = np.linalg.inv(reference["pieces"][c["piece_a"]]["native_to_reference"]) @ reference["pieces"][c["piece_b"]]["native_to_reference"]
                c["points_a"] = map_points(c["points_b"], h).tolist()
            manifest = read_json(root/"bundle.json")
            manifest["provenance"] = {"kind": "observable_saved_archive_pool_replay", "source_manifest_sha256": "a"*64,
                "implementation": {"pool": "b"*64}, "source_provenance": {"runtime_variant": "exact-evidence-v1",
                "search": [{"pieces": ["piece_001", "piece_002"]}], "retrieval": {"policy": "original"}}}
            manifest["manifest_sha256"] = digest({k:v for k,v in manifest.items() if k != "manifest_sha256"})
            atomic_json(root/"bundle.json", manifest)
            report = stage3_diagnosis(root, reference)
            self.assertEqual(report["runtime_variant"], "exact-evidence-v1")
            self.assertEqual(report["physical_pairs_processed"], 1)
            self.assertEqual(report["retrieval"], {"policy": "original"})
            self.assertEqual(len(report["pool_replay_lineage"]), 1)

    def test_selection_balances_metadata_and_is_order_independent(self):
        rows = [{"puzzle_id": ANCHOR if tier == "simple" and i == 0 else f"{tier}_{i}",
                 "difficulty_tier": tier, "source_family": f"family{i%4}", "source_id": f"page{tier}{i}"}
                for tier in ("simple", "moderate", "hard", "extreme") for i in range(7)]
        result = choose(rows, 20261022)
        self.assertEqual(result, choose(list(reversed(rows)), 20261022))
        self.assertEqual(len({r["puzzle_id"] for r in result}), 10)
        self.assertEqual(result[0]["puzzle_id"], ANCHOR)
        self.assertEqual([sum(r["difficulty_tier"] == t for r in result) for t in ("simple", "moderate", "hard", "extreme")], [3, 3, 2, 2])
        self.assertEqual(len({r["source_id"] for r in result}), 10)

    def test_full_accelerated_routes_for_twelve_no_smoke_preset(self):
        case = {"case_id": "puzzle12", "path": "cases/puzzle12", "pieces": 12}
        commands = dict(plan(case, Path("inputs"), Path("output"), {"path": "candidate"}, Path("config"), Path("snapshot"), "poolhash"))
        self.assertEqual(list(commands), ["stage2", "stage2_verify", "stage3", "pool", "stage4"])
        stage3 = commands["stage3"]
        self.assertIn("--candidate-runtime", stage3)
        self.assertEqual(stage3[stage3.index("--scope")+1], "full")
        self.assertIn("--no-cprofile", stage3)
        self.assertNotIn("--preset", stage3)
        self.assertNotIn("--baseline", stage3)
        self.assertIn("run_source_snapshot.py", " ".join(commands["stage4"]))
        self.assertNotIn("--initial-result", commands["stage4"])

    def test_protected_directories_reject_descendant_and_ancestor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for out, protected in ((root, root/"inputs"), (root/"inputs/out", root/"inputs")):
                with self.assertRaises(ValueError):
                    safe_output(out, [protected])
            safe_output(root/"out", [root/"inputs"])

    def test_profile_failure_does_not_open_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(root/"profile.json", {"request": {}, "status": "failed"})
            with patch("benchmark.validation.load_bundle") as loader:
                with self.assertRaises(ValueError):
                    check_profile(root, {}, {}, root/"stage1", root/"stage2")
                loader.assert_not_called()

    def test_completed_profile_binds_case_runtime_and_native_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            stage1, stage2 = root/"stage1", root/"stage2"
            candidate = {"manifest": {"variant": "qualified"}}
            p = {"status": "profile_complete", "scope": "full", "sources_verified_unchanged": True,
                 "source_validation_status": "passed", "ready_bundle_published": True,
                 "training_performed": False, "evaluation_truth_opened": False, "runtime_policy_changed": False,
                 "comparison": {"status": "not_requested", "equal": None},
                 "request": {"candidate_runtime": candidate["manifest"], "puzzle_id": "p12", "cprofile": False,
                             "baseline": None, "stage1": str(stage1), "stage2": str(stage2), "expected_inference_signature": "sig"}}
            atomic_json(root/"profile.json", p)
            bundle = SimpleNamespace(manifest={"puzzle_id": "p12", "counts": {"pieces": 12},
                                     "provenance": {"inference_signature": "sig"}, "manifest_sha256": "seal"})
            with patch("benchmark.validation.load_bundle", return_value=bundle):
                self.assertEqual(check_profile(root, {"case_id": "p12", "pieces": 12}, candidate, stage1, stage2), "seal")
                bundle.manifest["puzzle_id"] = "other"
                with self.assertRaises(ValueError):
                    check_profile(root, {"case_id": "p12", "pieces": 12}, candidate, stage1, stage2)
