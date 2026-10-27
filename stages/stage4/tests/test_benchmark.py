"""Benchmark checks emphasize coordinate correctness and failure accounting."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from paper_reconstruction_contracts import map_points
from benchmark.common import SIZES, atomic_json, digest, inventory, verify_files, load_cohort
from benchmark.prepare import crop_observation, select_cases
from benchmark.diagnostics import arc_mask, arc_ious, classify_contact, pose_error, stage3_diagnosis, seam_coverage, diagnose_case
from benchmark.runner import execute, select_subset, completed_stage2, main as run_benchmark
from benchmark.report import write_reports


class BenchmarkTests(unittest.TestCase):
    def test_metadata_only_selection_deterministic_balanced(self):
        rows = []
        for size in SIZES:
            for tier in ("simple", "moderate", "hard", "extreme"):
                for i in range(5):
                    rows.append({"actual_piece_count": size, "difficulty_tier": tier, "source_family": str(i),
                                 "puzzle_id": f"p{size}_{tier}_{i}", "split": "train"})
        selected = select_cases(rows)
        self.assertEqual([r["puzzle_id"] for r in selected], [r["puzzle_id"] for r in select_cases(list(reversed(rows)))])
        self.assertEqual(sum(r["actual_piece_count"] for r in selected), 568)
        for size in SIZES:
            group = [r for r in selected if r["actual_piece_count"] == size]
            self.assertEqual(len(group), 4)
            self.assertEqual(len({r["source_family"] for r in group}), 4)
        protected = [{**r, "split": "holdout", "puzzle_id": "protected"+r["puzzle_id"]} for r in rows]
        self.assertEqual(selected, select_cases(rows+protected))

    def test_crop_homography_preserves_observation_frame(self):
        mask = np.zeros((18, 22), np.uint8)
        mask[4:12, 7:19] = 255
        h = np.array([[.9, -.2, 123], [.1, 1.2, 45], [.0002, -.0001, 1.]])
        _, cropped, bbox, native_to_page = crop_observation(Image.new("RGB", (22, 18)), Image.fromarray(mask), h)
        self.assertEqual(bbox, [3, 0, 19, 16])
        self.assertEqual(cropped.size, (19, 16))
        probes = np.array([[0, 0], [11, 7], [3, 5]], float)
        np.testing.assert_allclose(map_points(probes, native_to_page), map_points(probes+[3, 0], h))
        self.assertFalse(np.asarray(cropped)[0].any())
        changed = mask.copy()
        changed[16, 21] = 255
        with self.assertRaises(ValueError):
            crop_observation(Image.new("RGB", (22, 18)), Image.fromarray(changed), h)

    def test_circular_arc_and_missing_seam_not_a_hit(self):
        c = {"start_index": 8, "length_points": 4, "rank": 1, "candidate_id": "a"}
        target = np.zeros(10, bool)
        target[[8, 9, 0, 1]] = True
        np.testing.assert_equal(arc_mask(c, 10), target)
        self.assertEqual(arc_ious([c], target)["1"], 1)
        self.assertEqual(arc_ious([c], np.zeros(10, bool))["32"], 0)
        broad = {**c, "length_points": 10}
        self.assertLess(arc_ious([broad], target)["32"], .5)
        self.assertEqual(seam_coverage([broad], target)["maximum_seam_fraction_in_any_candidate"], 1)
        self.assertEqual(seam_coverage([broad], target)["best_candidate_precision_when_seam_fraction_at_least_0_8"], .4)

    def test_pose_probes_use_saved_native_correspondences(self):
        contact = {"points_a": [[5, 3], [8, 7]], "points_b": [[0, 0], [3, 4]]}
        self.assertEqual(pose_error([[1, 0, 5], [0, 1, 3], [0, 0, 1]], contact), 0)
        self.assertGreater(pose_error(np.eye(3), contact), 4)

    def test_loss_localization_preserves_unknown_retrieval(self):
        row = {"working_accurate": 0, "archive_accurate": 0, "physical_pair_retrieved": None,
               "physical_pair_retained_for_pose_search": None}
        self.assertEqual(classify_contact(row), "stage3_no_accurate_direct_pose_unlocalized")
        row["physical_pair_retrieved"] = False
        self.assertEqual(classify_contact(row), "stage3_physical_pair_retrieval_miss")
        row.update(physical_pair_retrieved=True, physical_pair_retained_for_pose_search=False)
        self.assertEqual(classify_contact(row), "stage3_parent_selection_miss")
        row["physical_pair_retained_for_pose_search"] = True
        self.assertEqual(classify_contact(row), "stage3_pose_search_gap")
        row["archive_accurate"] = 1
        self.assertEqual(classify_contact(row), "stage3_working_pool_loss")
        row["working_accurate"] = 1
        self.assertEqual(classify_contact(row), "stage4_assembly_gap")
        self.assertEqual(classify_contact(row, True), "final_contact_correct")

    def test_receipts_detect_asset_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root / "image.png"
            p.write_bytes(b"input")
            files = inventory(root)
            verify_files(root, files)
            p.write_bytes(b"other")
            with self.assertRaises(ValueError):
                verify_files(root, files)

    def test_serving_verification_does_not_open_evaluation_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "case"
            (case / "stage1").mkdir(parents=True)
            (case / "evaluation_only").mkdir()
            (case / "stage1/input").write_bytes(b"image")
            truth = case / "evaluation_only/reference"
            truth.write_bytes(b"reference")
            manifest = {"schema_version": "stage4.development-benchmark.v1", "cases": [{"path": "case", "files": inventory(case)}]}
            manifest["benchmark_sha256"] = digest(manifest)
            atomic_json(root / "benchmark.json", manifest)
            truth.write_bytes(b"changed reference")
            load_cohort(root, serving_only=True)
            with self.assertRaises(ValueError):
                load_cohort(root)

    def test_failed_process_log_and_null_accuracy_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = execute([sys.executable, "-c", "print('diagnostic sentinel'); raise SystemExit(7)"], root / "failure.log")
            self.assertEqual(attempt["returncode"], 7)
            self.assertIn("diagnostic sentinel", (root / "failure.log").read_text())
            case = {"case_id": "puzzle_1", "pieces": 4, "difficulty": "hard", "source_family": "page <&>.pdf",
                    "source_page_index": 1, "path": "cases/puzzle_1"}
            report = {"case": case, "stages": {"stage4": {"status": "not_published"}}, "outputs": {}, "errors": [], "diagnosis_counts": {}}
            write_reports([report], root / "inputs", root)
            summary = json.loads((root / "report/summary.json").read_text())
            self.assertIsNone(summary["cases"][0]["strict_pass"])
            self.assertIsNone(summary["by_size"][0]["mean_contact_recall_published_only"])
            self.assertEqual(summary["by_size"][0]["predictions_published"], 0)
            self.assertIn("page &lt;&amp;&gt;.pdf", (root / "report/puzzle_1/index.html").read_text())

    def test_subset_rejects_silent_changes(self):
        rows = [{"case_id": "a", "pieces": 4}, {"case_id": "b", "pieces": 8}]
        self.assertEqual(select_subset(rows, sizes=[8]), [rows[1]])
        with self.assertRaises(ValueError):
            select_subset(rows, ids=["unknown"])
        with self.assertRaises(ValueError):
            select_subset(rows, limit=0)

    def test_resume_preserves_stage2_manifest_bytes_for_stage3_cache(self):
        from benchmark.common import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1 = root / "inputs"
            stage1.mkdir()
            (stage1 / "segmentation_manifest.jsonl").write_text('{}\n')
            manifest = {"status": "complete", "truth_free": True, "stage1_manifest_sha256": file_hash(stage1 / "segmentation_manifest.jsonl")}
            path = root / "run/stage2/edge_hypotheses_manifest.json"
            atomic_json(path, manifest)
            before = path.read_bytes()
            result = completed_stage2(root / "run", stage1)
            self.assertEqual(result["status"], "reused_complete")
            self.assertEqual(path.read_bytes(), before)
            atomic_json(root / "run/stage2_commit.json", {"manifest_sha256": file_hash(path)})
            path.write_bytes(before+b' ')
            with self.assertRaisesRegex(ValueError, "Committed Stage 2 manifest changed"):
                completed_stage2(root / "run", stage1)

    def test_archive_diagnosis_does_not_treat_missing_cache_as_missed_retrieval(self):
        from tests.fixtures import build_fixture
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = build_fixture(root, alternatives=True)
            for contact in reference["contacts"]:
                a, b = contact["piece_a"], contact["piece_b"]
                h = np.linalg.inv(reference["pieces"][a]["native_to_reference"]) @ reference["pieces"][b]["native_to_reference"]
                contact["points_a"] = map_points(contact["points_b"], h).tolist()
            diagnosis = stage3_diagnosis(root, reference)
            self.assertEqual(len(diagnosis["contacts"]), 4)
            self.assertEqual(diagnosis["archive_summary"]["archive_geometries"], 8)
            self.assertEqual(diagnosis["ranking_cache_status"], "unavailable")
            for contact in diagnosis["contacts"]:
                self.assertEqual(contact["archive_geometries"], 2)
                self.assertEqual(contact["archive_accurate"], 1)
                self.assertEqual(contact["working_accurate"], 1)
                self.assertIsNone(contact["physical_pair_retrieved"])

    def test_complete_published_prediction_is_evaluated_and_reported_separately(self):
        from tests.fixtures import build_fixture
        from paper_reconstruction_stage4.cli import main as solve_cli
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cid = "known_grid_fixture"
            output = root / "run/cases" / cid
            reference = build_fixture(output / "stage3", alternatives=True)
            stage1 = root / "inputs/cases" / cid / "stage1"
            stage1.mkdir(parents=True)
            manifest = json.loads((output / "stage3/bundle.json").read_text())
            rows = []
            for pid, p in manifest["pieces"].items():
                for kind in ("image", "mask"):
                    (stage1 / f"{pid}_{kind}.png").write_bytes((output / "stage3" / p[kind]).read_bytes())
                rows.append({"capture_id": pid, "accepted": True, "piece_path": f"{pid}_image.png", "piece_mask_path": f"{pid}_mask.png"})
            (stage1 / "segmentation_manifest.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
            for c in reference["contacts"]:
                a, b = c["piece_a"], c["piece_b"]
                h = np.linalg.inv(reference["pieces"][a]["native_to_reference"]) @ reference["pieces"][b]["native_to_reference"]
                c["points_a"] = map_points(c["points_b"], h).tolist()
            atomic_json(stage1.parent / "evaluation_only/reference_geometry.json", reference)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(solve_cli(["solve", "--bundle", str(output / "stage3"), "--output", str(output / "stage4")]), 0)
            case = {"case_id": cid, "pieces": 4, "difficulty": "simple", "path": f"cases/{cid}", "source_family": "fixture", "source_page_index": 0, "files": inventory(stage1.parent)}
            report = diagnose_case(case, root / "inputs", root / "run", {"status": "predicted"})
            self.assertEqual(report["errors"], [])
            self.assertTrue(report["stages"]["stage4"]["evaluation"]["strict_whole_page_pass"])
            self.assertEqual(report["diagnosis_counts"], {"final_contact_correct": 4})
            atomic_json(root / "run/report" / cid / "diagnosis.json", report)
            write_reports([report], root / "inputs", root / "run")
            self.assertTrue((root / "run/report/index.html").is_file())
            self.assertNotIn("evaluation", json.loads((output / "stage4/result.json").read_text()))
            (output / "stage4/preview.png").write_bytes(b"tampered")
            failed = diagnose_case(case, root / "inputs", root / "run", {"status": "predicted"})
            self.assertEqual(failed["stages"]["stage4"]["status"], "diagnostic_failed")
            self.assertNotIn("evaluation", failed["stages"]["stage4"])

    def test_runner_continues_failed_case_then_diagnoses_and_checks_resume_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cohort_root, output = root / "cohort", root / "run"
            cohort = {"benchmark_sha256": "fixed", "cases": [{"case_id": cid, "pieces": 4, "path": cid} for cid in ("first", "second")]}
            config = root / "config.json"
            config.write_text("{}")
            events = []
            def fake_commands(case, *args):
                return [(stage, [sys.executable, case["case_id"], stage]) for stage in ("stage2", "stage3", "stage4")]
            def fake_execute(command, log):
                events.append([str(c) for c in command])
                failed = command[1:] == ["first", "stage3"]
                return {"status": "failed" if failed else "complete", "returncode": 1 if failed else 0}
            patches = [patch("benchmark.runner.load_cohort", return_value=(cohort_root, cohort)),
                       patch("benchmark.runner.commands_for", side_effect=fake_commands),
                       patch("benchmark.runner.execute", side_effect=fake_execute),
                       patch("benchmark.runner.producer_identity", return_value={}),
                       patch("benchmark.runner.implementation_identity", return_value={})]
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                argv = ["--benchmark", str(cohort_root), "--output", str(output), "--config", str(config)]
                self.assertEqual(run_benchmark(argv), 1)
                self.assertNotIn([sys.executable, "first", "stage4"], events)
                self.assertIn([sys.executable, "second", "stage4"], events)
                self.assertIn("diagnose_benchmark.py", events[-1][2])
                self.assertEqual(run_benchmark(argv+["--resume"]), 1)
                config.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "Serving inputs/code/models/configuration changed"):
                    run_benchmark(argv+["--resume"])
                self.assertFalse((output / "run.lock").exists())
