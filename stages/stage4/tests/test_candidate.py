import contextlib
import copy
from html.parser import HTMLParser
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote
import numpy as np
from PIL import Image
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.cli import main as solve_cli
from benchmark.common import atomic_json, digest, file_hash, inventory, read_json, WORKSPACE
from benchmark.candidate import CASE_ID, COHORT_SHA, MAPPING, THREADS, CANDIDATE_SHA, RECIPE_SHA, pilot_commands, validate_profile, run_serving, fixed_case, inspect_candidate, request_for, main as pilot_main
from benchmark.runner import commands_for, execute
from benchmark.paths import artifact_paths
from benchmark.diagnostics import diagnose_case
from benchmark.quality import assembly_quality
from benchmark.feedback import write_feedback
from benchmark.report import write_reports
from evaluation.evaluate import evaluate
from tests.fixtures import build_fixture


class CandidateTests(unittest.TestCase):
    def manifest(self):
        return {"variant": "exact-evidence-v1", "candidate_implementation_sha256": CANDIDATE_SHA, "manifest_sha256": RECIPE_SHA,
                "runtime_policy_changed": False, "frozen_models_changed": False}

    def profile_context(self, root):
        case_root = root / "cases" / CASE_ID
        paths = artifact_paths(case_root, {"artifact_paths": MAPPING})
        request = {"candidate": {"manifest": self.manifest()}, "stage3_device": "cuda", "stage1_input": str(root / "inputs"),
                   "stage3_model_policy_sha256": "models-policy"}
        pr = {"baseline": None, "puzzle_id": CASE_ID, "device": "cuda", "scope": "full", "cprofile": True,
              "fresh_inference_caches": True, "candidate_runtime": self.manifest(), "stage1": request["stage1_input"],
              "stage2": str(paths["stage2"]), "models": str(WORKSPACE.parent / "paper-reconstruction-stage3/models"),
              "output": str(paths["stage3_profile"]), "expected_inference_signature": "signature", "implementation_sha256": CANDIDATE_SHA,
              "model_policy_sha256": "models-policy"}
        profile = {"status": "profile_complete", "scope": "full", "sources_verified_unchanged": True,
                   "source_validation_status": "passed", "ready_bundle_published": True, "training_performed": False,
                   "evaluation_truth_opened": False, "runtime_policy_changed": False, "request": pr,
                   "comparison": {"status": "not_requested", "equal": None}}
        bundle = SimpleNamespace(manifest={"puzzle_id": CASE_ID, "counts": {"pieces": 8}, "manifest_sha256": "ready",
                                          "provenance": {"runtime_variant": "exact-evidence-v1", "inference_signature": "signature"}})
        atomic_json(paths["stage3_profile"] / "profile.json", profile)
        return case_root, paths, request, profile, bundle

    def test_explicit_route_uses_full_profile_without_four_piece_preset_or_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = {"case_id": CASE_ID, "path": "cases/"+CASE_ID}
            commands = dict(pilot_commands(case, root / "cohort", root / "run", root / "candidate"))
            cmd = commands["stage3"]
            self.assertTrue(any(p.endswith("profile_stage3.py") for p in cmd))
            self.assertEqual(cmd[cmd.index("--scope")+1], "full")
            self.assertEqual(cmd[cmd.index("--puzzle-id")+1], CASE_ID)
            for forbidden in ("--preset", "--baseline", "--resume", "--no-cprofile"):
                self.assertNotIn(forbidden, cmd)
            stage4 = commands["stage4"]
            self.assertEqual(Path(stage4[stage4.index("--bundle")+1]), root / "run/cases" / CASE_ID / "stage3_profile/evidence")
            frozen = dict(commands_for(case, root / "cohort", root / "control", root / "config.json"))["stage3"]
            self.assertIn("infer", frozen)
            self.assertFalse(any(str(x).endswith("profile_stage3.py") for x in frozen))

    def test_fixed_case_checks_only_serving_receipts_and_rejects_other_cohort(self):
        case = {"case_id": CASE_ID, "pieces": 8, "difficulty": "simple", "split": "development_train", "path": "case",
                "files": [{"path": "stage1/image.png"}, {"path": "evaluation_only/reference.json"}]}
        cohort = {"benchmark_sha256": COHORT_SHA, "input_adapter_version": "observed-crop-with-4px-contour-margin.v2", "cases": [case]}
        with patch("benchmark.candidate.load_cohort", return_value=(Path("inputs"), cohort)), patch("benchmark.candidate.verify_files") as verify:
            _, _, selected = fixed_case("unused")
            self.assertEqual(selected["case_id"], CASE_ID)
            self.assertEqual(verify.call_args.args[1], [{"path": "stage1/image.png"}])
            cohort["benchmark_sha256"] = "different"
            with self.assertRaises(ValueError):
                fixed_case("unused")

    def test_candidate_preflight_rejects_recipe_identity_drift(self):
        payload = {"status": "candidate_verified", "path": "snapshot", "manifest": self.manifest()}
        with patch("benchmark.candidate.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")):
            self.assertEqual(inspect_candidate()["manifest"]["candidate_implementation_sha256"], CANDIDATE_SHA)
        payload["manifest"]["manifest_sha256"] = "changed"
        with patch("benchmark.candidate.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")):
            with self.assertRaises(ValueError):
                inspect_candidate()

    def test_request_identity_pins_candidate_recipe_threads_config_and_sources(self):
        root = WORKSPACE / "outputs/not-created-request-test"
        case = {"case_id": CASE_ID, "path": "cases/"+CASE_ID, "pieces": 8, "files": [{"path": "stage1/segmentation_manifest.jsonl", "sha256": "input"}]}
        candidate = {"path": "verified_snapshot", "manifest": self.manifest(), "verification": {"elapsed_seconds": 1.}}
        with patch("benchmark.candidate.producer_identity", return_value={"sha256": "upstream"}), patch("benchmark.candidate.implementation_identity", return_value={"sha256": "solver"}):
            first = request_for(Path("cohort"), {"benchmark_sha256": COHORT_SHA}, case, root, candidate, "cuda", 4)
            second = request_for(Path("cohort"), {"benchmark_sha256": COHORT_SHA}, case, root, {**candidate, "verification": {"elapsed_seconds": 9.}}, "cuda", 4)
            self.assertEqual(first["request_sha256"], second["request_sha256"])
            changed = request_for(Path("cohort"), {"benchmark_sha256": COHORT_SHA}, case, root, candidate, "cpu", 4)
            self.assertNotEqual(first["request_sha256"], changed["request_sha256"])
            altered = copy.deepcopy(candidate)
            altered["manifest"]["manifest_sha256"] = "new-recipe"
            changed = request_for(Path("cohort"), {"benchmark_sha256": COHORT_SHA}, case, root, altered, "cuda", 4)
            self.assertNotEqual(first["request_sha256"], changed["request_sha256"])
            self.assertEqual(first["stage3_environment_overrides"], THREADS)
            self.assertEqual(first["case_ids"], [CASE_ID])
            self.assertTrue(first["controller_files"])
            self.assertFalse(root.exists())

    def test_zero_exit_cannot_bypass_profile_completion_scope_and_no_baseline_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, paths, request, profile, bundle = self.profile_context(Path(tmp))
            with patch("benchmark.candidate.load_bundle", return_value=bundle):
                receipt = validate_profile(case, request)
                self.assertEqual(receipt["eight_piece_control_equivalence"], "not_requested")
                for key, invalid in (("status", "interrupted"), ("scope", "first-pair"), ("sources_verified_unchanged", False), ("ready_bundle_published", False)):
                    bad = {**profile, key: invalid}
                    atomic_json(paths["stage3_profile"] / "profile.json", bad)
                    with self.assertRaises(ValueError):
                        validate_profile(case, request)
                bad = copy.deepcopy(profile)
                bad["comparison"] = {"status": "passed", "equal": True}
                atomic_json(paths["stage3_profile"] / "profile.json", bad)
                with self.assertRaises(ValueError):
                    validate_profile(case, request)
                bad = copy.deepcopy(profile)
                bad["request"]["baseline"] = "a_four_piece_baseline"
                atomic_json(paths["stage3_profile"] / "profile.json", bad)
                with self.assertRaises(ValueError):
                    validate_profile(case, request)

    def test_mapping_rejects_escape_and_keeps_real_work_location(self):
        root = Path(tempfile.gettempdir()) / "case"
        mapped = artifact_paths(root, {"artifact_paths": MAPPING})
        self.assertEqual(mapped["stage3_work"], root.resolve() / "stage3_profile/evidence.work")
        for bad in ("../other", str(root.parent), "./"):
            with self.assertRaises(ValueError):
                artifact_paths(root, {"artifact_paths": {"stage3": bad}})

    def test_profile_failure_stops_before_stage4_and_never_replays_partial_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, paths, request, profile, _ = self.profile_context(root)
            # Reusing an interrupted report must not call the profiler or solver.
            profile["status"] = "interrupted"
            atomic_json(paths["stage3_profile"] / "profile.json", profile)
            request["commands"] = [("stage3", ["profile"]), ("stage4", ["solve"])]
            state = {"cases": {CASE_ID: {"attempts": []}}, "sessions": [{"stages": {}, "fresh_stage3": True}]}
            calls = []
            with contextlib.redirect_stderr(io.StringIO()):
                ok = run_serving(request, state, root, root, execute_fn=lambda *a, **k: calls.append(a))
            self.assertFalse(ok)
            self.assertEqual(calls, [])
            self.assertEqual(state["cases"][CASE_ID]["failed_stage"], "stage3")
            self.assertIn("fresh --output", state["cases"][CASE_ID]["error"])

    def test_successful_child_with_invalid_profile_still_blocks_stage4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "cases" / CASE_ID
            request = {"stage1_input": "inputs", "commands": [("stage3", ["profile"]), ("stage4", ["solve"])]}
            state = {"cases": {CASE_ID: {"attempts": []}}, "sessions": [{"stages": {}}]}
            calls = []
            def fake_execute(command, log, **kwargs):
                calls.append(command)
                atomic_json(case / "stage3_profile/profile.json", {"status": "profile_complete", "scope": "first-pair", "request": {}})
                return {"returncode": 0, "status": "complete"}
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(run_serving(request, state, root, root, execute_fn=fake_execute))
            self.assertEqual(calls, [["profile"]])
            self.assertEqual(state["cases"][CASE_ID]["failed_stage"], "stage3")

    def test_live_progress_and_child_environment_are_visible_and_logged(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stage3.log"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = execute([sys.executable, "-c", "import os;print('[profile +1s] '+os.environ['CANDIDATE_TEST_VALUE'],flush=True)"], log,
                                 env_overrides={"CANDIDATE_TEST_VALUE": "alive"}, echo_progress=True)
            self.assertEqual(result["returncode"], 0)
            self.assertIn("[profile +1s] alive", output.getvalue())
            self.assertIn("alive", log.read_text())
            self.assertIn("ended_utc", result)

    def test_completed_stage2_and_profile_reuse_still_verify_before_stage4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, paths, request, _, bundle = self.profile_context(root)
            stage1 = Path(request["stage1_input"])
            stage1.mkdir(parents=True)
            (stage1 / "segmentation_manifest.jsonl").write_text("fixture")
            atomic_json(paths["stage2"] / "edge_hypotheses_manifest.json", {"status": "complete", "truth_free": True,
                "stage1_manifest_sha256": file_hash(stage1 / "segmentation_manifest.jsonl")})
            before = (paths["stage2"] / "edge_hypotheses_manifest.json").read_bytes()
            atomic_json(root / "stage4_config.json", {"unchanged": True})
            request.update(stage4_config_sha256=file_hash(root / "stage4_config.json"), producers={"stage4": "solver"},
                commands=[("stage2", ["inference"]), ("stage2_verify", ["verify"]), ("stage3", ["profile"]), ("stage4", ["solve"])])
            state = {"cases": {CASE_ID: {"attempts": []}}, "sessions": [{"stages": {}, "fresh_stage2": None, "fresh_stage3": None, "fresh_stage4": None}]}
            calls = []
            def fake_execute(command, log, **kwargs):
                calls.append(command)
                return {"returncode": 0, "status": "complete", "elapsed_seconds": .1}
            with patch("benchmark.candidate.load_bundle", return_value=bundle), patch("benchmark.candidate.implementation_identity", return_value="solver"), patch("benchmark.candidate.read_result", return_value={"input": {"manifest_sha256": "ready"}, "run_id": "published"}):
                self.assertTrue(run_serving(request, state, root, root, execute_fn=fake_execute))
            self.assertEqual(calls, [["verify"], ["solve"]])
            self.assertEqual((paths["stage2"] / "edge_hypotheses_manifest.json").read_bytes(), before)
            session = state["sessions"][-1]
            self.assertIs(session["fresh_stage2"], False)
            self.assertIs(session["fresh_stage3"], False)
            self.assertIs(session["fresh_stage4"], True)
            self.assertIn("stage3_validate", session["stages"])
            self.assertEqual(session["stages"]["stage4_validate"]["run_id"], "published")

    def test_interruption_does_not_start_offline_diagnosis_or_mark_unrun_stages_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "inputs", Path(tmp) / "run"
            request = {"schema_version": "stage4.candidate-pilot-request.v1", "request_sha256": "request"}
            with patch("benchmark.candidate.fixed_case", return_value=(root, {}, {})), patch("benchmark.candidate.inspect_candidate", return_value={"verification": {}}), patch("benchmark.candidate.request_for", return_value=request), patch("benchmark.candidate.run_serving", side_effect=KeyboardInterrupt), patch("benchmark.candidate.execute") as execute_child:
                self.assertEqual(pilot_main(["--output", str(output)]), 130)
            execute_child.assert_not_called()
            progress = read_json(output / "progress.json")
            self.assertEqual(progress["phase"], "interrupted")
            self.assertFalse(progress["sessions"][-1]["fresh_end_to_end_measurement"])
            self.assertIsNone(progress["sessions"][-1]["fresh_stage4"])
            self.assertFalse((output / "run.lock").exists())

    def make_published_fixture(self, root):
        cid = "fixture"
        case_root = root / "run/cases" / cid
        stage3 = case_root / "stage3_profile/evidence"
        reference = build_fixture(stage3, rows=2, cols=4)
        manifest = read_json(stage3 / "bundle.json")
        manifest["provenance"].update(runtime_variant="exact-evidence-v1", inference_signature="sig", search=[{"pieces": [c["piece_a"], c["piece_b"]]} for c in reference["contacts"]])
        manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        atomic_json(stage3 / "bundle.json", manifest)
        reference["input_manifest_sha256"] = manifest["manifest_sha256"]
        reference.update(reference_size_hw=[64, 128], seams=[], borders=[])
        stage1 = root / "inputs/cases" / cid / "stage1"
        stage1.mkdir(parents=True)
        rows = []
        for pid, p in manifest["pieces"].items():
            for kind in ("image", "mask"):
                (stage1 / f"{pid}_{kind}.png").write_bytes((stage3 / p[kind]).read_bytes())
            rows.append({"capture_id": pid, "accepted": True, "piece_path": f"{pid}_image.png", "piece_mask_path": f"{pid}_mask.png"})
        (stage1 / "segmentation_manifest.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for c in reference["contacts"]:
            a, b = c["piece_a"], c["piece_b"]
            h = np.linalg.inv(reference["pieces"][a]["native_to_reference"]) @ reference["pieces"][b]["native_to_reference"]
            c["points_a"] = map_points(c["points_b"], h).tolist()
            reference["seams"].append({"piece_a": a, "piece_b": b, "seam_id": a+b,
                                      "points_reference": map_points(c["points_b"], reference["pieces"][b]["native_to_reference"]).tolist()})
        atomic_json(stage1.parent / "evaluation_only/reference_geometry.json", reference)
        Image.new("RGB", (128, 64), "white").save(stage1.parent / "evaluation_only/reference_page.jpg")
        indices = {p: next(i for i, c in enumerate(manifest["candidates"]) if c["piece_id"] == p) for p in manifest["pieces"]}
        ranking = [{"query": indices[c["piece_a"]], "target": indices[c["piece_b"]]} for c in reference["contacts"]]
        atomic_json(case_root / "stage3_profile/evidence.work/rankings.json", {"signature": "sig", "rows": ranking, "payload_sha256": digest(ranking)})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(solve_cli(["solve", "--bundle", str(stage3), "--output", str(case_root / "stage4")]), 0)
        case = {"case_id": cid, "path": "cases/fixture", "pieces": 8, "difficulty": "simple", "source_family": "fixture", "source_page_index": 0, "files": inventory(stage1.parent)}
        return case, case_root, reference

    def test_profile_evidence_and_work_are_used_by_real_sdk_diagnostics_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, output, reference = self.make_published_fixture(root)
            # A canonical-looking directory must not be read instead of the mapped source.
            atomic_json(output / "stage3/bundle.json", {"invalid": True})
            diagnostic = diagnose_case(case, root / "inputs", root / "run", {"status": "predicted", "artifact_paths": MAPPING})
            self.assertEqual(diagnostic["errors"], [])
            self.assertEqual(diagnostic["stages"]["stage3"]["ranking_cache_status"], "available_signature_bound")
            self.assertTrue(diagnostic["stages"]["stage4"]["evaluation"]["strict_whole_page_pass"])
            self.assertTrue(any("stage3_profile/evidence.work/rankings.json" == r["path"] for r in diagnostic["logs_and_work_receipts"]))
            self.assertTrue((root / "run/report/fixture/placement_overlay.png").is_file())
            quality = diagnostic["stages"]["stage4"]["assembly_quality"]
            self.assertIsNone(quality["structural_visual_assessment"])
            self.assertEqual(quality["selected_neighbor_identity_precision"], 1)
            atomic_json(root / "run/report/fixture/diagnosis.json", diagnostic)
            atomic_json(root / "run/request.json", {})
            atomic_json(root / "run/progress.json", {})
            write_reports([diagnostic], root / "inputs", root / "run")
            class Links(HTMLParser):
                def __init__(self): super().__init__(); self.paths = []
                def handle_starttag(self, tag, attrs): self.paths.extend(v for k, v in attrs if k in ("src", "href"))
            for path in (root / "run/report").rglob("*.html"):
                parser = Links()
                parser.feed(path.read_text(encoding="utf-8"))
                for href in parser.paths:
                    self.assertTrue((path.parent / unquote(href)).is_file(), href)

    def test_common_gauge_does_not_align_each_piece_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, output, reference = self.make_published_fixture(root)
            result = read_json(output / "stage4/result.json")
            gauge = np.array([[0., -1., 400.], [1., 0., -100.], [0., 0., 1.]])
            for p in result["layouts"][0]["pieces"]:
                p["native_to_component"] = (gauge @ p["native_to_component"]).tolist()
            metric = assembly_quality(result, reference, evaluate(result, reference))
            self.assertLess(metric["common_gauge"]["non_anchor_probe_errors_reference_px"]["maximum"], 1e-9)
            piece = result["layouts"][0]["pieces"][-1]
            shifted = np.array(piece["native_to_component"])
            shifted[1, 2] += 50
            piece["native_to_component"] = shifted.tolist()
            metric = assembly_quality(result, reference, evaluate(result, reference))
            self.assertGreater(metric["common_gauge"]["non_anchor_probe_errors_reference_px"]["maximum"], 49)
            self.assertIsNone(metric["structural_visual_assessment"])

    def test_feedback_keeps_unavailable_measurements_null_and_nested_times_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, output, reference = self.make_published_fixture(root)
            # Reuse the real eight-piece fixture without claiming a production experiment.
            diag = diagnose_case(case, root / "inputs", root / "run", {"status": "predicted", "artifact_paths": MAPPING})
            atomic_json(root / "run/report/fixture/diagnosis.json", diag)
            request = {"case_ids": ["fixture"], "case": case, "benchmark_sha256": "cohort", "request_sha256": "request",
                       "candidate": {"manifest": self.manifest()}, "stage3_model_policy_sha256": "policy", "stage3_device": "cuda",
                       "stage3_environment_overrides": THREADS, "inherited_thread_environment": {}, "producers": {},
                       "stage4_configuration": {}, "stage4_config_sha256": "config", "stage1_input_files": [], "stage1_input": str(root / "inputs/cases/fixture/stage1"),
                       "cprofile_enabled": True, "commands": []}
            atomic_json(root / "run/request.json", request)
            atomic_json(root / "run/progress.json", {"phase": "finished", "cases": {"fixture": {"status": "predicted", "artifact_paths": MAPPING}},
                "sessions": [{"stages": {}, "serving_wall_seconds": 20., "status": "predicted", "fresh_end_to_end_measurement": True}], "diagnostics": {"elapsed_seconds": 3}})
            atomic_json(output / "stage3_profile/profile.json", {"status": "profile_complete", "comparison": {"status": "not_requested", "equal": None},
                "functions": [{"function": "parent", "calls": 1, "inclusive_seconds": 10., "exclusive_seconds": 2.},
                              {"function": "child", "calls": 1, "inclusive_seconds": 8., "exclusive_seconds": 8.}], "timing": {"inference_wall_seconds": 10.}})
            feedback = write_feedback(root / "run")
            self.assertEqual(feedback["eight_piece_control_equivalence"], "not_requested")
            self.assertEqual(feedback["timing"]["fresh_end_to_end_wall_seconds"], 20.)
            self.assertIsNone(feedback["timing"]["stage4_render_seal_seconds"]["value"])
            self.assertIsNone(feedback["manual_review"]["correct_relative_placement_and_orientation"])
            self.assertTrue((root / "run/STAGE4_TO_STAGE3_CANDIDATE_FEEDBACK.md").is_file())
            atomic_json(output / "stage3_profile/profile.json", {"status": "failed", "comparison": {"status": "not_run", "equal": None}})
            failed = write_feedback(root / "run")
            self.assertEqual(failed["eight_piece_control_equivalence"], "not_requested")
            self.assertEqual(failed["profile_comparison"]["status"], "not_run")
