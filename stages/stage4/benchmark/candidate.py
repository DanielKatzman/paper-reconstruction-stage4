"""Opt-in, fixed eight-piece exact-evidence-v1 pilot. Frozen runner stays separate."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.outputs import implementation_identity, read_result
from .common import DESKTOP, WORKSPACE, atomic_json, digest, file_hash, load_cohort, read_json, verify_files
from .runner import commands_for, completed_stage2, execute, producer_identity
from .paths import artifact_paths

CASE_ID = "puzzle_000174_0f1f6d83b42a"
COHORT_SHA = "d036dcfac8157bcb12078cfbd80389c33b6c8ad9ed7ffbf260ad2c7791127068"
CONFIG_SHA = "bb05cfd90a143ad703d35d664c848fb8df3ec38cdbcf6d9c535b2dfd0e7c7c83"
CANDIDATE_SHA = "4ec0c4f0d3c632148d63f4c7426408f4a506bca35f2e2b8db3ef3c8ad5f52f1f"
RECIPE_SHA = "364036b169e4f2e0bfc4e0f25478c6a9fc6cfbd8635d26107b617ed9e36a684f"
THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
MAPPING = {"stage3": "stage3_profile/evidence", "stage3_work": "stage3_profile/evidence.work", "stage3_profile": "stage3_profile"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def inspect_candidate():
    stage3 = DESKTOP / "paper-reconstruction-stage3"
    command = [stage3 / ".venv/Scripts/python.exe", "-B", WORKSPACE / "scripts/inspect_stage3_candidate.py", stage3]
    started, stamp = time.perf_counter(), utc()
    done = subprocess.run([str(p) for p in command], cwd=WORKSPACE, capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if done.returncode:
        raise ValueError("Candidate verification failed: " + done.stderr.strip())
    value = json.loads(done.stdout)
    manifest = value["manifest"]
    if (value.get("status") != "candidate_verified" or manifest.get("variant") != "exact-evidence-v1" or
            manifest.get("candidate_implementation_sha256") != CANDIDATE_SHA or manifest.get("manifest_sha256") != RECIPE_SHA or
            manifest.get("runtime_policy_changed") is not False or manifest.get("frozen_models_changed") is not False):
        raise ValueError("This pilot requires the qualified exact-evidence-v1 implementation and recipe")
    value["verification"] = {"command": [str(p) for p in command], "elapsed_seconds": time.perf_counter()-started,
                             "started_utc": stamp, "ended_utc": utc(), "returncode": 0, "read_only": True}
    return value


def fixed_case(benchmark):
    root, cohort = load_cohort(benchmark, verify=False)
    if cohort["benchmark_sha256"] != COHORT_SHA or cohort.get("input_adapter_version") != "observed-crop-with-4px-contour-margin.v2":
        raise ValueError("The candidate pilot requires the fixed development_v2 cohort")
    case = next((c for c in cohort["cases"] if c["case_id"] == CASE_ID), None)
    if case is None:
        raise ValueError("Declared eight-piece case is absent")
    if case["pieces"] != 8 or case["difficulty"] != "simple" or case["split"] != "development_train":
        raise ValueError("Fixed eight-piece case metadata differs")
    verify_files(root / case["path"], [f for f in case["files"] if f["path"].startswith("stage1/")])
    return root, cohort, case


def pilot_commands(case, cohort_root, run_root, candidate_root, device="cuda", cpu_threads=4):
    stage3 = DESKTOP / "paper-reconstruction-stage3"
    case_root = run_root / "cases" / case["case_id"]
    frozen = commands_for(case, cohort_root, run_root, run_root / "stage4_config.json", device, cpu_threads)
    profile = [stage3 / ".venv/Scripts/python.exe", "-B", stage3 / "scripts/profile_stage3.py",
               "--candidate-runtime", candidate_root, "--stage1", cohort_root / case["path"] / "stage1",
               "--stage2", case_root / "stage2", "--models", stage3 / "models", "--device", device,
               "--puzzle-id", case["case_id"], "--scope", "full", "--output", case_root / "stage3_profile"]
    stage4 = list(frozen[-1][1])
    stage4[stage4.index("--bundle")+1] = case_root / MAPPING["stage3"]
    return [(name, [str(p) for p in command]) for name, command in [*frozen[:2], ("stage3", profile), ("stage4", stage4)]]


def validate_profile(case_root, request):
    """A successful child exit is insufficient: require a full, sealed candidate."""
    paths = artifact_paths(case_root, {"artifact_paths": MAPPING})
    profile = read_json(paths["stage3_profile"] / "profile.json")
    req = profile["request"]
    manifest = request["candidate"]["manifest"]
    for key, expected in {"status": "profile_complete", "scope": "full", "sources_verified_unchanged": True,
                          "source_validation_status": "passed", "ready_bundle_published": True,
                          "training_performed": False, "evaluation_truth_opened": False, "runtime_policy_changed": False}.items():
        if profile.get(key) != expected:
            raise ValueError(f"Candidate profile did not complete its required checks: {key}")
    if (req.get("baseline") is not None or req.get("puzzle_id") != CASE_ID or req.get("device") != request["stage3_device"] or
            req.get("scope") != "full" or req.get("cprofile") is not True or req.get("fresh_inference_caches") is not True or
            req.get("candidate_runtime") != manifest or profile.get("comparison") != {"status": "not_requested", "equal": None}):
        raise ValueError("Candidate profile context/baseline/instrumentation differs from the fixed pilot")
    for key, expected in (("stage1", request["stage1_input"]), ("stage2", str(paths["stage2"])),
                          ("models", str(DESKTOP / "paper-reconstruction-stage3/models")), ("output", str(paths["stage3_profile"]))):
        if Path(req[key]).resolve() != Path(expected).resolve():
            raise ValueError(f"Candidate profiling request path changed: {key}")
    bundle = load_bundle(paths["stage3"], full=True)
    if (bundle.manifest["puzzle_id"] != CASE_ID or bundle.manifest["counts"]["pieces"] != 8 or
            bundle.manifest.get("provenance", {}).get("runtime_variant") != "exact-evidence-v1" or
            bundle.manifest["provenance"].get("inference_signature") != req["expected_inference_signature"] or
            req["implementation_sha256"] != CANDIDATE_SHA or
            req["model_policy_sha256"] != request["stage3_model_policy_sha256"]):
        raise ValueError("Published evidence is not bound to this candidate profile")
    return {"manifest_sha256": bundle.manifest["manifest_sha256"], "counts": bundle.manifest["counts"],
            "profile_sha256": file_hash(paths["stage3_profile"] / "profile.json"), "eight_piece_control_equivalence": "not_requested"}


def request_for(root, cohort, case, run_root, candidate, device, cpu_threads):
    config = WORKSPACE / "configs/hybrid_development.json"
    if file_hash(config) != CONFIG_SHA:
        raise ValueError("Primary eight-piece pilot must keep the qualified hybrid configuration unchanged")
    plans = pilot_commands(case, root, run_root, Path(candidate["path"]), device, cpu_threads)
    controllers = [*Path(__file__).parent.glob("*.py"), WORKSPACE / "scripts/run_candidate8.py", WORKSPACE / "scripts/inspect_stage3_candidate.py", WORKSPACE / "scripts/diagnose_benchmark.py"]
    value = {"schema_version": "stage4.candidate-pilot-request.v1", "benchmark_sha256": cohort["benchmark_sha256"],
             "benchmark_manifest": str(root / "benchmark.json"), "case_ids": [CASE_ID], "case": {k: v for k, v in case.items() if k != "files"},
             "stage1_input": str(root / case["path"] / "stage1"),
             "stage1_input_files": [f for f in case["files"] if f["path"].startswith("stage1/")],
             "stage4_configuration": read_json(config), "stage4_config_sha256": CONFIG_SHA,
             "stage3_runtime": "exact-evidence-v1", "candidate": {k: v for k, v in candidate.items() if k != "verification"},
             "stage3_device": device, "stage2_cpu_threads": cpu_threads, "stage3_environment_overrides": THREADS,
             "inherited_thread_environment": {k: os.environ.get(k) for k in THREADS},
             "stage3_model_policy_sha256": read_json(DESKTOP / "paper-reconstruction-stage3/models/alpha.json")["manifest_sha256"],
             "cprofile_enabled": True, "eight_piece_control_equivalence": "not_requested", "artifact_paths": MAPPING,
             "commands": plans, "controller_files": {str(p.relative_to(WORKSPACE)): file_hash(p) for p in sorted(controllers)},
             "producers": {"stage2": producer_identity(DESKTOP / "paper-reconstruction-stage2"),
                           "stage3": producer_identity(DESKTOP / "paper-reconstruction-stage3"), "stage4": implementation_identity()}}
    value["request_sha256"] = digest(value)
    return value


def run_serving(request, state, root, run_root, execute_fn=None):
    """Stop on first failure; profile retries use a new directory, not resume."""
    case_root = run_root / "cases" / CASE_ID
    execute_fn = execute_fn or execute
    status = state["cases"][CASE_ID]
    session = state["sessions"][-1]
    stage1 = Path(request["stage1_input"])
    for stage, command in request["commands"]:
        status.update(status="running", active_stage=stage)
        atomic_json(run_root / "progress.json", state)
        print(f"[8-piece candidate] {stage}", flush=True)
        try:
            if stage == "stage2" and (reuse := completed_stage2(case_root, stage1)) is not None:
                attempt = reuse
                session["fresh_stage2"] = False
            elif stage == "stage3" and (case_root / "stage3_profile").exists():
                try:
                    receipt = validate_profile(case_root, request)
                    marker = case_root / "stage3_commit.json"
                    if marker.is_file() and read_json(marker) != receipt:
                        raise ValueError("Committed profile/evidence receipt changed")
                except (ValueError, OSError, KeyError) as exc:
                    raise ValueError("Existing profile cannot be resumed. Preserve it and choose a fresh --output directory. " + str(exc)) from exc
                attempt = {"status": "reused_complete", "returncode": 0, **receipt}
                session["fresh_stage3"] = False
            else:
                if stage in {"stage2", "stage3"}:
                    session["fresh_"+stage] = not (case_root / ("stage3_profile" if stage == "stage3" else "stage2")).exists()
                if stage == "stage4":
                    if (file_hash(run_root / "stage4_config.json") != request["stage4_config_sha256"] or
                            implementation_identity() != request["producers"]["stage4"]):
                        raise ValueError("Stage 4 code/config changed during the pilot; use a new run")
                    session["fresh_stage4"] = not any((case_root / folder).exists() for folder in ("stage4", "stage4.work"))
                attempt = execute_fn(command, case_root / "logs" / f"{stage}.log", echo_progress=True,
                                     env_overrides=THREADS if stage == "stage3" else None)
            status["attempts"].append({"stage": stage, **attempt})
            session["stages"][stage] = attempt
            if attempt["returncode"]:
                raise ValueError(f"{stage} exited {attempt['returncode']}; inspect {case_root / 'logs' / (stage+'.log')}")
            if stage == "stage2_verify":
                atomic_json(case_root / "stage2_commit.json", {"manifest_sha256": file_hash(case_root / "stage2/edge_hypotheses_manifest.json")})
            if stage == "stage3":
                started, stamp = time.perf_counter(), utc()
                receipt = validate_profile(case_root, request)
                session["stages"]["stage3_validate"] = {"started_utc": stamp, "ended_utc": utc(), "elapsed_seconds": time.perf_counter()-started, **receipt}
                atomic_json(case_root / "stage3_commit.json", receipt)
            if stage == "stage4":
                start_validation, validation_stamp = time.perf_counter(), utc()
                result = read_result(case_root / "stage4")
                if result["input"]["manifest_sha256"] != session["stages"]["stage3_validate"]["manifest_sha256"]:
                    raise ValueError("Stage 4 result has a different evidence binding")
                session["stages"]["stage4_validate"] = {"started_utc": validation_stamp, "ended_utc": utc(),
                    "elapsed_seconds": time.perf_counter()-start_validation, "status": "validated", "run_id": result["run_id"]}
        except (ValueError, OSError, KeyError) as exc:
            status.update(status="failed", failed_stage=stage, error=str(exc))
            session.update(status="failed", failed_stage=stage, error=str(exc))
            print(str(exc), file=sys.stderr, flush=True)
            atomic_json(run_root / "progress.json", state)
            return False
        atomic_json(run_root / "progress.json", state)
    status.update(status="predicted", active_stage=None)
    status.pop("failed_stage", None)
    status.pop("error", None)
    session["status"] = "predicted"
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "outputs/candidate8_exact_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stage2-cpu-threads", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed evidence; incomplete/failed profiles require a new run directory")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args(argv)
    started, stamp = time.perf_counter(), utc()
    if args.stage2_cpu_threads < 0:
        raise ValueError("Stage 2 thread count must be nonnegative")
    root, cohort, case = fixed_case(WORKSPACE / "benchmarks/development_v2/benchmark.json")
    output = args.output.resolve()
    for protected in (root, DESKTOP / "paper-reconstruction-stage3", DESKTOP / "paper-reconstruction-stage2"):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Pilot output must not overlap immutable inputs or producers")
    if args.diagnose_only:
        request = read_json(output / "request.json")
        if request["schema_version"] != "stage4.candidate-pilot-request.v1" or request["case_ids"] != [CASE_ID] or request["benchmark_sha256"] != COHORT_SHA:
            raise ValueError("Not this fixed candidate pilot")
        candidate = None
    else:
        candidate = inspect_candidate()
        request = request_for(root, cohort, case, output, candidate, args.device, args.stage2_cpu_threads)
    if args.plan_only:
        print(json.dumps({"request": request, "read_only_candidate_verification": candidate["verification"] if candidate else None,
                          "output_created": False, "inference_performed": False, "report": str(output / "report/index.html")}, indent=2))
        return 0
    if output.exists() and not (args.resume or args.diagnose_only):
        raise ValueError("Pilot output exists; use a new directory or --resume for completed artifacts")
    if (output / "request.json").exists() and not args.diagnose_only and read_json(output / "request.json")["request_sha256"] != request["request_sha256"]:
        raise ValueError("Pilot code/models/config/inputs/environment changed; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        if not (output / "request.json").exists():
            atomic_json(output / "request.json", request)
            (output / "stage4_config.json").write_bytes((WORKSPACE / "configs/hybrid_development.json").read_bytes())
        state = read_json(output / "progress.json") if (output / "progress.json").is_file() else {
            "schema_version": "stage4.benchmark-progress.v1", "cases": {CASE_ID: {"attempts": [], "artifact_paths": MAPPING}}, "sessions": []}
        if not args.diagnose_only:
            session = {"started_utc": stamp, "status": "running", "stages": {}, "fresh_stage2": None, "fresh_stage3": None, "fresh_stage4": None,
                       "candidate_preflight": candidate["verification"], "profiling_overhead_included": True}
            state["sessions"].append(session)
            state["phase"] = "predictions"
            try:
                run_serving(request, state, root, output)
            except KeyboardInterrupt:
                state["cases"][CASE_ID]["status"] = "interrupted"
                state["cases"][CASE_ID]["failed_stage"] = state["cases"][CASE_ID].get("active_stage")
                session["status"] = "interrupted"
                active = state["cases"][CASE_ID].get("active_stage")
                partial = output / "cases" / CASE_ID / "logs" / f"{active}.attempt.json"
                if partial.is_file() and (record := read_json(partial)).get("status") == "interrupted":
                    session["stages"][active] = record
                    state["cases"][CASE_ID]["attempts"].append({"stage": active, **record})
            session.update(ended_utc=utc(), serving_wall_seconds=time.perf_counter()-started)
            session["fresh_end_to_end_measurement"] = session["status"] == "predicted" and all(session[k] for k in ("fresh_stage2", "fresh_stage3", "fresh_stage4"))
            if session["status"] == "interrupted":
                state["phase"] = "interrupted"
                atomic_json(output / "progress.json", state)
                print("Interrupted. No offline evaluation was started. Preserve the profile and use a fresh output for a cold retry.", flush=True)
                return 130
        state["phase"] = "diagnostics"
        atomic_json(output / "progress.json", state)
        command = [sys.executable, "-B", WORKSPACE / "scripts/diagnose_benchmark.py", "--benchmark", root / "benchmark.json", "--run", output]
        diagnostic = execute(command, output / "logs/diagnostics.log", echo_progress=True)
        state["diagnostics"] = diagnostic
        state["phase"] = "finished" if not diagnostic["returncode"] else "diagnostics_failed"
        atomic_json(output / "progress.json", state)
        from .feedback import write_feedback
        write_feedback(output)
        print(f"Primary report: {output / 'report/index.html'}", flush=True)
        print(f"Stage 3 feedback: {output / 'STAGE4_TO_STAGE3_CANDIDATE_FEEDBACK.md'}", flush=True)
        if state["cases"][CASE_ID]["status"] == "interrupted":
            return 130
        return 1 if diagnostic["returncode"] or state["cases"][CASE_ID]["status"] != "predicted" else 0
    finally:
        lock.unlink(missing_ok=True)
