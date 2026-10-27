"""Serial accelerated pipeline with frozen consumer, resumable evidence and phase barrier."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.outputs import read_result
from .candidate import inspect_candidate, THREADS
from .common import DESKTOP, WORKSPACE, atomic_json, digest, file_hash, load_cohort, read_json, verify_files
from .runner import commands_for, completed_stage2, execute, producer_identity, select_subset


def check_profile(folder, case, candidate, stage1, stage2):
    profile = read_json(folder / "profile.json")
    req = profile["request"]
    checks = {"status": "profile_complete", "scope": "full", "sources_verified_unchanged": True,
              "source_validation_status": "passed", "ready_bundle_published": True,
              "training_performed": False, "evaluation_truth_opened": False, "runtime_policy_changed": False}
    if any(profile.get(k) != v for k, v in checks.items()):
        raise ValueError("Incomplete or unverified Stage 3 profile")
    if (req["candidate_runtime"] != candidate["manifest"] or req["puzzle_id"] != case["case_id"] or
            req["cprofile"] is not False or req["baseline"] is not None or
            Path(req["stage1"]).resolve() != stage1.resolve() or Path(req["stage2"]).resolve() != stage2.resolve() or
            profile["comparison"] != {"status": "not_requested", "equal": None}):
        raise ValueError("Stage 3 profile context differs")
    bundle = load_bundle(folder / "evidence", full=True)
    if (bundle.manifest["puzzle_id"] != case["case_id"] or bundle.manifest["counts"]["pieces"] != case["pieces"] or
            bundle.manifest["provenance"].get("inference_signature") != req["expected_inference_signature"]):
        raise ValueError("Stage 3 bundle identity differs")
    return bundle.manifest["manifest_sha256"]


def plan(case, inputs, output, candidate, config, snapshot, pool_hash, profile_name="stage3_profile"):
    upstream = DESKTOP / "paper-reconstruction-stage3"
    frozen = commands_for(case, inputs, output, config, "cuda", 4)
    target = output / "cases" / case["case_id"]
    stage3 = [upstream / ".venv/Scripts/python.exe", "-B", upstream / "scripts/profile_stage3.py",
              "--candidate-runtime", candidate["path"], "--stage1", inputs / case["path"] / "stage1",
              "--stage2", target / "stage2", "--models", upstream / "models", "--device", "cuda",
              "--puzzle-id", case["case_id"], "--scope", "full", "--no-cprofile", "--output", target / profile_name]
    pool = [sys.executable, "-B", WORKSPACE / "scripts/replay_observable_pool.py", "--stage3-root", upstream,
            "--source", target / profile_name / "evidence", "--output", target / "stage3",
            "--expected-script-sha256", pool_hash]
    consumer = [sys.executable, "-B", WORKSPACE / "scripts/run_source_snapshot.py", snapshot, "solve",
                "--bundle", target / "stage3", "--config", config, "--output", target / "stage4", "--resume"]
    return [(k, [str(p) for p in v]) for k, v in [*frozen[:2], ("stage3", stage3), ("pool", pool), ("stage4", consumer)]]


def safe_output(output, protected):
    for path in protected:
        path = Path(path).resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Output overlaps protected input: " + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=WORKSPACE / "benchmarks/validation8x10_scale12_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--case-id", action="append", help="Optional declared case IDs within the selected sizes")
    parser.add_argument("--snapshot", type=Path, default=WORKSPACE / "controls/stage4_joint_v4_validation_baseline")
    parser.add_argument("--config-name", choices=("joint_robust_development.json", "joint_refine_development.json", "joint_coarse_fast_development.json"), default="joint_robust_development.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args(argv)
    root, cohort = load_cohort(args.benchmark, serving_only=True)
    selected = select_subset(cohort["cases"], args.sizes, args.case_id)
    output, snapshot = args.output.resolve(), args.snapshot.resolve()
    safe_output(output, [root, snapshot, WORKSPACE / "src", WORKSPACE / "scripts", WORKSPACE / "benchmark",
                         *[DESKTOP / x for x in ("paper-reconstruction-stage2", "paper-reconstruction-stage3", "paper-segmentation-stage")]])
    receipt = read_json(snapshot / "receipt.json")
    for name, sha in receipt["files"].items():
        if file_hash(snapshot / name) != sha:
            raise ValueError("Consumer snapshot changed")
    source_config = snapshot / "configs" / args.config_name
    config = output / "stage4_config.json"
    candidate = inspect_candidate()
    upstream = DESKTOP / "paper-reconstruction-stage3"
    pool_hash = file_hash(upstream / "scripts/stage3_pool_replay_tools.py")
    request = {"schema_version": "stage4.validation-run.v1", "benchmark_sha256": cohort["benchmark_sha256"],
        "benchmark_manifest": str(root / "benchmark.json"), "case_ids": [c["case_id"] for c in selected],
        "stage4_config_sha256": file_hash(source_config), "consumer_snapshot": str(snapshot),
        "consumer_implementation": receipt["implementation"], "candidate": {k: candidate[k] for k in ("path", "manifest", "environment")},
        "pool_script_sha256": pool_hash, "stage3_equivalence": "not_requested_for_new_cases",
        "timing_scope": "Stage1 bypass; fresh serial Stage2/accelerated Stage3 plus observable archive pool replay and frozen Stage4; wall instrumentation enabled, cProfile disabled. Resumed stages separately marked reused.",
        "controller_files": {str(p.relative_to(WORKSPACE)): file_hash(p) for p in
            [Path(__file__), WORKSPACE / "benchmark/runner.py", WORKSPACE / "scripts/run_source_snapshot.py", WORKSPACE / "scripts/replay_observable_pool.py"]},
        "producers": {"stage2": producer_identity(DESKTOP / "paper-reconstruction-stage2"), "stage3": producer_identity(upstream)},
        "commands": {c["case_id"]: plan(c, root, output, candidate, config, snapshot, pool_hash) for c in selected}}
    request["request_sha256"] = digest(request)
    if args.plan_only:
        print(__import__("json").dumps(request, indent=2))
        return 0
    if output.exists() and not (args.resume or args.diagnose_only):
        raise ValueError("Existing output; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    state_path = output / "progress.json"
    state = read_json(state_path) if state_path.exists() else {"cases": {}, "phase": "predictions"}
    try:
        old = output / "request.json"
        if old.exists():
            prior = read_json(old)
            if prior.get("request_sha256") != digest({k: v for k, v in prior.items() if k != "request_sha256"}):
                raise ValueError("Request checksum mismatch")
            if prior["benchmark_sha256"] != cohort["benchmark_sha256"] or prior["case_ids"] != [c["case_id"] for c in selected]:
                raise ValueError("Original cohort and case subset required")
            if not args.diagnose_only and prior["request_sha256"] != request["request_sha256"]:
                raise ValueError("Run inputs/code/models changed; use new output")
            request = prior
        else:
            if args.diagnose_only:
                raise ValueError("No prediction request")
            atomic_json(old, request)
            shutil.copyfile(source_config, config)
        if file_hash(config) != request["stage4_config_sha256"]:
            raise ValueError("Config changed")
        if not args.diagnose_only:
            state["phase"] = "predictions"
            atomic_json(state_path, state)
            for index, case in enumerate(selected, 1):
                cid = case["case_id"]
                case_root = output / "cases" / cid
                status = state["cases"].setdefault(cid, {"attempts": []})
                status.update(status="running")
                started = time.perf_counter()
                profile_name = status.get("profile_name", "stage3_profile")
                profile = case_root / profile_name
                if profile.exists() and not (profile / "evidence/bundle.json").is_file():
                    # A failed profiling attempt is immutable; next retry gets a new path.
                    profile_name = "stage3_profile_retry_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                status.update(profile_name=profile_name, artifact_paths={"stage3_profile": profile_name,
                    "stage3_work": profile_name + "/evidence.work"})
                plans = plan(case, root, output, candidate, config, snapshot, pool_hash, profile_name)
                for stage, command in plans:
                    print(f"[{index}/{len(selected)}] {cid} ({case['pieces']} pieces): {stage}", flush=True)
                    status["active_stage"] = stage
                    atomic_json(state_path, state)
                    try:
                        reuse = completed_stage2(case_root, root / case["path"] / "stage1") if stage == "stage2" else None
                        if stage == "stage3" and (case_root / profile_name / "evidence/bundle.json").is_file():
                            check_profile(case_root / profile_name, case, candidate, root / case["path"] / "stage1", case_root / "stage2")
                            reuse = {"status": "reused", "returncode": 0, "elapsed_seconds": 0., "fresh_inference": False}
                        if stage == "pool" and (case_root / "stage3/bundle.json").is_file():
                            bundle = load_bundle(case_root / "stage3", full=True)
                            source = load_bundle(case_root / profile_name / "evidence", full=True)
                            if (bundle.manifest["provenance"]["source_manifest_sha256"] != source.manifest["manifest_sha256"] or
                                    bundle.manifest["provenance"]["implementation"]["stage3_pool_replay_tools.py"] != pool_hash):
                                raise ValueError("Reused pool source/policy differs")
                            reuse = {"status": "reused", "returncode": 0, "elapsed_seconds": 0.}
                        if stage == "stage4" and (case_root / "stage4/manifest.json").is_file():
                            result = read_result(case_root / "stage4")
                            if (result["implementation"] != receipt["implementation"] or
                                    result["input"]["manifest_sha256"] != load_bundle(case_root / "stage3").manifest["manifest_sha256"]):
                                raise ValueError("Reused consumer identity differs")
                            reuse = {"status": "reused", "returncode": 0, "elapsed_seconds": 0.}
                        if reuse is not None:
                            attempt = reuse
                        else:
                            if stage in {"stage2", "stage3"}:
                                key = "stage2" if stage == "stage2" else "stage3"
                                if producer_identity(Path(request["producers"][key]["root"])) != request["producers"][key]:
                                    raise ValueError("Upstream code/models changed during run")
                            verify_files(root / case["path"], [f for f in case["files"] if f["path"].startswith("stage1/")])
                            attempt = execute(command, case_root / "logs" / f"{stage}.log", env_overrides=THREADS, echo_progress=True)
                        if attempt["returncode"] == 0:
                            if stage == "stage2_verify":
                                atomic_json(case_root / "stage2_commit.json", {"manifest_sha256": file_hash(case_root / "stage2/edge_hypotheses_manifest.json")})
                            elif stage == "stage3":
                                check_profile(case_root / profile_name, case, candidate, root / case["path"] / "stage1", case_root / "stage2")
                    except (ValueError, OSError, KeyError) as exc:
                        attempt = {"status": "failed", "returncode": 2, "error": str(exc)}
                    status["attempts"].append({"stage": stage, **attempt})
                    if attempt["returncode"]:
                        status.update(status="failed", failed_stage=stage)
                        break
                else:
                    status.update(status="predicted", active_stage=None)
                status["latest_attempt_wall_seconds"] = time.perf_counter() - started
                atomic_json(state_path, state)
        state["phase"] = "diagnostics"
        atomic_json(state_path, state)
        command = [sys.executable, "-B", WORKSPACE / "scripts/diagnose_benchmark.py", "--benchmark", root, "--run", output]
        state["diagnostics"] = execute(command, output / "logs/diagnostics.log", env_overrides=THREADS, echo_progress=True)
        state["phase"] = "finished" if state["diagnostics"]["returncode"] == 0 else "diagnostics_failed"
        atomic_json(state_path, state)
        print("Report:", output / "report/index.html", flush=True)
        return int(state["diagnostics"]["returncode"] != 0 or any(c["status"] != "predicted" for c in state["cases"].values()))
    except KeyboardInterrupt:
        state["phase"] = "interrupted"
        atomic_json(state_path, state)
        return 130
    finally:
        lock.unlink(missing_ok=True)
