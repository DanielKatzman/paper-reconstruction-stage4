"""Serial, resumable Stage 2 -> 3 -> 4 benchmark. No truth is read here.

Evaluation is a separate child process, launched after the prediction phase.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import DESKTOP, WORKSPACE, atomic_json, digest, file_hash, load_cohort, read_json
from paper_reconstruction_stage4.outputs import implementation_identity


def producer_identity(root):
    files = {}
    for folder in ("src", "scripts", "models"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".log"}:
                files[path.relative_to(root).as_posix()] = file_hash(path)
    return {"root": str(root), "sha256": digest(files), "files": files}


def execute(command, log, cwd=WORKSPACE, env_overrides=None, echo_progress=False):
    """Persist failures and interruption; never construct a shell command."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {"command": [str(c) for c in command], "started_utc": datetime.now(timezone.utc).isoformat(), "log": str(log),
              "environment_overrides": dict(env_overrides or {})}
    with log.open("ab") as stream:
        stream.write(("\n" + record["started_utc"] + " " + repr(record["command"]) + "\n").encode("utf-8"))
        stream.flush()
        try:
            process = subprocess.Popen(record["command"], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", **(env_overrides or {})})
        except OSError as exc:
            record.update(status="failed", returncode=127, error=str(exc), elapsed_seconds=time.perf_counter()-started,
                          ended_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(log.with_suffix(".attempt.json"), record)
            return record
        try:
            if not echo_progress:
                code = process.wait()
            else:
                last_echo = time.perf_counter()
                with log.open("r", encoding="utf-8", errors="replace") as tail:
                    pending = ""
                    while True:
                        try:
                            code = process.wait(timeout=1)
                            finished = True
                        except subprocess.TimeoutExpired:
                            finished = False
                        pending += tail.read()
                        lines = pending.split("\n")
                        pending = lines.pop()
                        for line in lines:
                            if line.startswith(("[", "Authenticate ", "Frozen pair scoring:", "Observable pose search:", "Profile output:", "Fresh ")):
                                print(line.rstrip(), flush=True)
                                last_echo = time.perf_counter()
                        if finished:
                            break
                        if time.perf_counter()-last_echo >= 30:
                            print(f"[working +{time.perf_counter()-started:.0f}s] {log.stem}; log: {log}", flush=True)
                            last_echo = time.perf_counter()
        except KeyboardInterrupt:
            # Ctrl+C is normally delivered to both console processes on Windows.
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            record.update(status="interrupted", returncode=130, elapsed_seconds=time.perf_counter()-started,
                          ended_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(log.with_suffix(".attempt.json"), record)
            raise
    record.update(status="complete" if code == 0 else "failed", returncode=code, elapsed_seconds=time.perf_counter()-started,
                  ended_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(log.with_suffix(".attempt.json"), record)
    return record


def commands_for(case, cohort_root, run_root, config, device="cuda", cpu_threads=0):
    stage2 = DESKTOP / "paper-reconstruction-stage2"
    stage3 = DESKTOP / "paper-reconstruction-stage3"
    output = run_root / "cases" / case["case_id"]
    stage1 = cohort_root / case["path"] / "stage1"
    py2, py3 = stage2 / ".venv/Scripts/python.exe", stage3 / ".venv/Scripts/python.exe"
    return [
        ("stage2", [py2, "-B", stage2 / "scripts/run_stage2_inference.py", "--input", stage1, "--output", output / "stage2",
                    "--release", stage2 / "models/release.json", "--cpu-threads", str(cpu_threads), "--resume"]),
        ("stage2_verify", [py2, "-B", stage2 / "scripts/verify_stage2_output.py", "--input", output / "stage2", "--release", stage2 / "models/release.json"]),
        ("stage3", [py3, "-B", stage3 / "scripts/stage3.py", "infer", "--stage1", stage1, "--stage2", output / "stage2",
                    "--output", output / "stage3", "--models", stage3 / "models", "--device", device, "--puzzle-id", case["case_id"]]),
        ("stage4", [sys.executable, "-B", "-m", "paper_reconstruction_stage4", "solve", "--bundle", output / "stage3",
                    "--config", config, "--output", output / "stage4", "--resume"])]


def select_subset(cases, sizes=None, ids=None, limit=None):
    if sizes and set(sizes) - {c["pieces"] for c in cases}:
        raise ValueError("Requested size is not in benchmark")
    if ids and set(ids) - {c["case_id"] for c in cases}:
        raise ValueError("Unknown case IDs")
    selected = [c for c in cases if (not sizes or c["pieces"] in sizes) and (not ids or c["case_id"] in ids)]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No cases selected")
    return selected


def completed_stage2(case_root, input_root):
    """Never rerun a completed Stage 2: it rewrites timestamps in its manifest.

    Its independent verifier runs next. Keep byte identity for Stage 3 receipts.
    """
    path = case_root / "stage2/edge_hypotheses_manifest.json"
    if not path.is_file():
        return None
    manifest = read_json(path)
    if manifest.get("status") != "complete" or manifest.get("truth_free") is not True:
        raise ValueError("Existing Stage 2 ready manifest is not a complete truth-free output")
    if manifest["stage1_manifest_sha256"] != file_hash(input_root / "segmentation_manifest.jsonl"):
        raise ValueError("Existing Stage 2 output belongs to different Stage 1 inputs")
    sha = file_hash(path)
    marker = case_root / "stage2_commit.json"
    if marker.is_file() and read_json(marker)["manifest_sha256"] != sha:
        raise ValueError("Committed Stage 2 manifest changed; choose a new run or restore its original bytes")
    return {"status": "reused_complete", "returncode": 0, "manifest_sha256": sha,
            "note": "Inference skipped to preserve Stage 3 receipt identity; verifier follows"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=WORKSPACE / "benchmarks/development_v2/benchmark.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=WORKSPACE / "configs/hybrid_development.json")
    parser.add_argument("--stage3-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stage2-cpu-threads", type=int, default=0)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args(argv)
    root, cohort = load_cohort(args.benchmark, serving_only=True)
    run_root, config = args.output.resolve(), args.config.resolve()
    if run_root.is_relative_to(root) or root.is_relative_to(run_root):
        raise ValueError("Run output must be separate from immutable benchmark inputs")
    for folder in (DESKTOP / "paper-reconstruction-stage2", DESKTOP / "paper-reconstruction-stage3"):
        if run_root.is_relative_to(folder) or folder.is_relative_to(run_root):
            raise ValueError("Run output must not overlap an upstream producer")
    selected = select_subset(cohort["cases"], args.sizes, args.cases, args.limit)
    plans = {c["case_id"]: [(stage, [str(x) for x in cmd]) for stage, cmd in commands_for(c, root, run_root, config, args.stage3_device, args.stage2_cpu_threads)] for c in selected}
    for commands in plans.values():
        for _, command in commands:
            if not Path(command[0]).is_file():
                raise ValueError(f"Missing runtime: {command[0]}")
    if args.plan_only:
        print(__import__("json").dumps({"cases": len(selected), "pieces": sum(c["pieces"] for c in selected),
              "output": str(run_root), "commands": plans, "diagnosis": "Separate process after prediction phase; report/index.html"}, indent=2))
        return 0
    if run_root.exists() and not (args.resume or args.diagnose_only):
        raise ValueError("Output exists; choose a new run or --resume with unchanged serving code and configuration")
    if args.stage2_cpu_threads < 0:
        raise ValueError("CPU threads must be nonnegative")
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        request_path = run_root / "request.json"
        if args.diagnose_only:
            request = read_json(request_path)
            if request["benchmark_sha256"] != cohort["benchmark_sha256"] or request["case_ids"] != [c["case_id"] for c in selected]:
                raise ValueError("Diagnosis must use the original cohort and case subset")
        else:
            request = {"schema_version": "stage4.benchmark-run.v1", "benchmark_sha256": cohort["benchmark_sha256"],
                "benchmark_manifest": str(root / "benchmark.json"), "case_ids": [c["case_id"] for c in selected],
                "stage4_config_sha256": file_hash(config), "stage3_device": args.stage3_device,
                "stage2_cpu_threads": args.stage2_cpu_threads, "commands": plans,
                "producers": {"stage2": producer_identity(DESKTOP / "paper-reconstruction-stage2"),
                              "stage3": producer_identity(DESKTOP / "paper-reconstruction-stage3"),
                              "stage4": implementation_identity()}}
            request["request_sha256"] = digest(request)
            if request_path.exists() and read_json(request_path)["request_sha256"] != request["request_sha256"]:
                raise ValueError("Serving inputs/code/models/configuration changed. Use a new output run.")
            if not request_path.exists():
                atomic_json(request_path, request)
        state_path = run_root / "progress.json"
        state = read_json(state_path) if state_path.exists() else {"schema_version": "stage4.benchmark-progress.v1", "cases": {}}
        if not args.diagnose_only:
            state["phase"] = "predictions"
            atomic_json(state_path, state)
            for index, case in enumerate(selected, 1):
                cid = case["case_id"]
                status = state["cases"].setdefault(cid, {"attempts": []})
                status["status"] = "running"
                status.pop("failed_stage", None)
                for stage, command in plans[cid]:
                    print(f"[{index}/{len(selected)}] {cid} ({case['pieces']} pieces): {stage}", flush=True)
                    status["active_stage"] = stage
                    atomic_json(state_path, state)
                    try:
                        reuse = completed_stage2(run_root / "cases" / cid, root / case["path"] / "stage1") if stage == "stage2" else None
                        if reuse is not None:
                            attempt = reuse
                        elif stage == "stage4" and (file_hash(config) != request["stage4_config_sha256"] or
                                                   implementation_identity() != request["producers"]["stage4"]):
                            attempt = {"status": "failed", "returncode": 2, "error": "Stage 4 serving identity changed during the run; use a new output directory"}
                        else:
                            attempt = execute(command, run_root / "cases" / cid / "logs" / f"{stage}.log")
                        if stage == "stage2_verify" and attempt["returncode"] == 0:
                            ready = run_root / "cases" / cid / "stage2/edge_hypotheses_manifest.json"
                            atomic_json(ready.parent.parent / "stage2_commit.json", {"manifest_sha256": file_hash(ready)})
                    except (ValueError, OSError, KeyError) as exc:
                        attempt = {"status": "failed", "returncode": 2, "error": str(exc)}
                    status["attempts"].append({"stage": stage, **attempt})
                    if attempt["returncode"]:
                        status.update(status="failed", failed_stage=stage)
                        atomic_json(state_path, state)
                        break
                else:
                    status.update(status="predicted", active_stage=None)
                atomic_json(state_path, state)
        # A phase barrier: no reference file has been read by this controller or its serving children.
        state["phase"] = "diagnostics"
        atomic_json(state_path, state)
        command = [sys.executable, "-B", WORKSPACE / "scripts/diagnose_benchmark.py", "--benchmark", root / "benchmark.json", "--run", run_root]
        attempt = execute(command, run_root / "logs/diagnostics.log")
        state["diagnostics"] = attempt
        state["phase"] = "finished" if attempt["returncode"] == 0 else "diagnostics_failed"
        atomic_json(state_path, state)
        failed = [cid for cid, c in state["cases"].items() if c["status"] != "predicted"]
        if attempt["returncode"]:
            print(f"Diagnostic generation failed or reported errors. Inspect {run_root / 'logs/diagnostics.log'}; reports may be partial.", flush=True)
        print(f"Report: {run_root / 'report/index.html'}; failed serving cases: {len(failed)}", flush=True)
        return 1 if failed or attempt["returncode"] else 0
    except KeyboardInterrupt:
        print("Interrupted. Outputs and caches preserved. Re-run the same command with --resume.", flush=True)
        return 130
    finally:
        lock.unlink(missing_ok=True)
