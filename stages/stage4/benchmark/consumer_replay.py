"""Replay a frozen consumer against the same ten bundles, without upstream inference."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.outputs import read_result
from .candidate import THREADS
from .common import WORKSPACE, atomic_json, digest, file_hash, load_cohort, read_json
from .runner import execute
from .validation import safe_output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--config", type=Path, help="Explicit ablation configuration; source evidence stays fixed")
    p.add_argument("--comparison-run", type=Path, help="Committed consumer run to compare; defaults to source run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--follow", action="store_true", help="Wait for each committed source prediction; no reference access")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    source, snapshot, output = args.source_run.resolve(), args.snapshot.resolve(), args.output.resolve()
    comparison = args.comparison_run.resolve() if args.comparison_run else source
    configuration = args.config.resolve() if args.config else source / "stage4_config.json"
    expected_config = SolverConfig.from_dict(read_json(configuration)).to_dict()
    original = read_json(source / "request.json")
    if original["request_sha256"] != digest({k: v for k, v in original.items() if k != "request_sha256"}):
        raise ValueError("Source request checksum mismatch")
    cohort_root, cohort = load_cohort(original["benchmark_manifest"], serving_only=True)
    if cohort["benchmark_sha256"] != original["benchmark_sha256"]:
        raise ValueError("Source cohort changed")
    compare_request = read_json(comparison / "request.json")
    if (compare_request["request_sha256"] != digest({k:v for k,v in compare_request.items() if k != "request_sha256"}) or
            compare_request["benchmark_sha256"] != original["benchmark_sha256"] or compare_request["case_ids"] != original["case_ids"]):
        raise ValueError("Comparison request/cohort differs")
    if comparison != source and read_json(comparison / "progress.json")["phase"] != "finished":
        raise ValueError("Comparison run must be finished")
    safe_output(output, [source, comparison, snapshot, cohort_root, WORKSPACE / "src", WORKSPACE / "scripts", WORKSPACE / "benchmark"])
    receipt = read_json(snapshot / "receipt.json")
    for name, sha in receipt["files"].items():
        if file_hash(snapshot / name) != sha:
            raise ValueError("Snapshot changed")
    request = {"schema_version": "stage4.consumer-replay.v1", "source_run": str(source),
               "source_request_sha256": original["request_sha256"], "benchmark_sha256": cohort["benchmark_sha256"],
               "case_ids": original["case_ids"], "consumer_snapshot": str(snapshot), "consumer_implementation": receipt["implementation"],
               "configuration_sha256": file_hash(configuration), "configuration_source": str(configuration),
               "configuration_changed_from_source": expected_config != SolverConfig.from_dict(read_json(source / "stage4_config.json")).to_dict(),
               "comparison_run": str(comparison), "comparison_request_sha256": compare_request["request_sha256"],
               "controller_sha256": file_hash(__file__), "upstream_inference_repeated": False,
               "timing_scope": "Consumer-only replay; upstream evidence reused. Concurrent serial upstream development inference may use another CPU core. Not fresh pipeline timing."}
    request["request_sha256"] = digest(request)
    if output.exists() and not args.resume:
        raise ValueError("Existing replay; choose new output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    state_path = output / "progress.json"
    state = read_json(state_path) if state_path.exists() else {"phase": "predictions", "cases": {}}
    try:
        if (output / "request.json").exists():
            if read_json(output / "request.json")["request_sha256"] != request["request_sha256"]:
                raise ValueError("Replay identity changed")
        else:
            atomic_json(output / "request.json", request)
            shutil.copyfile(configuration, output / "stage4_config.json")
        if file_hash(output / "stage4_config.json") != request["configuration_sha256"]:
            raise ValueError("Replay configuration changed")
        for index, cid in enumerate(request["case_ids"], 1):
            while True:
                progress = read_json(source / "progress.json")
                src = progress["cases"].get(cid, {})
                if src.get("status") in {"predicted", "failed"}:
                    break
                if not args.follow or progress["phase"] in {"finished", "diagnostics_failed", "interrupted"}:
                    raise ValueError("Source prediction unavailable: " + cid)
                state.update(waiting_for_source_case=cid)
                atomic_json(state_path, state)
                time.sleep(10)
            if src["status"] == "failed":
                state["cases"][cid] = {"status": "source_failed", "failed_stage": src.get("failed_stage")}
                atomic_json(state_path, state)
                continue
            target = output / "cases" / cid / "stage4"
            baseline = read_result(source / "cases" / cid / "stage4")
            bundle = load_bundle(source / "cases" / cid / "stage3", full=True)
            if baseline["input"]["manifest_sha256"] != bundle.manifest["manifest_sha256"]:
                raise ValueError("Source bundle/primary mismatch")
            if not (target / "manifest.json").exists():
                print(f"[{index}/{len(request['case_ids'])}] cached consumer: {cid}", flush=True)
                command = [sys.executable, "-B", WORKSPACE / "scripts/run_source_snapshot.py", snapshot, "solve",
                           "--bundle", bundle.root, "--config", output / "stage4_config.json", "--output", target]
                attempt = execute(command, target.parent / "consumer.log", env_overrides=THREADS, echo_progress=True)
                state["cases"][cid] = attempt
                atomic_json(state_path, state)
                if attempt["returncode"]:
                    continue
            result = read_result(target)
            if (result["input"]["manifest_sha256"] != baseline["input"]["manifest_sha256"] or
                    result["configuration"] != expected_config or result["implementation"] != receipt["implementation"]):
                raise ValueError("Consumer replay context differs")
            state["cases"].setdefault(cid, {}).update(status="predicted", input_manifest_sha256=bundle.manifest["manifest_sha256"])
            atomic_json(state_path, state)
        state.update(phase="diagnostics", waiting_for_source_case=None)
        atomic_json(state_path, state)
        command = [sys.executable, "-B", WORKSPACE / "scripts/diagnose_consumer_replay.py", "--run", output]
        state["diagnostics"] = execute(command, output / "diagnostics.log", env_overrides=THREADS, echo_progress=True)
        state["phase"] = "finished" if state["diagnostics"]["returncode"] == 0 else "diagnostics_failed"
        atomic_json(state_path, state)
        return int(state["diagnostics"]["returncode"] != 0 or any(c["status"] != "predicted" for c in state["cases"].values()))
    except KeyboardInterrupt:
        state["phase"] = "interrupted"
        atomic_json(state_path, state)
        return 130
    finally:
        lock.unlink(missing_ok=True)


def diagnose(argv=None):
    p = argparse.ArgumentParser(description="Post-prediction consumer speed/quality comparison")
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args(argv)
    output = args.run.resolve()
    request, progress = read_json(output / "request.json"), read_json(output / "progress.json")
    if progress["phase"] not in {"diagnostics", "finished", "diagnostics_failed"}:
        raise ValueError("Prediction barrier not reached")
    if request["request_sha256"] != digest({k: v for k, v in request.items() if k != "request_sha256"}):
        raise ValueError("Replay request checksum mismatch")
    source = Path(request["source_run"])
    comparison = Path(request.get("comparison_run", source))
    compare_request = read_json(comparison / "request.json")
    if compare_request["request_sha256"] != request.get("comparison_request_sha256", request["source_request_sha256"]):
        raise ValueError("Comparison request changed")
    original = read_json(source / "request.json")
    if original["request_sha256"] != request["source_request_sha256"]:
        raise ValueError("Source request changed")
    root, cohort = load_cohort(original["benchmark_manifest"])
    if cohort["benchmark_sha256"] != request["benchmark_sha256"]:
        raise ValueError("Comparison cohort changed")
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from .report import document, esc, image, table, number, link
    rows, cells, visuals = [], [], []
    for cid in request["case_ids"]:
        if progress["cases"].get(cid, {}).get("status") != "predicted":
            rows.append({"case_id": cid, "status": progress["cases"].get(cid)})
            continue
        case = next(c for c in cohort["cases"] if c["case_id"] == cid)
        reference = read_json(root / case["path"] / "evaluation_only/reference_geometry.json")
        results = [read_result(run / "cases" / cid / "stage4") for run in (comparison, output)]
        if results[0]["input"]["manifest_sha256"] != results[1]["input"]["manifest_sha256"]:
            raise ValueError("Comparison bundle differs")
        reference["input_manifest_sha256"] = results[0]["input"]["manifest_sha256"]
        qualities = [assembly_quality(r, reference, evaluate(r, reference, 4.)) for r in results]
        row = {"case_id": cid, "status": "compared", "primary_geometry_identical": all(results[0]["layouts"][0][k] == results[1]["layouts"][0][k] for k in ("pieces", "selected_joins", "components", "objective_score")),
               "quality_baseline": qualities[0], "quality_cached": qualities[1],
               "elapsed_before_publication_seconds": [r["resources"]["elapsed_before_publication_seconds"] for r in results],
               "proposals": [r["search"]["proposals"] for r in results],
               "visual_review": None, "source_diagnosis": str(source / "report" / cid / "index.html")}
        rows.append(row)
        cells.append((esc(cid), esc(row["primary_geometry_identical"]), *[esc(q["component_sizes"]) for q in qualities],
                      *[number(q["selected_neighbor_identity_precision"]) for q in qualities], *[number(t) for t in row["elapsed_before_publication_seconds"]]))
        visuals.extend(image(run / "cases" / cid / "stage4/preview.png", output, cid + " — " + label) for run, label in ((comparison, "comparison primary"), (output, "replay primary")))
    atomic_json(output / "comparison.json", {"request": request, "cases": rows, "precision_is_not_structural_success": True})
    content = '<h1>Identical-evidence consumer comparison</h1><p>Primary constructions, no reference-based portfolio selection. Consumer-only instrumented development timings. Four-pixel accuracy is separate from visual construction quality.</p>'
    content += table(["Case", "Same primary geometry", "Baseline components", "Cached components", "Baseline true-neighbor fraction", "Cached true-neighbor fraction", "Baseline seconds", "Cached seconds"], cells)
    content += '<div class="visuals">'+''.join(visuals)+'</div>'
    (output / "index.html").write_text(document("Consumer comparison", content), encoding="utf-8")
    print("Comparison:", output / "index.html", flush=True)
    return 0
