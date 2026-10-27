"""Fixed eight-piece consumer experiment. Truth is opened only by later diagnosis."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time
from paper_reconstruction_stage4.outputs import implementation_identity, read_result
from .candidate import CASE_ID, fixed_case
from .common import WORKSPACE, atomic_json, digest, file_hash, inventory, read_json, verify_files
from .paths import artifact_paths
from .runner import completed_stage2, execute


def utc():
    return datetime.now(timezone.utc).isoformat()


def make_plan(source, output, profile, archive, config_override=None):
    root, cohort, case = fixed_case(WORKSPACE / "benchmarks/development_v2/benchmark.json")
    source = source.resolve()
    output = output.resolve()
    source_request, source_progress = read_json(source / "request.json"), read_json(source / "progress.json")
    if source_request["case_ids"] != [CASE_ID] or source_request["benchmark_sha256"] != cohort["benchmark_sha256"]:
        raise ValueError("Source must be the fixed eight-piece cohort case")
    if source_progress["phase"] not in {"finished", "diagnostics_failed"}:
        raise ValueError("Source serving must have stopped before copying committed artifacts")
    source_case = source / "cases" / CASE_ID
    paths = artifact_paths(source_case, source_progress["cases"][CASE_ID])
    result = read_result(paths["stage4"])
    evidence_manifest = read_json(paths["stage3"] / "bundle.json")
    if result["input"]["manifest_sha256"] != evidence_manifest["manifest_sha256"]:
        raise ValueError("Source prediction/evidence binding differs")
    if completed_stage2(source_case, root / case["path"] / "stage1") is None:
        raise ValueError("Source Stage 2 publication is incomplete")
    for protected in (source, root, WORKSPACE / "src", WORKSPACE / "controls", WORKSPACE.parent / "paper-reconstruction-stage3", WORKSPACE.parent / "paper-reconstruction-stage2"):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Experiment output overlaps protected inputs/source")
    if profile == "control" and archive:
        raise ValueError("Frozen control does not implement archive expansion")
    config = (config_override or (WORKSPACE / "controls/stage4_alpha_0_1_0_review_baseline/configs/hybrid_development.json" if profile == "control"
              else WORKSPACE / ("configs/structural_archive_development.json" if archive else "configs/structural_development.json"))).resolve()
    case_output = output / "cases" / CASE_ID
    command = [sys.executable, "-B"]
    command += [str(WORKSPACE / "scripts/run_frozen_stage4_control.py")] if profile == "control" else ["-m", "paper_reconstruction_stage4"]
    command += ["solve", "--bundle", str(case_output / "stage3"), "--config", str(output / "stage4_config.json"), "--output", str(case_output / "stage4")]
    value = {"schema_version": "stage4.consumer-experiment.v1", "case_ids": [CASE_ID], "benchmark_sha256": cohort["benchmark_sha256"],
        "benchmark_manifest": str(root / "benchmark.json"), "case": {k:v for k,v in case.items() if k != "files"},
        "source_run": str(source), "source_request_sha256": file_hash(source / "request.json"), "source_run_id": result["run_id"],
        "source_primary_layout_sha256": result["layouts"][0]["layout_sha256"], "input_manifest_sha256": evidence_manifest["manifest_sha256"],
        "source_artifact_paths": {k:str(v) for k,v in paths.items()}, "consumer_profile": profile,
        "stage4_config_source": str(config), "stage4_config_sha256": file_hash(config), "configuration": read_json(config),
        "consumer_implementation": read_json(WORKSPACE / "controls/stage4_alpha_0_1_0_review_baseline/receipt.json")["implementation"] if profile == "control" else implementation_identity(),
        "controller_files": {str(p.relative_to(WORKSPACE)):file_hash(p) for p in [Path(__file__), WORKSPACE / "scripts/run_stage4_experiment.py", WORKSPACE / "scripts/run_frozen_stage4_control.py"]},
        "commands": {"stage4": command}, "stage2_reused": True, "stage3_reused": True, "fresh_full_pipeline_measurement": False,
        "source_profile": str(paths.get("stage3_profile", paths["stage3"].parent))}
    value["request_sha256"] = digest(value)
    return root, case, value


def materialize(request, output):
    started = time.perf_counter()
    case_root = output / "cases" / CASE_ID
    receipts = {}
    for stage in ("stage2", "stage3"):
        source = Path(request["source_artifact_paths"][stage])
        expected = inventory(source)
        shutil.copytree(source, case_root / stage)
        verify_files(case_root / stage, expected)
        verify_files(source, expected)
        receipts[stage] = {"source": str(source), "copied_bytes": sum(r["bytes"] for r in expected), "files": expected, "status": "reused_byte_preserving_copy"}
    ranking = Path(request["source_artifact_paths"]["stage3_work"]) / "rankings.json"
    if ranking.is_file():
        target = case_root / "stage3.work/rankings.json"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ranking, target)
        assert file_hash(target) == file_hash(ranking)
    receipts["elapsed_seconds"] = time.perf_counter()-started
    atomic_json(output / "reused_input_receipts.json", receipts)
    return receipts


def summarize(output):
    """Called only after the independent diagnostic process has stopped."""
    request, progress = read_json(output / "request.json"), read_json(output / "progress.json")
    source = Path(request["source_run"])
    def metrics(run):
        path = run / "report" / CASE_ID / "diagnosis.json"
        if not path.is_file():
            return {"status": "unavailable", "reason": "No completed diagnosis"}
        data = read_json(path)
        s4 = data.get("stages", {}).get("stage4", {})
        return {"status": s4.get("status"), "quality": s4.get("assembly_quality"), "evaluation": s4.get("evaluation"),
                "termination": s4.get("termination"), "search": s4.get("search"), "resources": s4.get("resources"),
                "refinement": s4.get("refinement"), "input_load": s4.get("input_load"), "errors": data.get("errors")}
    value = {"schema_version": "stage4.consumer-comparison.v1", "source_control": metrics(source), "experiment": metrics(output),
             "request_sha256": request["request_sha256"], "profile": request["consumer_profile"], "configuration": request["configuration"],
             "execution": progress, "source_run": str(source), "experiment_run": str(output),
             "stage2_and_stage3_reused": True, "fresh_end_to_end_wall_seconds": None,
             "timing_reason": "Only the consumer ran; copy/verification and post-prediction diagnostics are separate costs.",
             "primary_layout_only": True, "manual_visual_review": None,
             "visual_review_reason": "Inspect the sealed primary preview; connectivity and a higher score are not accuracy claims."}
    atomic_json(output / "comparison.json", value)
    lines = ["# Stage 4 consumer experiment", "", f"Profile: `{request['consumer_profile']}`.",
        "Stage 2 and Stage 3 were reused byte-for-byte. This is not a fresh full-pipeline timing.", "",
        f"- Source: `{source}`.", f"- Primary preview: `{output / 'cases' / CASE_ID / 'stage4/preview.png'}`.",
        f"- Pipeline diagnosis: `{output / 'report/index.html'}`.", f"- Comparison and source measurements: `{output / 'comparison.json'}`.", "",
        "Reference files were opened only after the consumer stopped, in a separate process. Only the committed primary layout was evaluated. Visual review is required.", ""]
    (output / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=WORKSPACE / "outputs/candidate8_exact_v1")
    parser.add_argument("--spec", type=Path, help="Explicit external-bundle experiment specification")
    parser.add_argument("--variant", choices=("baseline", "search", "refine", "active", "combined", "robust"), default="robust")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=("structural", "control"), default="structural")
    parser.add_argument("--archive-expansion", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args(argv)
    if args.spec:
        if args.archive_expansion or args.profile != "structural":
            parser.error("External specs use --variant; archive/control flags belong to the pipeline-source route")
        from .joint_experiment import run
        return run(args)
    if args.diagnose_only and args.output is None:
        parser.error("--diagnose-only requires the existing --output directory")
    suffix = "control" if args.profile == "control" else "archive" if args.archive_expansion else "working"
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or WORKSPACE / ("outputs/stage4_structural8_"+suffix+"_"+label)).resolve()
    started, stamp = time.perf_counter(), utc()
    root, case, request = make_plan(args.source_run, output, args.profile, args.archive_expansion, args.config)
    if args.plan_only:
        print(__import__("json").dumps({"request": request, "output": str(output), "inference_performed": False, "output_created": False}, indent=2))
        return 0
    if output.exists() and not args.diagnose_only:
        raise ValueError("Experiment output exists; choose a fresh --output. Predictions are immutable.")
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        if args.diagnose_only:
            request = read_json(output / "request.json")
            state = read_json(output / "progress.json")
            if request["schema_version"] != "stage4.consumer-experiment.v1" or request["case_ids"] != [CASE_ID]:
                raise ValueError("Not this fixed consumer experiment")
        else:
            atomic_json(output / "request.json", request)
            (output / "stage4_config.json").write_bytes(Path(request["stage4_config_source"]).read_bytes())
            status = {"status": "running", "active_stage": "stage4", "attempts": []}
            state = {"schema_version": "stage4.benchmark-progress.v1", "phase": "predictions", "cases": {CASE_ID:status},
                     "started_utc": stamp, "stage2_reused": True, "stage3_reused": True}
            atomic_json(output / "progress.json", state)
            try:
                state["materialization"] = materialize(request, output)
                if file_hash(output / "stage4_config.json") != request["stage4_config_sha256"]:
                    raise ValueError("Experiment configuration bytes changed")
                attempt = execute(request["commands"]["stage4"], output / "cases" / CASE_ID / "logs/stage4.log", echo_progress=True)
                status["attempts"].append({"stage": "stage4", **attempt})
                if attempt["returncode"] == 0:
                    result = read_result(output / "cases" / CASE_ID / "stage4")
                    if result["input"]["manifest_sha256"] != request["input_manifest_sha256"]:
                        raise ValueError("Consumer evidence binding differs")
                    if result["implementation"] != request["consumer_implementation"]:
                        raise ValueError("Consumer implementation changed during experiment preparation/execution")
                    status.update(status="predicted", active_stage=None)
                else:
                    status.update(status="failed", failed_stage="stage4", error="Consumer child failed; inspect its log")
            except KeyboardInterrupt:
                status.update(status="interrupted", failed_stage="stage4")
                state.update(phase="interrupted", ended_utc=utc())
                atomic_json(output / "progress.json", state)
                return 130
            except (ValueError, OSError, KeyError) as exc:
                status.update(status="failed", failed_stage="stage4", error=str(exc))
            state.update(consumer_experiment_wall_seconds=time.perf_counter()-started, ended_utc=utc())
        state["phase"] = "diagnostics"
        atomic_json(output / "progress.json", state)
        diagnostic = execute([sys.executable, "-B", WORKSPACE / "scripts/diagnose_benchmark.py", "--benchmark", root / "benchmark.json", "--run", output], output / "logs/diagnostics.log", echo_progress=True)
        state.update(phase="finished" if diagnostic["returncode"] == 0 else "diagnostics_failed", diagnostics=diagnostic)
        atomic_json(output / "progress.json", state)
        summarize(output)
        print("Experiment report: " + str(output / "report/index.html"), flush=True)
        return 0 if state["cases"][CASE_ID]["status"] == "predicted" and diagnostic["returncode"] == 0 else 1
    finally:
        (output / "run.lock").unlink(missing_ok=True)
