"""Verified consumer-only experiments over external v2 evidence bundles.

This controller does not import evaluation or read reference geometry. A
separate child receives reference access only after serving exits successfully.
"""
import json
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.outputs import implementation_identity, read_result
from .common import WORKSPACE, atomic_json, digest, file_hash, inventory, load_cohort, read_json, verify_files
from .runner import execute


def resolve(spec_path, value):
    path = Path(value)
    return (path if path.is_absolute() else spec_path.parent/path).resolve()


def make_plan(spec_path, output, variant, override=None):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec = read_json(spec_path)
    if spec.get("schema_version") != "stage4.external-experiment.v1":
        raise ValueError("Unknown external experiment specification")
    cohort_path = resolve(spec_path, spec["benchmark"])
    root, cohort = load_cohort(cohort_path, verify=False)
    if cohort["benchmark_sha256"] != spec["benchmark_sha256"]:
        raise ValueError("External experiment cohort binding differs")
    case = next((c for c in cohort["cases"] if c["case_id"] == spec["case_id"]), None)
    if case is None:
        raise ValueError("Case not present in the declared cohort")
    case_root = (root/case["path"]).resolve()
    verify_files(case_root, [r for r in case["files"] if r["path"].startswith("stage1/")])
    bundle_path = resolve(spec_path, spec["bundle"])
    data = load_bundle(bundle_path, full=True)
    if data.manifest["manifest_sha256"] != spec["input_manifest_sha256"] or data.manifest["puzzle_id"] != case["case_id"]:
        raise ValueError("External evidence identity/case differs")
    rows = [json.loads(line) for line in (case_root/"stage1/segmentation_manifest.jsonl").read_text().splitlines() if line.strip()]
    if {r["capture_id"] for r in rows} != set(data.entries):
        raise ValueError("External bundle physical inventory differs from case")
    for row in rows:
        piece = data.manifest["pieces"][row["capture_id"]]
        for field, source in (("image", "piece_path"), ("mask", "piece_mask_path")):
            if file_hash(case_root/"stage1"/row[source]) != data.manifest["assets"][piece[field]]["sha256"]:
                raise ValueError("External evidence native asset differs from case")
    control_path = resolve(spec_path, spec["control_result"])
    control = read_result(control_path)
    if control["input"]["manifest_sha256"] != spec["input_manifest_sha256"] or control["run_id"] != spec["control_run_id"]:
        raise ValueError("External control/input binding differs")
    frozen = read_json(WORKSPACE/"controls/stage4_structural_v2_joint_baseline/receipt.json")
    if control["implementation"] != frozen["implementation"]:
        raise ValueError("External control does not match preserved structural-v2 implementation")
    for protected in (bundle_path, control_path, root, spec_path, WORKSPACE/"src", WORKSPACE/"controls",
                      WORKSPACE.parent/"paper-reconstruction-stage3", WORKSPACE.parent/"paper-reconstruction-stage2"):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("External experiment output overlaps protected inputs/source")
    config_path = resolve(spec_path, spec["control_config"]) if variant == "baseline" else (override or WORKSPACE/f"configs/joint_{variant}_development.json").resolve()
    config_raw = read_json(config_path)
    if variant == "baseline":
        if override:
            raise ValueError("Baseline configuration cannot be overridden")
        frozen_config_path = WORKSPACE/"controls/stage4_structural_v2_joint_baseline/src/paper_reconstruction_stage4/config.py"
        if file_hash(frozen_config_path) != frozen["files"]["src/paper_reconstruction_stage4/config.py"]:
            raise ValueError("Frozen baseline configuration source changed")
        module_spec = importlib.util.spec_from_file_location("_stage4_structural_control_config", frozen_config_path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        normalized = module.SolverConfig.from_dict(config_raw).to_dict()
        if normalized != control["configuration"]:
            raise ValueError("Baseline config differs from saved control")
        impl = frozen["implementation"]
        command = [sys.executable, "-B", str(WORKSPACE/"scripts/run_frozen_structural_control.py")]
        configuration = normalized
    else:
        config = SolverConfig.from_dict(config_raw)
        if config.solver != "joint_beam_v3" or config.evidence_policy != "working_pool":
            raise ValueError("Joint experiment requires the joint solver and frozen official working pool")
        impl, configuration = implementation_identity(), config.to_dict()
        command = [sys.executable, "-B", "-m", "paper_reconstruction_stage4"]
    command += ["solve", "--bundle", str(output/"stage3"), "--config", str(output/"stage4_config.json"), "--output", str(output/"stage4")]
    if variant != "baseline":
        command += ["--initial-result", str(output/"control")]
    request = {"schema_version": "stage4.external-consumer-request.v1", "case_id": case["case_id"],
        "variant": variant, "spec_path": str(spec_path), "spec_sha256": file_hash(spec_path),
        "benchmark": str(cohort_path), "benchmark_sha256": cohort["benchmark_sha256"], "case_input": str(case_root),
        "source_bundle": str(bundle_path), "input_manifest_sha256": spec["input_manifest_sha256"],
        "source_bundle_files": inventory(bundle_path), "control_result": str(control_path), "control_run_id": control["run_id"],
        "control_files": inventory(control_path), "control_layout_sha256": control["layouts"][0]["layout_sha256"],
        "config_source": str(config_path), "config_sha256": file_hash(config_path), "configuration": configuration,
        "consumer_implementation": impl, "commands": {"stage4": command},
        "controller_files": {str(p.relative_to(WORKSPACE)): file_hash(p) for p in [Path(__file__), WORKSPACE/"benchmark/experiment.py", WORKSPACE/"scripts/run_frozen_structural_control.py",
            WORKSPACE/"benchmark/joint_diagnostics.py", WORKSPACE/"scripts/diagnose_stage4_experiment.py", WORKSPACE/"benchmark/quality.py", WORKSPACE/"evaluation/evaluate.py"]},
        "stage3_reused": True, "stage2_inference_or_publication_verified": False, "fresh_full_pipeline_measurement": False}
    request["request_sha256"] = digest(request)
    return request


def materialize(request, output):
    started = time.perf_counter()
    for source_key, files_key, name in (("source_bundle", "source_bundle_files", "stage3"), ("control_result", "control_files", "control")):
        verify_files(Path(request[source_key]), request[files_key])
        shutil.copytree(request[source_key], output/name)
        verify_files(output/name, request[files_key])
    shutil.copyfile(request["config_source"], output/"stage4_config.json")
    if file_hash(output/"stage4_config.json") != request["config_sha256"]:
        raise ValueError("Frozen external experiment config changed")
    return {"elapsed_seconds": time.perf_counter()-started, "byte_preserving_copy": True,
            "stage2_status": "not invoked; no new Stage 2 qualification claimed", "stage3_status": "verified reused bundle"}


def verify_after_child(request, output, verify_current=True):
    result = read_result(output/"stage4")
    if result["input"]["manifest_sha256"] != request["input_manifest_sha256"] or result["implementation"] != request["consumer_implementation"]:
        raise ValueError("Published consumer implementation/input differs from request")
    if result["configuration"] != request["configuration"]:
        raise ValueError("Published consumer config differs from request")
    if file_hash(output/"stage4_config.json") != request["config_sha256"]:
        raise ValueError("Copied consumer config bytes changed")
    verify_files(Path(request["source_bundle"]), request["source_bundle_files"])
    verify_files(Path(request["control_result"]), request["control_files"])
    for name, expected in request["controller_files"].items():
        if verify_current and file_hash(WORKSPACE/name) != expected:
            raise ValueError("Experiment controller changed during serving")
    if verify_current and request["variant"] != "baseline" and implementation_identity() != request["consumer_implementation"]:
        raise ValueError("Consumer source/dependencies changed during serving")
    return result


def run(args):
    if args.diagnose_only and args.output is None:
        raise ValueError("--diagnose-only requires --output")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or WORKSPACE/f"outputs/joint_{args.variant}_{stamp}").resolve()
    if args.diagnose_only:
        request = read_json(output/"request.json")
        progress = read_json(output/"progress.json")
        if progress.get("phase") not in {"predicted", "finished", "diagnostics_failed"}:
            raise ValueError("Only a stopped, verified prediction may be diagnosed")
        verify_after_child(request, output, verify_current=False)
    else:
        request = make_plan(args.spec, output, args.variant, args.config)
        if args.plan_only:
            print(json.dumps({"request": request, "output": str(output), "inference_performed": False}, indent=2))
            return 0
        if output.exists():
            raise ValueError("External experiment output exists; choose a fresh directory")
        output.mkdir(parents=True)
        progress = {"phase": "prepared", "variant": args.variant}
        atomic_json(output/"request.json", request)
    descriptor = os.open(output/"run.lock", os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    try:
        if not args.diagnose_only:
            progress["materialization"] = materialize(request, output)
            progress["phase"] = "serving"; atomic_json(output/"progress.json", progress)
            print(f"{request['case_id']}: Stage 4 {args.variant}; upstream evidence reused", flush=True)
            attempt = execute(request["commands"]["stage4"], output/"logs/stage4.log", echo_progress=True)
            progress["serving"] = attempt
            if attempt["returncode"] != 0:
                progress["phase"] = "failed"; atomic_json(output/"progress.json", progress)
                return 1
            verify_after_child(request, output)
            progress["phase"] = "predicted"
            progress["through_prediction_seconds"] = time.perf_counter()-started
            atomic_json(output/"progress.json", progress)
        # This subprocess is the first component allowed to open reference maps.
        diagnostic = execute([sys.executable, "-B", WORKSPACE/"scripts/diagnose_stage4_experiment.py", "--run", output],
                             output/"logs/diagnostics.log", echo_progress=True)
        progress.update(phase="finished" if diagnostic["returncode"] == 0 else "diagnostics_failed", diagnostics=diagnostic)
        atomic_json(output/"progress.json", progress)
        print("Comparison: "+str(output/"comparison.json"), flush=True)
        return 0 if diagnostic["returncode"] == 0 else 1
    except KeyboardInterrupt:
        progress["phase"] = "interrupted"; atomic_json(output/"progress.json", progress)
        return 130
    except (ValueError, OSError, KeyError) as exc:
        progress.update(phase="failed", error=str(exc)); atomic_json(output/"progress.json", progress)
        raise
    finally:
        (output/"run.lock").unlink(missing_ok=True)
