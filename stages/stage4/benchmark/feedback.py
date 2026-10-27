"""Collect a portable Stage 3 return package after the separate diagnostic phase."""
from pathlib import Path
import json
from .common import atomic_json, read_json, file_hash
from .paths import artifact_paths
from .quality import quantiles


def missing(reason):
    return {"value": None, "reason": reason}


def read_if_present(path):
    return read_json(path) if Path(path).is_file() else None


def write_feedback(run_root):
    run_root = Path(run_root).resolve()
    request, progress = read_json(run_root / "request.json"), read_json(run_root / "progress.json")
    if progress.get("phase") not in {"finished", "diagnostics_failed"}:
        raise ValueError("Feedback requires a stopped serving attempt and a completed diagnostic attempt")
    cid = request["case_ids"][0]
    case_root = run_root / "cases" / cid
    case_progress = progress["cases"][cid]
    paths = artifact_paths(case_root, case_progress)
    diagnostic = read_if_present(run_root / "report" / cid / "diagnosis.json") or {}
    prof = read_if_present(paths.get("stage3_profile", case_root / "stage3_profile") / "profile.json") or {}
    s2_manifest = read_if_present(paths["stage2"] / "edge_hypotheses_manifest.json") or {}
    s2, s3, s4 = (diagnostic.get("stages", {}).get(s, {}) for s in ("stage2", "stage3", "stage4"))
    session = progress.get("sessions", [{}])[-1]
    stages = session.get("stages", {})
    outer = {k: {f: v[f] for f in ("started_utc", "ended_utc", "elapsed_seconds", "status", "returncode", "error") if f in v} for k, v in stages.items()}
    outer_sum = sum(v.get("elapsed_seconds", 0.) for v in stages.values()) + session.get("candidate_preflight", {}).get("elapsed_seconds", 0.)
    serving_wall = session.get("serving_wall_seconds")
    pairs = prof.get("pairs", [])
    counters = prof.get("counters", {})
    pair_times = [p["elapsed_seconds"] for p in pairs if p.get("elapsed_seconds") is not None]
    attempted_pairs = len(pairs) if "pairs" in prof else None
    actual_inference = prof.get("inference_result", {})
    artifacts = {name: str(path) for name, path in paths.items()}
    artifacts.update({"run_root": str(run_root), "request": str(run_root / "request.json"), "progress": str(run_root / "progress.json"),
                      "stage4_configuration": str(run_root / "stage4_config.json"), "report": str(run_root / "report/index.html"),
                      "diagnosis": str(run_root / "report" / cid / "diagnosis.json"), "primary_preview": str(paths["stage4"] / "preview.png"),
                      "stage4_result": str(paths["stage4"] / "result.json"), "stage4_manifest": str(paths["stage4"] / "manifest.json"),
                      "input_contact_sheet": str(Path(request["stage1_input"]) / "input_pieces.jpg"),
                      "reference_overlay": str(run_root / "report" / cid / "placement_overlay.png")})
    profile_root = paths.get("stage3_profile", case_root / "stage3_profile")
    artifacts.update({name: str(profile_root / name) for name in ("profile.json", "report.md", "functions.json", "calls.pstats")})
    stage4_load = s4.get("input_load", {})
    functions = prof.get("functions", [])
    value = {"schema_version": "stage4.to-stage3-candidate-feedback.v1", "review_status": "automated_measurements_visual_review_pending",
        "case": request["case"], "benchmark_sha256": request["benchmark_sha256"],
        "adapter": "observed-crop-with-4px-contour-margin.v2", "request_sha256": request["request_sha256"],
        "eight_piece_control_equivalence": request.get("eight_piece_control_equivalence", "not_requested"),
        "profile_comparison": prof.get("comparison", missing("Profiler did not publish a comparison record")),
        "equivalence_scope": "No matching frozen eight-piece control was requested. Four-piece qualification is separate evidence.",
        "execution": {"serving_status": case_progress["status"], "failed_stage": case_progress.get("failed_stage"),
                      "error": case_progress.get("error"), "diagnostic_status": progress["phase"],
                      "fresh_reused_flags": {k: session.get(k) for k in ("fresh_stage2", "fresh_stage3", "fresh_stage4", "fresh_end_to_end_measurement")},
                      "fresh_flag_semantics": "True: fresh stage output/work directory; false: prior artifacts reused/replayed; null: stage was not invoked in this session.",
                      "stage_processes_and_checks": outer, "all_sessions": progress.get("sessions", []),
                      "solver_status": s4.get("solver_status"), "solver_termination": s4.get("termination"),
                      "complete_execution_is_accuracy_claim": False},
        "identities": {"candidate": request["candidate"], "stage3_model_policy_sha256": request["stage3_model_policy_sha256"],
                       "stage3_device": request["stage3_device"], "stage3_thread_overrides": request["stage3_environment_overrides"],
                       "inherited_thread_environment": request["inherited_thread_environment"],
                       "producers": request["producers"], "stage4_configuration": request["stage4_configuration"],
                       "stage4_config_sha256": request["stage4_config_sha256"],
                       "stage1_manifest_sha256": next((f["sha256"] for f in request["stage1_input_files"] if f["path"] == "stage1/segmentation_manifest.jsonl"), None),
                       "stage2_manifest_file_sha256": file_hash(paths["stage2"] / "edge_hypotheses_manifest.json") if s2_manifest else None,
                       "stage2_release": {k: s2_manifest.get(k) for k in ("software_version", "release_version", "runtime", "model_sha256", "release_manifest_sha256", "source_checkpoint_sha256", "stage1_input_fingerprint")}},
        "timing": {"session_started_utc": session.get("started_utc"), "session_ended_utc": session.get("ended_utc"),
                   "serving_session_wall_seconds": serving_wall, "fresh_end_to_end_wall_seconds": serving_wall if session.get("fresh_end_to_end_measurement") else None,
                   "nonfresh_reason": None if session.get("fresh_end_to_end_measurement") else "At least one stage was reused, interrupted or failed; this is not a fresh complete serving measurement.",
                   "candidate_snapshot_preflight": session.get("candidate_preflight"), "outer_stage_times": outer,
                   "orchestration_and_other_measured_gap_seconds": max(0., serving_wall-outer_sum) if serving_wall is not None else None,
                   "byte_copy_seconds": 0., "copy_reason": "Evidence/work are mapped in place; no copy or reserialization was performed.",
                   "stage3_profiler": prof.get("timing"), "stage3_function_timings": functions,
                   "inclusive_function_times_overlap": True, "cprofile_enabled": request["cprofile_enabled"],
                   "stage4_input_load": stage4_load, "stage4_resources": s4.get("resources"),
                   "stage4_search_loop_repair_support_timings": missing("Not separately timed by the unchanged Stage 4 solver; counts/rejections are retained below."),
                   "stage4_render_seal_seconds": missing("Outer Stage 4 wall time covers publication; rendering/sealing do not have separate timers."),
                   "offline_diagnostic_process": progress.get("diagnostics")},
        "stage3": {"profile_status": prof.get("status"), "sources_verified_unchanged": prof.get("sources_verified_unchanged"),
                   "counts": s3.get("counts", actual_inference), "possible_unordered_physical_pairs": 28,
                   "completed_expensive_physical_pairs": attempted_pairs, "per_pair_seconds": quantiles(pair_times), "pairs": pairs,
                   "counters": counters if counters else missing("No completed profiler counters are available"),
                   "peak_rss": prof.get("memory_at_report", missing("Profiler memory report unavailable")),
                   "cuda_peak_allocated_bytes": prof.get("cuda_peak_allocated_bytes"), "cuda_peak_reserved_bytes": prof.get("cuda_peak_reserved_bytes"),
                   "ranking_reused": actual_inference.get("rankings_reused"), "physical_pair_caches_reused": actual_inference.get("physical_pair_caches_reused"),
                   "candidate_graph_components": s3.get("candidate_graph_components"), "candidate_graph_isolated_piece_ids": s3.get("candidate_graph_isolated_piece_ids"),
                   "candidate_counts_by_piece": s3.get("candidate_counts_by_piece"), "working_pool_pairs": s3.get("working_pool_pairs")},
        "stage4": {"assembly_quality": s4.get("assembly_quality"), "strict_precision_evaluation": s4.get("evaluation"),
                   "components": s4.get("components"), "selected_joins": s4.get("selected_joins"), "feasibility": s4.get("feasibility"),
                   "solver_singleton_pieces_with_working_neighbors": [c[0] for c in s4["components"] if len(c) == 1 and c[0] not in s3.get("candidate_graph_isolated_piece_ids", [])] if s4.get("components") and "candidate_graph_isolated_piece_ids" in s3 else None,
                   "search": s4.get("search"), "refinement": s4.get("refinement"), "rejections": s4.get("rejections")},
        "per_contact_attribution": diagnostic.get("contact_diagnoses"), "stage2_diagnostics": s2,
        "diagnostic_errors": diagnostic.get("errors", missing("No completed diagnostic JSON")),
        "largest_inclusive_costs_do_not_sum": sorted(functions, key=lambda f: -f["inclusive_seconds"])[:6],
        "manual_review": {"correct_relative_placement_and_orientation": None, "swapped_or_grossly_misplaced_piece_ids": None,
                          "gap_overlap_and_drift_assessment": None, "most_consequential_wrong_or_missing_constraints": None,
                          "reason": "Primary image and diagnostic overlays require review after the user completes this run; no automatic structural success threshold is being invented."},
        "artifacts": {k: {"path": v, "exists": Path(v).exists()} for k, v in artifacts.items()},
        "commands": request["commands"], "training_performed": False, "reliable_piece_capacity": None,
        "reliable_piece_capacity_reason": "One fixed development probe cannot establish repeatable reconstruction capacity.",
        "unavailable_measurement_reasons": {
            "stage3_null_fields": "Unavailable when the profiler did not reach that measurement; inspect profile_status, execution errors and raw profile artifacts.",
            "stage4_null_fields": "Unavailable when no sealed result could be verified and diagnosed; inspect serving_status and diagnostic_errors.",
            "null_identity_fields": "No corresponding producer manifest field or completed publication is present.",
            "per_pair_null_quantiles": "No completed physical-pair timings were reported."}}
    if attempted_pairs and prof.get("timing", {}).get("inference_wall_seconds") is not None:
        value["stage3"]["inference_seconds_per_completed_pair_including_shared_costs"] = prof["timing"]["inference_wall_seconds"]/attempted_pairs
    counts = s3.get("counts", actual_inference)
    denominators = {"physical_piece": request["case"].get("pieces"), "stage3_input_candidate_arc": counts.get("candidates"),
                    "generated_geometry": counts.get("geometries"), "retained_working_edge": counts.get("working_edges"),
                    "completed_expensive_pair": attempted_pairs}
    numerator = prof.get("timing", {}).get("inference_wall_seconds")
    value["stage3"]["normalized_inference_cost"] = {
        name: {"count": count, "seconds_per_unit_including_shared_costs": numerator/count if numerator is not None and count else None,
               "unavailable_reason": None if numerator is not None and count else "Missing inference duration or positive denominator"}
        for name, count in denominators.items()}
    atomic_json(run_root / "feedback.json", value)
    lines = ["# Eight-piece candidate feedback", "", "Automated measurements; primary-preview visual review is pending.", "",
             f"- Case: `{request['case_ids'][0]}`; candidate `exact-evidence-v1`.",
             f"- Serving: `{case_progress['status']}`; diagnostics: `{progress['phase']}`.",
             f"- Eight-piece control equivalence: `{value['eight_piece_control_equivalence']}`. No matching control was run automatically.",
             f"- Fresh complete end-to-end wall time: `{value['timing']['fresh_end_to_end_wall_seconds']}` seconds.",
             f"- Observed serving session: `{serving_wall}` seconds. Reused/failed attempts are not cold full-run timings.",
             "- Profiler inclusive function rows overlap; do not sum them or extrapolate them into a measured 30-piece result.", "",
             "## Stage process times", "", "| Stage/check | Status | Wall seconds |", "| --- | --- | ---: |"]
    for k, v in outer.items():
        lines.append(f"| {k} | {v.get('status', 'validated')} | {v.get('elapsed_seconds', 'unavailable')} |")
    lines += ["", "## Largest inclusive Stage 3 costs", "", "| Function | Calls | Inclusive seconds | Exclusive seconds |", "| --- | ---: | ---: | ---: |"]
    for f in value["largest_inclusive_costs_do_not_sum"]:
        lines.append(f"| `{f['function']}` | {f['calls']} | {f['inclusive_seconds']:.3f} | {f['exclusive_seconds']:.3f} |")
    lines += ["", "## Assembly and precision", "", *diagnostic.get("findings", []), "",
              "The 4px result describes precision. It is not a standalone structural/visual verdict. Connectivity and correct neighbor identities do not alone establish correct placement or orientation.",
              "The per-contact table below retains selected-path error as well as best direct poses. Counterfactual support compatibility has not been proven.", "",
              "| Contact | Archive best px | Pool best px | Selected-path source px | Final px | 4px localization |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for c in diagnostic.get("contact_diagnoses", []):
        lines.append(f"| {c['piece_a']} / {c['piece_b']} | {c['archive_best_p95_px']} | {c['working_best_p95_px']} | {(c.get('selected_path_comparison') or {}).get('selected_path_source_p95_px')} | {(c.get('final') or {}).get('p95_native_a_px')} | {c['diagnosis']} |")
    lines += ["", "## Manual review still required", "", value["manual_review"]["reason"], "",
              "## Raw artifacts", "", *[f"- {k}: `{v['path']}` ({'present' if v['exists'] else 'unavailable'})." for k, v in value["artifacts"].items()], "",
              "`feedback.json` contains exact identities, commands, counters, per-contact evidence, common-gauge placement errors, tolerance sensitivity, null reasons and all attempt flags.", ""]
    (run_root / "STAGE4_TO_STAGE3_CANDIDATE_FEEDBACK.md").write_text("\n".join(lines), encoding="utf-8")
    return value
