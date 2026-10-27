"""Offline-only measurements of already published predictions/candidate traces."""
import argparse
import copy
import html
from pathlib import Path
import time
import numpy as np
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.active_support import verify_result_support
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.outputs import read_result, serialize_layout
from evaluation.evaluate import evaluate
from .common import atomic_json, digest, file_hash, read_json
from .quality import assembly_quality, placement_overlay


def measure(result, reference):
    evaluation = evaluate(result, reference, 4.)
    quality = assembly_quality(result, reference, evaluation)
    return {"run_id": result["run_id"], "evaluation": evaluation, "quality": quality,
            "objective_score": result["layouts"][0]["objective_score"], "termination": result["termination"],
            "resources": result["resources"]}


def parity(before, after):
    first, second = before["layouts"][0], after["layouts"][0]
    a = {p["piece_id"]: p for p in first["pieces"]}
    delta = max(float(np.linalg.norm(map_points(p["native_probes"], p["native_to_component"])-
                                        map_points(p["native_probes"], a[p["piece_id"]]["native_to_component"]), axis=1).max()) for p in second["pieces"])
    ids = lambda x: {(j["geometry_id"], j["occurrence_id"]) for j in x["selected_joins"]}
    valid = ids(first) == ids(second) and first["components"] == second["components"] and delta <= 1e-6 and abs(first["objective_score"]-second["objective_score"]) <= 1e-9
    return {"passed": valid, "same_selected_geometry_occurrence_ids": ids(first) == ids(second),
            "maximum_probe_displacement_component_px": delta, "tolerance_component_px": 1e-6,
            "score_delta": second["objective_score"]-first["objective_score"]}


def audit_candidates(result, problem, config, reference, maximum=64, seconds=45., preview_path=None):
    started = time.perf_counter(); feasibility = Feasibility(problem, config)
    records = result["search"].get("complete_candidate_records", [])
    # Order is declared before any reference measurement; no feedback to serving.
    ranked = sorted(records, key=lambda r: (-r["score"], r["selected_geometry_ids"]))
    ordered, seen = [], set()
    for record in ranked:
        key = digest({k: record[k] for k in ("selected_geometry_ids", "occurrence_ids", "native_to_component", "active_support") if k in record})
        if key not in seen:
            seen.add(key); ordered.append(record)
    rows, rejected = [], []
    best_state, best_error = None, float("inf")
    for record in ordered[:maximum]:
        if time.perf_counter()-started >= seconds:
            break
        poses = {p: np.asarray(h, float) for p, h in record["native_to_component"].items()}
        state = State(poses, {p: min(poses) for p in poses}, tuple(record["selected_geometry_ids"]),
                      tuple(record["selected_geometry_ids"]), record["occurrence_ids"], record["score"], record.get("active_support", {}))
        physical = feasibility.state(state, final=True)
        if not physical["valid"]:
            rejected.append({"origin": record["origin"], "reason": physical["reason"]}); continue
        candidate = copy.deepcopy(result)
        candidate["layouts"] = [serialize_layout(problem, state, 0, physical, {"attempted": False})]
        try:
            verify_result_support(problem, candidate, config)
        except ValueError as exc:
            rejected.append({"origin": record["origin"], "reason": str(exc)}); continue
        measured = measure(candidate, reference)
        quality = measured["quality"]
        orientations = [abs(p["local_orientation_error_degrees"]) for p in quality["common_gauge"]["per_piece"]
                        if p.get("local_orientation_error_degrees") is not None]
        rows.append({"origin": record["origin"], "score": record["score"],
            "selected_true_neighbor_pairs": quality["selected_true_neighbor_pairs"],
            "selected_physical_pairs": quality["selected_physical_pairs"],
            "maximum_common_gauge_orientation_error_degrees": max(orientations, default=None),
            "placement_p95_reference_px": measured["quality"]["common_gauge"]["non_anchor_probe_errors_reference_px"]["p95"],
            "contact_median_native_a_px": measured["quality"]["contact_p95_error_distribution_native_a_px"]["p50"],
            "selected_geometry_ids": record["selected_geometry_ids"], "final_physical_and_support_valid": True})
        error = rows[-1]["placement_p95_reference_px"]
        if error < best_error:
            best_state, best_error = state, error
    diagnostic_preview = None
    if preview_path is not None and best_state is not None:
        from paper_reconstruction_stage4.preview import render
        from PIL import Image, ImageDraw
        preview_path = Path(preview_path)
        render(problem, best_state, preview_path)
        # Keep the selection method visible even when the image is shared alone.
        with Image.open(preview_path) as original:
            annotated = Image.new("RGB", (original.width, original.height+48), "white")
            annotated.paste(original, (0, 48))
        draw = ImageDraw.Draw(annotated)
        draw.text((12, 8), "DIAGNOSTIC ONLY - reference-selected existing serving candidate", fill="#a02020")
        draw.text((12, 25), "Not the committed prediction. No new poses/search; primary output unchanged.", fill="#a02020")
        annotated.save(preview_path)
        diagnostic_preview = {"path": str(preview_path), "sha256": file_hash(preview_path),
                              "selection": "Minimum common-gauge placement p95 among measured existing, physically/support-valid records"}
    return {"scope": "Measurement of existing serving traces only; no new placement proposals or primary selection",
            "records_available": len(records), "measured_valid": len(rows), "rejected": rejected,
            "unique_records_available": len(ordered), "exact_duplicate_records_skipped": len(records)-len(ordered),
            "maximum": maximum, "seconds_limit": seconds, "elapsed_seconds": time.perf_counter()-started,
            "truncated": len(rows)+len(rejected) < len(ordered), "candidates": rows,
            "diagnostic_preview": diagnostic_preview,
            "best_placement_diagnostic_only": min(rows, key=lambda r: r["placement_p95_reference_px"]) if rows else None}


def support_plot(problem, result, target):
    """Raw gray vs active green intervals in native contour arclength."""
    from paper_reconstruction_stage4.support import occurrence_arcs
    edges = {e.id: e for e in problem.edges}
    rows = []
    for join in result["layouts"][0]["selected_joins"]:
        occurrence = next(o for o in edges[join["geometry_id"]].occurrences if o["occurrence_id"] == join["occurrence_id"])
        raw = occurrence_arcs(problem, occurrence)
        for i, arc in enumerate(raw):
            xy = problem.pieces[arc.piece].contour["points_xy"]
            circumference = float(np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1).sum())
            rows.append((join, arc, join.get("active_support", {}).get("arcs", [None, None])[i], circumference))
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{55+len(rows)*35}" viewBox="0 0 1000 {55+len(rows)*35}">',
             '<rect width="100%" height="100%" fill="white"/><text x="15" y="22" font-family="sans-serif">Native contour intervals: raw gray; active green. Each row spans one full piece contour.</text>']
    for index, (join, raw, active, circumference) in enumerate(rows):
        y = 45+index*35
        label = f'{raw.piece}: {join["piece_a"]} / {join["piece_b"]}'
        items.append(f'<text x="15" y="{y+5}" font-size="12" font-family="sans-serif">{html.escape(label)}</text>')
        items.append(f'<path d="M300 {y} H980" stroke="#ddd"/>')
        for a,b in raw.intervals:
            items.append(f'<path d="M{300+680*a/circumference} {y} H{300+680*b/circumference}" stroke="#888" stroke-width="8"/>')
        if active:
            for a,b in active["intervals"]:
                items.append(f'<path d="M{300+680*a/circumference} {y} H{300+680*b/circumference}" stroke="#15803d" stroke-width="4"/>')
    target.write_text("\n".join([*items,"</svg>"]),encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    args=parser.parse_args(argv); root=args.run.resolve()
    request, progress=read_json(root/"request.json"), read_json(root/"progress.json")
    if request.get("schema_version") != "stage4.external-consumer-request.v1" or progress.get("phase") not in {"predicted","finished","diagnostics_failed"}:
        raise ValueError("Offline diagnosis requires a stopped, published external experiment")
    if request["request_sha256"] != digest({k:v for k,v in request.items() if k != "request_sha256"}):
        raise ValueError("External experiment request checksum differs")
    result, control=read_result(root/"stage4"),read_result(root/"control")
    if result["input"]["manifest_sha256"] != request["input_manifest_sha256"] or control["run_id"] != request["control_run_id"]:
        raise ValueError("Diagnostic source bindings differ")
    reference_path=Path(request["case_input"])/"evaluation_only/reference_geometry.json"
    reference=read_json(reference_path)
    # Native pixels were matched to the cohort before serving. The producer pool
    # manifest changed, so bind a separate in-memory evaluation copy explicitly.
    reference["input_manifest_sha256"] = request["input_manifest_sha256"]
    config=SolverConfig.from_dict(result["configuration"])
    problem=load_problem(root/"stage3",config)
    primary,baseline=measure(result,reference),measure(control,reference)
    support_check=verify_result_support(problem,result,config)
    overlay=placement_overlay(result,reference,Path(request["case_input"]),root/"stage3",root/"placement_overlay.png")
    support_plot(problem,result,root/"support_intervals.svg")
    comparison={"schema_version":"stage4.external-consumer-comparison.v1","variant":request["variant"],
        "request_sha256":request["request_sha256"],"input_manifest_sha256":request["input_manifest_sha256"],
        "control":baseline,"experiment":primary,"baseline_replay_parity":parity(control,result) if request["variant"]=="baseline" else None,
        "support_verification":support_check,"candidate_audit":audit_candidates(result,problem,config,reference),
        "placement_overlay":overlay,"reference_file_sha256":file_hash(reference_path),
        "diagnostic_code_sha256":{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('quality.py'),Path(__file__).parents[1]/'evaluation/evaluate.py']},
        "fresh_full_pipeline_timing_measured":False,"execution":progress,"manual_visual_assessment":None,
        "evaluation_after_prediction":True,"prediction_unchanged_by_diagnosis":True}
    atomic_json(root/"comparison.json",comparison)
    def p95(value):
        gauge=value["quality"]["common_gauge"]
        return gauge["non_anchor_probe_errors_reference_px"]["p95"] if gauge["whole_page_error_available"] else None
    lines=["# Consumer joint-assembly experiment","",f"Variant: `{request['variant']}`. Upstream evidence reused; no fresh full-pipeline timing.","",
        "| Measurement | Control | Experiment |","| --- | ---: | ---: |",
        f"| Official placement p95, reference px | {p95(baseline)} | {p95(primary)} |",
        f"| Median contact p95, native-A px | {baseline['quality']['contact_p95_error_distribution_native_a_px']['p50']} | {primary['quality']['contact_p95_error_distribution_native_a_px']['p50']} |",
        f"| Observable score | {baseline['objective_score']} | {primary['objective_score']} |","",
        "Inspect `stage4/preview.png`, `placement_overlay.png`, `support_intervals.svg` and `comparison.json`.",
        "Connectivity, a higher score or lower median seam error does not establish a solved page. Only the sealed primary is the serving prediction.",""]
    (root/"EXPERIMENT_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print("Post-prediction comparison published: "+str(root/"comparison.json"),flush=True)
    return 1 if comparison["baseline_replay_parity"] and not comparison["baseline_replay_parity"]["passed"] else 0
