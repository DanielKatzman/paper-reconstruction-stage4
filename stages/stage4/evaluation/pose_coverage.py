"""Reference-guided, offline pose-coverage diagnosis. Never a serving solver.

Candidates and all transforms come from the saved producer evidence. Reference
data rank them here solely to distinguish evidence/search/support limitations.
No output from this module is accepted as a sealed Stage 4 prediction or seed.
"""
from dataclasses import replace
import copy
import time
import numpy as np
from PIL import Image, ImageDraw
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import State, singleton_state
from paper_reconstruction_stage4.outputs import serialize_layout
from paper_reconstruction_stage4.preview import render
from paper_reconstruction_stage4.search.partial_beam import PartialSearch
from paper_reconstruction_stage4.support import SupportIndex
from benchmark.quality import assembly_quality
from evaluation.evaluate import evaluate


def physical_probes(problem, maximum=129):
    result = {}
    for pid, piece in problem.pieces.items():
        y, x = np.nonzero(piece.mask)
        indices = np.linspace(0, len(x)-1, min(maximum, len(x))).astype(int)
        result[pid] = np.column_stack((x[indices], y[indices])).astype(float)
    return result


def full_pose_error(edge, probes, reference_maps, shapes):
    """Bidirectional physical-piece error, beyond just points on one seam."""
    a, b = edge.a, edge.b
    expected = np.linalg.inv(reference_maps[a]) @ reference_maps[b]
    forward = np.linalg.norm(map_points(probes[b], edge.transform)-map_points(probes[b], expected), axis=1)
    reverse = np.linalg.norm(map_points(probes[a], np.linalg.inv(edge.transform))-
                             map_points(probes[a], np.linalg.inv(expected)), axis=1)
    ea, eb = float(np.quantile(forward, .95)), float(np.quantile(reverse, .95))
    return {"p95_native_a_px": ea, "p95_native_b_px": eb,
            "worst_fraction_native_crop_diagonal": max(ea/max(1., np.linalg.norm(shapes[a])), eb/max(1., np.linalg.norm(shapes[b])))}


def state_from_layout(layout):
    return State({p["piece_id"]: np.asarray(p["native_to_component"], float) for p in layout["pieces"]},
                 {p["piece_id"]: p["component_id"] for p in layout["pieces"]},
                 tuple(j["geometry_id"] for j in layout["selected_joins"]),
                 tuple(j["geometry_id"] for j in layout["selected_joins"] if j["tree_join"]),
                 {j["geometry_id"]: j["occurrence_id"] for j in layout["selected_joins"]}, layout["objective_score"],
                 {j["geometry_id"]: copy.deepcopy(j["active_support"]) for j in layout["selected_joins"] if j.get("active_support")})


def oracle_constructions(problem, config, candidates, primary, reference, output, trials=16, seconds=20.):
    """Bounded NEW reference-ranked combinations, explicitly separate from trace audits."""
    started = time.perf_counter()
    feasibility = Feasibility(problem, config)
    support = ActiveSupportIndex(problem, config) if config.support_policy == "active-subarc.v1" else SupportIndex(problem, config)
    problem.structural = JointScorer(problem, config, support)
    search = PartialSearch(problem, config, feasibility, support)
    baseline = state_from_layout(primary["layouts"][0])
    baseline_score = problem.structural.evaluate(baseline)["score"]
    rng = np.random.default_rng(20261023)
    best, best_quality, best_physical, best_error = None, None, None, float("inf")
    complete, valid, attempts, completed_trials = 0, 0, 0, 0
    largest = 1
    for trial in range(trials):
        if time.perf_counter()-started >= seconds:
            break
        state = singleton_state(problem)
        priorities = {e.id: error + (0. if trial == 0 else .02*rng.gumbel()) for e,error in candidates}
        for edge, _ in sorted(candidates, key=lambda item: (priorities[item[0].id], item[0].id)):
            if time.perf_counter()-started >= seconds:
                break
            if state.groups[edge.a] == state.groups[edge.b]:
                continue
            proposed = search.attempt(state, edge)
            attempts += 1
            if proposed is not None:
                state = proposed
                largest = max(largest, max(map(len,state.components())))
                if len(state.components()) == 1:
                    break
        completed_trials += 1
        if len(state.components()) != 1:
            continue
        complete += 1
        physical = feasibility.state(state, final=True)
        if not physical["valid"]:
            continue
        layout = serialize_layout(problem, state, 0, physical, {"attempted": False, "reason": "offline_oracle_has_no_continuous_refinement"})
        diagnostic = {"run_id": "oracle_diagnostic_not_a_serving_prediction", "evaluation_truth_opened": True, "layouts": [layout],
                      "input": {"manifest_sha256": problem.manifest["manifest_sha256"]}}
        try:
            verify_result_support(problem, diagnostic, config)
        except ValueError:
            continue
        valid += 1
        quality = assembly_quality(diagnostic, reference, evaluate(diagnostic, reference, 4.))
        error = quality["common_gauge"]["non_anchor_probe_errors_reference_px"]["p95"]
        if error < best_error:
            best, best_quality, best_physical, best_error = state, quality, physical, error
    preview = None
    if best is not None:
        render(problem, best, output)
        with Image.open(output) as original:
            annotated = Image.new("RGB", (original.width, original.height+48), "white")
            annotated.paste(original, (0,48))
        draw = ImageDraw.Draw(annotated)
        draw.text((12,8), "ORACLE DIAGNOSTIC - NEW reference-guided combination of saved poses", fill="#a02020")
        draw.text((12,25), "Not a prediction, not a serving seed, not evidence of deployable success.", fill="#a02020")
        annotated.save(output)
        preview = str(output)
    return {"evaluation_truth_opened": True, "new_diagnostic_combinations_searched": True,
            "continuous_pose_fitting_performed": False, "serving_primary_changed": False,
            "oracle_candidate_count": len(candidates), "trial_limit": trials, "seconds_limit": seconds,
            "trials_completed": completed_trials, "attempts": attempts, "complete_before_final_check": complete,
            "complete_final_physical_and_support_valid": valid, "largest_partial_component_seen": largest,
            "elapsed_seconds": time.perf_counter()-started,
            "time_limit_exhausted": completed_trials < trials, "rejections": dict(search.rejections),
            "primary_rescored_in_this_evidence_family": baseline_score,
            "best_reference_ranked_witness": None if best is None else {
                "observable_score": best.score, "quality": best_quality, "preview": preview,
                "layout": serialize_layout(problem,best,0,best_physical,{"attempted":False})},
            "limitation": "A found witness proves this supplied-pose combination is feasible. Failure in this bounded oracle search does not prove that no compatible assembly exists. Archive-family scores require the primary rescored with the same augmented evidence."}
