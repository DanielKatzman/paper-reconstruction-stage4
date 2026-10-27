"""Bounded observable boundary/appearance correction, including tree layouts."""
import time
import numpy as np
from scipy.optimize import least_squares
from .geometry import normalize
from .model import State


class RefinementBudget(Exception):
    pass


def refine_state(state, problem, config, feasibility, support):
    if not state.selected:
        return state, {"attempted": False, "reason": "no_selected_observable_contacts"}
    free = [p for p in sorted(state.poses) if p not in set(state.groups.values())]
    if not free:
        return state, {"attempted": False, "reason": "no_free_poses"}
    scales = np.tile([np.deg2rad(config.refine_rotation_degrees), config.refine_log_scale,
                      config.refine_translation_px, config.refine_translation_px], len(free))
    if np.any(scales <= 0):
        return state, {"attempted": False, "reason": "zero_correction_bounds"}
    started = time.perf_counter()
    scorer = problem.structural
    initial = scorer.evaluate(state)
    edges = {e.id: e for e in problem.edges}

    def candidate_for(parameters):
        poses = dict(state.poses)
        for p, (angle, log_scale, tx, ty) in zip(free, parameters.reshape(-1, 4), strict=True):
            c, s = np.cos(angle)*np.exp(log_scale), np.sin(angle)*np.exp(log_scale)
            correction = np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])
            poses[p] = normalize(correction @ state.poses[p])
        return State(poses, dict(state.groups), state.selected, state.tree, dict(state.assignments))

    def residual(parameters):
        if time.perf_counter()-started >= config.refine_seconds:
            raise RefinementBudget()
        candidate = candidate_for(parameters)
        boundary, appearance = scorer.boundary_residual(candidate, config.structural_refine_samples)
        values = [boundary*np.sqrt(config.structural_boundary_weight*len(state.selected)/max(1, len(boundary)))]
        if len(appearance):
            values.append(appearance*np.sqrt(config.structural_appearance_weight*len(state.selected)/len(appearance)))
        values.append(np.sqrt(config.refine_regularization)*parameters/scales)
        return np.concatenate(values)

    try:
        before = residual(np.zeros(len(scales)))
        fitted = least_squares(residual, np.zeros(len(scales)), bounds=(-scales, scales), loss="soft_l1",
                               x_scale=scales, max_nfev=config.refine_max_nfev)
        candidate = candidate_for(fitted.x)
        report = feasibility.state(candidate, final=True)
        assignments, reason, nodes = support.assign(candidate.selected, candidate.poses, edges)
        if assignments is not None:
            candidate.assignments = assignments
        after = scorer.evaluate(candidate)
        candidate.score = after["score"]
        accepted = report["valid"] and assignments is not None and candidate.score > state.score+1e-10
        return candidate if accepted else state, {"attempted": True, "accepted": accepted,
            "parameterization": "bounded_similarity_left_corrections", "objective": "observable_boundary_appearance_with_prior",
            "tree_layout_supported": True, "gauge_roots_fixed": sorted(set(state.groups.values())),
            "function_evaluations": int(fitted.nfev), "optimizer_status": int(fitted.status),
            "initial_residual_rms": float(np.sqrt(np.mean(before**2))), "final_residual_rms": float(np.sqrt(np.mean(fitted.fun**2))),
            "score_before": initial, "score_after": after, "feasibility_valid": report["valid"],
            "support_status": reason, "support_assignment_nodes": nodes,
            "elapsed_seconds": time.perf_counter()-started, "evaluation_truth_opened": False}
    except RefinementBudget:
        return state, {"attempted": True, "accepted": False, "reason": "refinement_time_budget", "elapsed_seconds": time.perf_counter()-started}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return state, {"attempted": True, "accepted": False, "reason": "invalid_observable_refinement", "detail": str(exc), "elapsed_seconds": time.perf_counter()-started}
