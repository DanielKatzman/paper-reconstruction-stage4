"""Bounded similarity corrections preserve complete projective input seeds."""
from dataclasses import replace
import time
import numpy as np
from scipy.optimize import least_squares
from paper_reconstruction_contracts import map_points
from .geometry import normalize, probes, relative
from .model import State
from .search.common import objective


class RefinementBudget(Exception):
    pass


def refine_state(state, problem, config, feasibility, support):
    if not state.selected or len(state.selected) <= len(state.poses) - len(state.components()):
        return state, {"attempted": False, "reason": "no_redundant_selected_contacts"}
    edges = {e.id: e for e in problem.edges}
    roots = set(state.groups.values())
    free = [p for p in sorted(state.poses) if p not in roots]
    if not free:
        return state, {"attempted": False, "reason": "no_free_poses"}
    start = time.perf_counter()
    scales = np.tile([np.deg2rad(config.refine_rotation_degrees), config.refine_log_scale,
                      config.refine_translation_px, config.refine_translation_px], len(free))
    if np.any(scales <= 0):
        return state, {"attempted": False, "reason": "zero_correction_bounds"}

    def poses_for(parameters):
        poses = dict(state.poses)
        for p, (angle, log_scale, tx, ty) in zip(free, parameters.reshape(-1, 4), strict=True):
            c, s = np.cos(angle) * np.exp(log_scale), np.sin(angle) * np.exp(log_scale)
            correction = np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])
            poses[p] = normalize(correction @ state.poses[p])
        return poses

    def residual(parameters):
        if time.perf_counter() - start >= config.refine_seconds:
            raise RefinementBudget()
        poses = poses_for(parameters)
        values = []
        for eid in state.selected:
            edge = edges[eid]
            h = relative(poses, edge.a, edge.b)
            for first, second, shape in ((h, edge.transform, problem.pieces[edge.b].shape),
                                         (np.linalg.inv(h), np.linalg.inv(edge.transform), problem.pieces[edge.a].shape)):
                points = probes(shape)
                values.extend(((map_points(points, first) - map_points(points, second)) / config.cycle_tolerance_px).ravel())
        values.extend(np.sqrt(config.refine_regularization) * parameters / scales)
        return np.asarray(values)

    try:
        initial_residual = residual(np.zeros(len(scales)))
        fitted = least_squares(residual, np.zeros(len(scales)), bounds=(-scales, scales), loss="soft_l1",
                               max_nfev=config.refine_max_nfev, x_scale=scales)
        candidate = State(poses_for(fitted.x), dict(state.groups), state.selected, state.tree, dict(state.assignments))
        report = feasibility.state(candidate, final=True)
        assignments, reason, _ = support.assign(candidate.selected, candidate.poses, edges)
        candidate.score = objective(candidate, problem, edges, config)
        accepted = report["valid"] and assignments is not None and candidate.score > state.score + 1e-12
        if accepted:
            candidate.assignments = assignments
        return candidate if accepted else state, {
            "attempted": True, "accepted": accepted, "parameterization": "bounded_similarity_left_corrections",
            "initial_rms_normalized": float(np.sqrt(np.mean(initial_residual ** 2))),
            "final_rms_normalized": float(np.sqrt(np.mean(fitted.fun ** 2))),
            "function_evaluations": int(fitted.nfev), "optimizer_status": int(fitted.status),
            "feasibility_valid": report["valid"], "support_status": reason,
            "elapsed_seconds": time.perf_counter() - start, "confidence_calibrated": False}
    except RefinementBudget:
        return state, {"attempted": True, "accepted": False, "reason": "refinement_time_budget"}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return state, {"attempted": True, "accepted": False, "reason": "invalid_refinement", "detail": str(exc)}
