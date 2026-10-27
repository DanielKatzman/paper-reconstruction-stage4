"""Centered joint corrections optimizing the declared full observable score."""
import copy
import time
import numpy as np
from scipy.optimize import minimize
from paper_reconstruction_contracts import map_points
from .geometry import normalize
from .model import State


class JointRefinementLimit(Exception):
    pass


def corrected_state(state, problem, parameters, scales):
    roots = set(state.groups.values())
    free = [p for p in sorted(state.poses) if p not in roots]
    poses = dict(state.poses)
    for pid, values in zip(free, (parameters*scales).reshape(-1, 4), strict=True):
        angle, log_scale, tx, ty = values
        center = map_points([[problem.pieces[pid].shape[1]/2, problem.pieces[pid].shape[0]/2]], state.poses[pid])[0]
        c, s = np.exp(log_scale)*np.cos(angle), np.exp(log_scale)*np.sin(angle)
        linear = np.array([[c, -s], [s, c]])
        correction = np.eye(3); correction[:2, :2] = linear
        correction[:2, 2] = center+np.array([tx, ty])-linear@center
        poses[pid] = normalize(correction@state.poses[pid])
    return State(poses, dict(state.groups), state.selected, state.tree, dict(state.assignments), state.score, copy.deepcopy(state.active_support))


def refine_state(state, problem, config, feasibility, support):
    free = [p for p in sorted(state.poses) if p not in set(state.groups.values())]
    if not free or not state.selected:
        return state, {"attempted": False, "reason": "no_free_observed_poses"}
    scales = np.tile([np.deg2rad(config.refine_rotation_degrees), config.refine_log_scale,
                      config.refine_translation_px, config.refine_translation_px], len(free))
    if np.any(scales <= 0):
        return state, {"attempted": False, "reason": "zero_correction_bounds"}
    started = time.perf_counter(); scorer = problem.structural
    edges = {e.id: e for e in problem.edges}
    original_terms = scorer.evaluate(state); winner = state
    rounds, calls, exhausted = [], 0, False
    def check_time():
        if time.perf_counter()-started >= config.refine_seconds or (problem.deadline is not None and time.perf_counter() >= problem.deadline-config.joint_finalize_seconds):
            raise JointRefinementLimit()
    try:
        for _ in range(config.joint_refinement_rounds):
            seed = winner
            def loss(parameters):
                nonlocal calls
                check_time(); calls += 1
                proposed = corrected_state(seed, problem, parameters, scales)
                # Every selection term participates. The only additional term
                # is the explicitly reported correction-size prior.
                return -scorer.evaluate(proposed)["score"]+config.refine_regularization*float(np.dot(parameters, parameters))
            fit = minimize(loss, np.zeros(len(scales)), method="L-BFGS-B", bounds=[(-1., 1.)]*len(scales),
                           options={"maxiter": config.refine_max_nfev, "maxfun": config.refine_max_nfev*(len(scales)+1),
                                    "eps": 1e-3, "ftol": 1e-10, "maxls": 8})
            record = {"optimizer_status": int(fit.status), "optimizer_message": str(fit.message),
                      "bound_coordinates": int(np.sum(np.abs(fit.x) > .99)), "trials": []}
            for fraction in (1., .5, .25):
                check_time()
                candidate = corrected_state(seed, problem, fit.x*fraction, scales)
                report = feasibility.state(candidate, final=True)
                assigned, reason, nodes = support.assign(candidate.selected, candidate.poses, edges)
                if assigned is not None:
                    candidate.assignments = assigned
                    if hasattr(support, "bind"):
                        support.bind(candidate)
                terms = scorer.evaluate(candidate)
                candidate.score = terms["score"]
                accepted = report["valid"] and assigned is not None and candidate.score > winner.score+1e-10
                record["trials"].append({"fraction": fraction, "accepted": accepted, "physical_reason": report["reason"],
                    "support_reason": reason, "assignment_nodes": nodes, "score_terms": terms,
                    "correction_prior": config.refine_regularization*float(np.dot(fit.x*fraction, fit.x*fraction))})
                if accepted:
                    winner = candidate
            rounds.append(record)
            if winner is seed:
                break
    except JointRefinementLimit:
        exhausted = True
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        rounds.append({"reason": "invalid_joint_fit", "detail": str(exc)})
    return winner, {"attempted": True, "accepted": winner is not state, "objective": config.objective_version,
        "optimized_terms": "negative full declared observable score plus correction-size prior",
        "parameterization": "centered_similarity_left_corrections", "translation_units": "component anchor native pixels",
        "gauge_roots_fixed": sorted(set(state.groups.values())), "free_piece_ids": free,
        "function_evaluations": calls, "rounds": rounds, "budget_exhausted": exhausted,
        "score_before": original_terms, "score_after": scorer.evaluate(winner),
        "elapsed_seconds": time.perf_counter()-started, "evaluation_truth_opened": False}
