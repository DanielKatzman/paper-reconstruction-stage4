import time
from .feasibility import Feasibility
from .model import singleton_state
from .search.common import identity
from .support import SupportIndex, contact_error


def solve(problem, config):
    config.validate()
    start = time.perf_counter()
    if config.solver == "joint_beam_v3" and problem.deadline is None:
        problem.deadline = start+config.joint_wall_seconds
    feasibility = Feasibility(problem, config)
    feasibility_setup = time.perf_counter()-start
    support_start = time.perf_counter()
    if config.support_policy == "active-subarc.v1":
        from .active_support import ActiveSupportIndex
        support = ActiveSupportIndex(problem, config)
    else:
        support = SupportIndex(problem, config)
    support_setup = time.perf_counter()-support_start
    if config.solver == "joint_beam_v3":
        from .joint_score import JointScorer
        problem.structural = JointScorer(problem, config, support)
    elif config.solver == "structural_beam_v2":
        from .structural import StructuralScorer
        problem.structural = StructuralScorer(problem, config, support)
    if config.solver == "legacy_forest_control_v1":
        from .search.legacy_forest import solve as legacy_solve
        result = legacy_solve(problem, config)
        # Preserve the raw control winner even if stronger checks disagree.
        portfolio = result.states[:config.portfolio_size]
        reports = [feasibility.state(state, final=True) for state in portfolio]
        refinement = [{"attempted": False, "reason": "legacy_objective_preserved"} for _ in portfolio]
    else:
        if config.solver == "joint_beam_v3":
            from .search.joint_beam import JointSearch as Search
        elif config.solver == "structural_beam_v2":
            from .search.structural_beam import StructuralSearch as Search
        else:
            from .search.partial_beam import PartialSearch as Search
        search_start = time.perf_counter()
        result = Search(problem, config, feasibility, support).solve()
        search_seconds = time.perf_counter()-search_start
        portfolio, reports, refinement = [], [], []
        rejected_finalists = 0
        refine_start = time.perf_counter()
        seen = set()
        for state in result.states:
            if len(portfolio) >= config.portfolio_size:
                break
            report = feasibility.state(state, final=True)
            if not report["valid"]:
                rejected_finalists += 1
                continue
            refined = {"attempted": False, "reason": "disabled"}
            if config.refine and result.termination != "interrupted" and time.perf_counter() - refine_start < config.refine_seconds:
                from dataclasses import replace
                if config.solver == "joint_beam_v3" and config.joint_refinement:
                    from .refine_joint import refine_state
                elif config.solver in {"structural_beam_v2", "joint_beam_v3"}:
                    from .refine_observable import refine_state
                else:
                    from .refine import refine_state
                remaining = config.refine_seconds - (time.perf_counter() - refine_start)
                state, refined = refine_state(state, problem, replace(config, refine_seconds=max(.001, remaining)), feasibility, support)
                report = feasibility.state(state, final=True)
            key = identity(state)
            if key not in seen:
                seen.add(key)
                portfolio.append(state)
                reports.append(report)
                refinement.append(refined)
        result.diagnostics["final_feasibility_rejections"] = rejected_finalists
        result.diagnostics["search_and_ordering_seconds"] = search_seconds
        # Never publish unchecked geometry if a final check or coarse beam fails.
        if not portfolio:
            portfolio = [singleton_state(problem)]
            reports = [feasibility.state(portfolio[0], final=True)]
            refinement = [{"attempted": False, "reason": "singleton_fallback"}]
        ordering = sorted(range(len(portfolio)), key=lambda i: (-portfolio[i].score, identity(portfolio[i])))
        portfolio, reports, refinement = ([values[i] for i in ordering] for values in (portfolio, reports, refinement))
    result.states = portfolio
    result.diagnostics.update({"elapsed_solver_seconds": time.perf_counter() - start,
                               "collision_cache_hits": feasibility.cache_hits,
                               "invalid_support_examples": dict(list(sorted(support.invalid.items()))[:10]),
                               "final_feasibility": reports, "refinement": refinement,
                               "retained_layouts_are_complete_k_best": False})
    result.diagnostics["setup_phase_seconds"] = {"feasibility": feasibility_setup, "support_index": support_setup}
    if hasattr(support, "report"):
        result.diagnostics["support_policy"] = support.report()
    if config.solver == "joint_beam_v3":
        result.diagnostics["serving_budget"] = {"wall_seconds": config.joint_wall_seconds,
            "finalization_reserve_seconds": config.joint_finalize_seconds,
            "seconds_remaining_before_publication": problem.deadline-time.perf_counter(),
            "enforcement": "cooperative checks; input validation and an in-flight final check may overrun"}
    if config.solver in {"structural_beam_v2", "joint_beam_v3"}:
        result.diagnostics["observable_structure"] = {
            "version": config.objective_version, "native_image_profiles": problem.structural.profile_report(),
            "profile_setup_seconds": problem.structural.setup_seconds,
            "layout_score_terms": [problem.structural.evaluate(s) for s in portfolio],
            "score_evaluations": problem.structural.evaluations, "score_cache_hits": problem.structural.cache_hits,
            "evaluation_truth_opened": False}
    return result
