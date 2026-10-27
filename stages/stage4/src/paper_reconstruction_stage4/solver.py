import time
from .feasibility import Feasibility
from .model import singleton_state
from .search.common import identity
from .support import SupportIndex, contact_error
from .config import ATTACHMENT_OBJECTIVES


def solve(problem, config):
    config.validate()
    from .constraints import require_bound
    require_bound(problem, config)
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
    if config.page_policy != 'disabled.v1':
        from .page import make_page_context
        problem.page_context = make_page_context(problem, config, support)
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
        search = None
        context = problem.page_context
        result = context.solve_geometry(feasibility) if context is not None and hasattr(context, 'solve_geometry') else None
        if result is None:
            search = Search(problem, config, feasibility, support)
            result = search.solve()
        search_seconds = time.perf_counter()-search_start
        if hasattr(search,'ordering_audit'):
            result.diagnostics['proposal_ordering'] = search.ordering_audit
        if config.objective_version in ATTACHMENT_OBJECTIVES:
            from .attachments import without_join
            remaining = max(0, config.max_proposals-result.diagnostics.get('proposals',0))
            limit = min(config.attachment_deletion_checks,remaining)
            additions, checked, seen_deletions = [], [], set()
            deletion_edges = {e.id:e for e in problem.edges}
            for source in result.states[:config.portfolio_size*2]:
                terms = problem.structural.evaluate(source)
                evidence = terms['attachment_evidence']
                continuation = {j['geometry_id']:j['cost'] for j in terms.get('continuation_evidence',{}).get('joins',[])}
                for bridge in sorted(evidence['bridges'],key=lambda b:(-b['uncertainty_cost']-continuation.get(b['geometry_id'],0.),b['geometry_id'])):
                    if len(checked) >= limit or (problem.deadline is not None and time.perf_counter() >= problem.deadline-config.joint_finalize_seconds):
                        break
                    from .config import support_aware_search
                    canonical = config.objective_version == 'observable-joint.v8-continuation' or support_aware_search(config)
                    candidate = without_join(source,deletion_edges,bridge['geometry_id'],canonical=canonical)
                    key = identity(candidate)
                    if key in seen_deletions:
                        continue
                    seen_deletions.add(key)
                    candidate.score = problem.structural.evaluate(candidate)['score']
                    additions.append(candidate)
                    checked.append({'removed_geometry_id':bridge['geometry_id'],'source_score':source.score,
                        'candidate_score':candidate.score,'marginal_gain_from_join':source.score-candidate.score,
                        'poses_refitted':False})
                    if canonical:
                        checked[-1]['independent_component_gauges_restored'] = True
            from .search.common import retain
            candidates = [*result.states,*additions]
            result.states = retain(candidates,len(candidates))
            result.diagnostics['proposals'] = result.diagnostics.get('proposals',0)+len(checked)
            result.diagnostics['attachment_deletions'] = {'maximum_checks':config.attachment_deletion_checks,
                'checks':checked,'initial_pose_source':'cold observable search','evaluation_truth_opened':False}
        if problem.page_context is not None:
            if hasattr(problem.page_context,'deletion_audit'):
                problem.page_context.finalization_proposal_allowance=max(0,config.max_proposals-result.diagnostics.get('proposals',0))
            if hasattr(problem.page_context,'prioritize_finalists'):
                prioritize_finalists = problem.page_context.prioritize_finalists
            else:
                from .page import prioritize_finalists
            result.states, page_checks = prioritize_finalists(problem, config, result.states, feasibility)
            from .config import REPAIR_V18_POLICY
            if config.native_repair_policy == REPAIR_V18_POLICY:
                result.diagnostics['legacy_ordering_batch_termination'] = result.termination
                result.termination = ('verified_construction' if problem.page_context.geometry_finished else
                    'time_budget' if time.perf_counter() >= problem.deadline else 'exhausted_candidates')
            result.diagnostics['page_finalist_checks'] = page_checks
            if hasattr(problem.page_context,'deletion_audit'):
                extra_checks=len(problem.page_context.deletion_audit['checks'])
                result.diagnostics['proposals']=result.diagnostics.get('proposals',0)+extra_checks
                result.diagnostics['ambiguity_deletion_proposals']=extra_checks
            if hasattr(problem.page_context,'closure_audit'):
                closure=problem.page_context.closure_audit
                result.diagnostics['proposals']=result.diagnostics.get('proposals',0)+closure['generated_proposals']
                result.diagnostics['component_closure']=closure
                result.diagnostics['native_bridge_evidence']=[e.raw['native_bridge'] for e in problem.edges if 'native_bridge' in e.raw]
            if hasattr(problem.page_context,'repair_audit'):
                repair=problem.page_context.repair_audit
                result.diagnostics['proposals']=result.diagnostics.get('proposals',0)+repair['proposal_checks']
                result.diagnostics['native_repair']=repair
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
        execution = getattr(search,'execution',None)
        if execution is not None:
            execution.seconds['finalization'] = time.perf_counter()-refine_start
            result.diagnostics['execution_reuse'] = execution.report()
        # Never publish unchecked geometry if a final check or coarse beam fails.
        if not portfolio:
            portfolio = [singleton_state(problem)]
            reports = [feasibility.state(portfolio[0], final=True)]
            refinement = [{"attempted": False, "reason": "singleton_fallback"}]
        if problem.page_context is not None:
            context = problem.page_context
            utilities = [context.utility(s, context.features(s)) for s in portfolio]
            priorities = [context.closure_priority(s,context.features(s)) for s in portfolio]
            ordering = sorted(range(len(portfolio)), key=lambda i: (-priorities[i], -utilities[i], identity(portfolio[i])))
            portfolio, reports, refinement = ([values[i] for i in ordering] for values in (portfolio, reports, refinement))
            assessments = [context.assess(s, [other for other in result.states if other is not s]) for s in portfolio]
            scores = []
            for state in portfolio:
                utility = context.utility(state, context.features(state))
                scores.append({'source_v8_score': state.score, 'page_selection_utility': utility,
                               'closure_priority':context.closure_priority(state,context.features(state))})
                if 'geometry' in context.features(state):
                    scores[-1]['geometry'] = context.features(state)['geometry']
                state.score = utility
            result.diagnostics['reconstruction_assessment'] = assessments[0]
            result.diagnostics['layout_reconstruction_assessments'] = assessments
            result.diagnostics['page_selection'] = {'policy': config.page_policy, 'layout_scores': scores,
                'selection_policy':config.page_selection_policy,
                'selection_order':'descending closure priority, then utility; objective_score is secondary utility',
                'formula': 'v8_score - (pieces-1)*(2*unfilled + overlap + 0.15*unresolved_weak_bridge_fraction)',
                'partial_candidates': 'source score; no missing-area penalty', 'features': context.report()}
            if hasattr(context,'candidate_audit'):
                result.diagnostics['geometry_candidate_audit'] = context.candidate_audit
                result.diagnostics['page_selection'].update(
                    selection_order='geometry gate, complete supported closure, then contact/page utility',
                    partial_candidates='per-component page-area bound and weak-bridge cost; no missing-area penalty')
            ordering = list(range(len(portfolio)))
        else:
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
            "boundary_cache_hits": problem.structural.boundary_cache_hits,
            "boundary_cache_misses": problem.structural.boundary_cache_misses,
            "evaluation_truth_opened": False}
        if hasattr(problem.structural,'continuation_context'):
            result.diagnostics['local_continuation']=problem.structural.continuation_context.report()
    return result
