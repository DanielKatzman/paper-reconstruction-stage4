"""Two cold observable policies share one budget and one final objective."""
from collections import Counter, OrderedDict
import copy
from dataclasses import replace
import time
from ..model import singleton_state
from .common import SearchResult, identity, retain


def solve(search):
    if search.problem.initial_states:
        raise ValueError('Cold portfolio does not accept initial-result seeds')
    config, problem = search.config, search.problem
    reserve = config.joint_finalize_seconds+(config.refine_seconds if config.refine else 0.)
    deadline = min(time.perf_counter()+config.max_seconds,
                   problem.deadline-reserve if problem.deadline is not None else float('inf'))
    states = [singleton_state(problem)]; arms = []; proposals = 0; rejections = Counter(); records = []
    for name in ('v4_constructions','component_beam'):
        remaining_seconds = deadline-time.perf_counter()
        remaining_proposals = config.max_proposals-proposals
        if remaining_seconds <= .001 or remaining_proposals < 1:
            arms.append({'name':name,'executed':False,'reason':'shared_budget_exhausted'})
            continue
        first = name == 'v4_constructions'
        seconds = remaining_seconds*config.joint_portfolio_fraction if first else remaining_seconds
        # Reserve a proposal share for the other arm, reclaim all unused work.
        allowed = max(1,int(config.max_proposals*.65)) if first else remaining_proposals
        allowed = min(allowed,remaining_proposals)
        policy = 'diverse-constructions.v1' if first else config.joint_portfolio_component_policy
        options = dict(max_seconds=max(.001,seconds),max_proposals=allowed,joint_search_policy=policy,
                       joint_portfolio_fraction=.45,joint_portfolio_component_policy='component-beam.v2')
        if first:
            options.update(objective_version='observable-joint.v4',scale_coherence_weight=.12,
                scale_coherence_sigma=.30,scale_coherence_min_confidence=.40,
                attachment_uncertainty_cost=.10,attachment_axis_weight=.12,
                attachment_corroboration_discount=.75,attachment_deletion_checks=24)
            options.update(continuation_weight=1.25,continuation_min_coverage=.60,
                continuation_samples=65,continuation_image_side=768)
            from ..config import LOOKAHEAD_DEFAULTS
            options.update(LOOKAHEAD_DEFAULTS)
        arm_config = replace(config,**options).validate()
        arm_problem = replace(problem,deadline=time.perf_counter()+seconds+reserve)
        if first:
            scorer = copy.copy(problem.structural)
            scorer.problem, scorer.config = arm_problem, arm_config
            scorer.cache = OrderedDict(); scorer.attachment_cache = OrderedDict()
            scorer.evaluations = scorer.cache_hits = 0
            arm_problem.structural = scorer
            from .joint_beam import JointSearch
            arm_search = JointSearch(arm_problem,arm_config,search.feasibility,search.support,execution=search.execution)
        else:
            # Ordering is already built under the common objective. Shared
            # immutable observations and physical/support caches avoid a reload.
            arm_search = copy.copy(search)
            arm_search.problem, arm_search.config = arm_problem, arm_config
            arm_search.rejections = Counter(); arm_search.assignment_nodes = 0
            arm_search.phase_seconds = {}; arm_search.trace = []; arm_search.completed = []
        started = time.perf_counter()
        result = arm_search.solve()
        used = result.diagnostics['proposals']; proposals += used
        rejections.update(result.rejections)
        candidates = []
        for state in result.states:
            old_score = state.score
            state.score = problem.structural.evaluate(state)['score']
            states.append(state)
            candidates.append({'source_score':old_score,'common_score':state.score,
                               'components':[len(c) for c in state.components()],
                               'selected_geometry_ids':list(state.selected)})
        # Trace scores from the reference arm have another objective; keep them
        # inside the labelled arm rather than mixing them into common records.
        if not first:
            records.extend(result.diagnostics.get('complete_candidate_records',[]))
        arms.append({'name':name,'executed':True,'cold':True,'source_objective':arm_config.objective_version,
            'common_objective':config.objective_version,'policy':policy,'proposal_allowance':allowed,
            'seconds_allowance_including_ordering':seconds,'proposals':used,'elapsed_search_seconds':time.perf_counter()-started,
            'termination':result.termination,'candidates':candidates,'search':result.diagnostics})
        if result.termination == 'interrupted':
            break
    # No beam pruning across arms: every supplied finalist reaches the final
    # physical check. In particular a strong old-policy state cannot disappear.
    finalists = retain(states,len(states))
    termination = 'interrupted' if any(a.get('termination') == 'interrupted' for a in arms) else (
        'proposal_budget' if proposals >= config.max_proposals else
        'time_budget' if time.perf_counter() >= deadline else 'exhausted_candidates')
    return SearchResult(finalists,{'search_policy':'cold-portfolio.v1','proposals':proposals,
        'candidate_edges':len(problem.edges),'searched_edges':len(search.candidates),
        'assignment_nodes':sum(a.get('search',{}).get('assignment_nodes',0) for a in arms),
        'proposal_strata':search.shortlist_audit,'complete_candidate_records':records[:128],
        'complete_candidate_record_limit':128,'retained_complete_layouts':sum(len(s.components()) == 1 for s in finalists),
        'cold_portfolio':{'arms':arms,'shared_proposal_limit':config.max_proposals,
            'shared_search_seconds':config.max_seconds,'seconds_remaining':max(0.,deadline-time.perf_counter()),
            'common_objective':config.objective_version,'cross_arm_pruning':False,'initial_result_seeds':False},
        'evaluation_truth_opened':False},termination,dict(rejections))
