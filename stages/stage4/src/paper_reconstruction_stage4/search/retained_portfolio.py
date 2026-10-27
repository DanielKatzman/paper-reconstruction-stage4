"""Three cold policies; preserve feasible incumbents outside beam pruning."""
from collections import Counter, OrderedDict
import copy
from dataclasses import replace
import time
import numpy as np
from ..config import LOOKAHEAD_DEFAULTS,RETENTION_DEFAULTS
from ..model import singleton_state,State
from .common import SearchResult,retain
from .execution import clone
from .incumbents import Incumbents


def source_config(config,route,allowed,seconds):
    options=dict(max_proposals=allowed,max_seconds=max(.001,seconds),
        joint_portfolio_fraction=.45,joint_portfolio_component_policy='component-beam.v2',**RETENTION_DEFAULTS)
    if route=='v8_component':
        options.update(joint_search_policy='component-beam.v3-lookahead')
    else:
        options.update(joint_search_policy='diverse-constructions.v1' if route=='v4_constructions' else 'component-beam.v1',
            objective_version='observable-joint.v4' if route=='v4_constructions' else 'observable-joint.v6-attachment',
            continuation_weight=1.25,continuation_min_coverage=.60,continuation_samples=65,continuation_image_side=768,
            **LOOKAHEAD_DEFAULTS)
        if route=='v4_constructions':
            options.update(scale_coherence_weight=.12,scale_coherence_sigma=.30,scale_coherence_min_confidence=.40,
                attachment_uncertainty_cost=.10,attachment_axis_weight=.12,
                attachment_corroboration_discount=.75,attachment_deletion_checks=24)
    return replace(config,**options).validate()


def common_record(record,scorer,route,source_objective):
    poses={p:np.asarray(h,float) for p,h in record['native_to_component'].items()}
    state=State(poses,{p:min(poses) for p in poses},tuple(record['selected_geometry_ids']),
        tuple(record['selected_geometry_ids']),record['occurrence_ids'],record['score'],record.get('active_support',{}))
    return {**record,'origin':route+':'+record['origin'],'source_objective':source_objective,
        'source_score':record['score'],'score':scorer.evaluate(state)['score']}


def solve(search):
    if search.problem.initial_states:raise ValueError('Retained portfolio requires cold states')
    config,problem=search.config,search.problem
    reserve=config.joint_finalize_seconds+(config.refine_seconds if config.refine else 0.)
    deadline=min(time.perf_counter()+config.max_seconds,problem.deadline-reserve if problem.deadline else float('inf'))
    bank=Incumbents(search,deadline)
    states=[singleton_state(problem)];arms=[];proposals=0;v4_used=0;records=[];rejections=Counter()
    for name in ('v4_constructions','v6_component','v8_component'):
        remaining=config.max_proposals-proposals;seconds=deadline-time.perf_counter()
        if remaining<1 or seconds<=.001:
            arms.append({'name':name,'executed':False,'reason':'shared_budget_exhausted'});continue
        nominal=(max(1,int(config.joint_retention_route_proposals*.65)) if name=='v4_constructions'
                 else max(1,config.joint_retention_route_proposals-v4_used))
        allowed=min(nominal,remaining)
        seconds*=config.joint_portfolio_fraction if name=='v4_constructions' else .5 if name=='v6_component' else 1.
        arm_config=source_config(config,name,allowed,seconds)
        arm_problem=replace(problem,deadline=time.perf_counter()+seconds+reserve)
        started=time.perf_counter()
        if name=='v8_component':
            arm_search=copy.copy(search)
            arm_search.problem,arm_search.config=arm_problem,arm_config
            arm_search.rejections=Counter();arm_search.assignment_nodes=0
            arm_search.phase_seconds={};arm_search.trace=[];arm_search.completed=[]
        else:
            scorer=copy.copy(problem.structural)
            scorer.problem,scorer.config=arm_problem,arm_config
            scorer.cache=OrderedDict();scorer.attachment_cache=OrderedDict()
            scorer.evaluations=scorer.cache_hits=0
            arm_problem.structural=scorer
            from .joint_beam import JointSearch
            arm_search=JointSearch(arm_problem,arm_config,search.feasibility,search.support,execution=search.execution)
        route_bank=bank.route(name,arm_problem.deadline-reserve)
        arm_search.incumbents=route_bank
        result=arm_search.solve()
        used=result.diagnostics['proposals'];proposals+=used
        if name=='v4_constructions':v4_used=used
        rejections.update(result.rejections)
        candidates=[]
        for source in result.states:
            original_score=source.score
            route_bank.consider(source,'route_finalist')
            candidate=clone(source);candidate.score=problem.structural.evaluate(candidate)['score']
            states.append(candidate)
            candidates.append({'source_score':original_score,'common_score':candidate.score,
                'components':[len(c) for c in source.components()],'selected_geometry_ids':list(source.selected)})
        states.extend(route_bank.states)
        records.extend(common_record(r,problem.structural,name,arm_config.objective_version)
                       for r in result.diagnostics.get('complete_candidate_records',[]))
        arms.append({'name':name,'executed':True,'cold':True,'source_objective':arm_config.objective_version,
            'common_objective':config.objective_version,'policy':arm_config.joint_search_policy,
            'proposal_allowance':allowed,'nominal_route_allowance':nominal,'proposals':used,
            'seconds_allowance_including_ordering_and_checks':seconds,'elapsed_search_seconds':time.perf_counter()-started,
            'termination':result.termination,'candidates':candidates,'search':result.diagnostics})
        if result.termination=='interrupted':break
    if bank.page_bank is not None:
        states.extend(bank.page_bank.finalists())
    finalists=retain(states,len(states))
    termination=('interrupted' if any(a.get('termination')=='interrupted' for a in arms) else
                 'proposal_budget' if proposals>=config.max_proposals else 'time_budget' if time.perf_counter()>=deadline else 'exhausted_candidates')
    diagnostics={'search_policy':config.joint_search_policy,'proposals':proposals,
        'candidate_edges':len(problem.edges),'searched_edges':len(search.candidates),
        'assignment_nodes':sum(a.get('search',{}).get('assignment_nodes',0) for a in arms),
        'proposal_strata':search.shortlist_audit,'complete_candidate_records':records[:384],
        'complete_candidate_record_limit':384,'feasible_incumbents':bank.report(),
        'retained_complete_layouts':sum(len(s.components())==1 for s in finalists),
        'cold_portfolio':{'arms':arms,'shared_proposal_limit':config.max_proposals,'shared_search_seconds':config.max_seconds,
            'nominal_two_route_proposal_basis':config.joint_retention_route_proposals,
            'common_objective':config.objective_version,'cross_arm_pruning':False,'initial_result_seeds':False},
        'evaluation_truth_opened':False}
    if bank.page_bank is not None:
        diagnostics['page_candidate_retention']=bank.page_bank.report()
    return SearchResult(finalists,diagnostics,termination,dict(rejections))
