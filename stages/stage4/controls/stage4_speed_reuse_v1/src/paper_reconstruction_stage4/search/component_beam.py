"""Opt-in component merge beam with gauge-invariant pose diversity."""
from collections import defaultdict,Counter
import time
import numpy as np
from paper_reconstruction_contracts import map_points
from ..geometry import probes,relative
from ..model import singleton_state
from .common import Budget,SearchResult,identity
from .joint_beam import topology


def pose_signature(state, problem):
    rows = []
    for component in state.components():
        root = min(component);scale = max(1.,float(np.linalg.norm(problem.pieces[root].shape)))*.025
        rows.append(tuple(component))
        for pid in component:
            if pid != root:
                xy = map_points(probes(problem.pieces[pid].shape,2),relative(state.poses,root,pid))
                rows.append((pid,tuple(np.rint(xy.ravel()/scale).astype(int).tolist())))
    return tuple(rows)


def retain(states, width, search):
    unique = {identity(s):s for s in states}
    ranked = sorted(unique.values(),key=lambda s:(-s.score,identity(s)))
    if len(ranked) <= width:
        return ranked
    buckets = defaultdict(list)
    for state in ranked:
        buckets[(topology(state,search.edges),pose_signature(state,search.problem))].append(state)
    # Objective winner remains first; round robin prevents one pose/backbone
    # family from consuming every slot while other valid families remain.
    ordered = sorted(buckets,key=lambda k:(-buckets[k][0].score,k))
    selected = []
    for depth in range(max(len(v) for v in buckets.values())):
        for key in ordered:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                if len(selected) == width:
                    return sorted(selected,key=lambda s:(-s.score,identity(s)))
    return selected


def merge_order(search,state):
    by_merge = defaultdict(list)
    for edge in search.candidates:
        a,b = state.groups[edge.a],state.groups[edge.b]
        if a != b:
            by_merge[tuple(sorted((a,b)))].append(edge)
    for values in by_merge.values():
        values.sort(key=lambda e:(-search.priorities[e.id],e.id))
    keys = sorted(by_merge,key=lambda k:(-search.priorities[by_merge[k][0].id],k))
    # Visit different component attachments before taking every alternative of
    # the strongest local pair. No input pair/pose is declared wrong or removed.
    for depth in range(max((len(v) for v in by_merge.values()),default=0)):
        for key in keys:
            if depth < len(by_merge[key]):
                yield by_merge[key][depth]


def family_order(search, state):
    """Visit distinct physical attachments before more poses of the same pair."""
    families = defaultdict(list)
    for edge in merge_order(search,state):
        families[(edge.a,edge.b)].append(edge)
    keys = sorted(families,key=lambda k:(-search.priorities[families[k][0].id],k))
    for depth in range(max((len(v) for v in families.values()),default=0)):
        for key in keys:
            if depth < len(families[key]):
                yield families[key][depth]


def solve(search):
    problem,config = search.problem,search.config
    budget = Budget(config)
    if problem.deadline is not None:
        reserve = config.joint_finalize_seconds+(config.refine_seconds if config.refine else 0.)
        budget.start = min(budget.start,problem.deadline-reserve-config.max_seconds)
    controls = []
    for source in problem.initial_states:
        assigned,reason,nodes = search.support.assign(source.selected,source.poses,search.edges)
        search.assignment_nodes += nodes
        if assigned is not None and search.feasibility.state(source,final=True)["valid"]:
            source.assignments = assigned
            if hasattr(search.support,"bind"):search.support.bind(source)
            source.score = problem.structural.evaluate(source)["score"]
            controls.append(source)
    frontier = [singleton_state(problem)];pool = [*frontier,*controls]
    traces = [];accepted = 0;started = time.perf_counter()
    child_cap = max(4,config.joint_rebuild_width)
    diverse = config.joint_search_policy == 'component-beam.v2'
    try:
        for depth in range(max(0,len(problem.pieces)-1)):
            children = [];before = Counter(search.rejections);proposals_before = budget.proposals
            remaining_depth = max(1,len(problem.pieces)-1-depth)
            quota = max(1,(config.max_proposals-budget.proposals)//max(1,remaining_depth*len(frontier)))
            states_visited = 0; families_considered = set(); redundant = 0
            for source in frontier:
                successful = checked = 0;states_visited += 1; seen_children = set()
                order = family_order(search,source) if diverse else merge_order(search,source)
                for edge in order:
                    if checked >= quota or successful >= child_cap or not budget.take():break
                    checked += 1
                    if diverse:
                        families_considered.add((edge.a,edge.b,search.strata[edge.id][0]))
                    candidate = search.attempt(source,edge)
                    if candidate is not None:
                        if diverse:
                            signature = (topology(candidate,search.edges),pose_signature(candidate,problem))
                            if signature in seen_children:
                                redundant += 1
                                continue
                            seen_children.add(signature)
                        children.append(candidate);accepted += 1;successful += 1
                        search.remember(candidate,"component_merge_depth_"+str(depth+1))
                if budget.reason != "exhausted_candidates":break
            if children:
                frontier = retain(children,config.beam_width,search)
                pool = retain([*pool,*frontier],config.beam_width,search)
            traces.append({"depth":depth+1,"states_visited":states_visited,"quota_per_state":quota,
                "proposals":budget.proposals-proposals_before,"accepted_children":len(children),"frontier":len(frontier),
                "best_score":max(s.score for s in pool),"rejections":dict(search.rejections-before),
                "complete_children":sum(len(s.components()) == 1 for s in children)})
            if diverse:
                kept = {identity(s) for s in frontier}
                discarded = [s.score for s in children if identity(s) not in kept]
                traces[-1].update(attachment_pose_families_considered=len(families_considered),
                    accepted_redundant_children_skipped=redundant,best_discarded_score=max(discarded,default=None),
                    best_retained_score=max((s.score for s in frontier),default=None))
            if not children or budget.reason != "exhausted_candidates":break
    except KeyboardInterrupt:
        budget.reason = "interrupted"
    search.phase_seconds["component_merges"] = time.perf_counter()-started
    finalists = {identity(s):s for s in [*pool,*frontier,*controls,singleton_state(problem)]}
    finalists = sorted(finalists.values(),key=lambda s:(-s.score,identity(s)))
    return SearchResult(finalists,{"search_policy":config.joint_search_policy,"proposals":budget.proposals,
        "accepted_proposals":accepted,"construction_trials":0,"candidate_edges":len(problem.edges),"searched_edges":len(search.candidates),
        "assignment_nodes":search.assignment_nodes,"invalid_support_occurrences":len(search.support.invalid),
        "proposal_strata":search.shortlist_audit,"phase_seconds":search.phase_seconds,
        "component_beam":{"depths":traces,"valid_children_per_state":child_cap,"beam_width":config.beam_width,
                          "diversity":"physical backbone and gauge-invariant projected relative pose","evaluation_truth_opened":False},
        "complete_candidate_records":search.completed,"complete_candidate_record_limit":128,
        "retained_complete_layouts":sum(len(s.components()) == 1 for s in finalists),
        "retained_backbones":len({topology(s,search.edges) for s in finalists}),"protected_feasible_control_states":len(controls),
        "pruning":{"beam_width":config.beam_width,"selection":"objective-ranked backbone/relative-pose buckets",
                   "exhaustion_means":"bounded component expansions, not exhaustive global search"},"evaluation_truth_opened":False},
        budget.reason,dict(search.rejections))
