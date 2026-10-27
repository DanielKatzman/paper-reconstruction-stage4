"""Assigned-support ordering, deferred child selection and bounded lookahead.

Lookahead influences exploration only. Published states retain their actual
objective score, physical inventory and checked occurrence/support assignments.
"""
from collections import Counter
import time
from ..model import singleton_state
from .common import Budget, SearchResult, identity
from .component_beam import family_order, pose_signature, retain
from .joint_beam import topology


def next_contacts(search, state, merged_root, maximum):
    """Bound the inspected prefix; prefer a second contact inside the component.

    A failed/absent extra contact is neutral. Tree-only attachments stay legal.
    No neighbor oracle, continuous fitting or page rectangle is consulted.
    """
    selected_pairs = {(search.edges[e].a,search.edges[e].b) for e in state.selected}
    internal, external = [], []
    external_pairs=set();internal_pairs=set()
    inspected = 0
    for edge in family_order(search,state):
        # family_order enumerates merges only. Internal cycle candidates are
        # handled separately below, without scanning every pose on every pair.
        if inspected >= maximum*16:
            break
        inspected += 1
        if merged_root in (state.groups[edge.a],state.groups[edge.b]) and (edge.a,edge.b) not in external_pairs:
            external.append(edge)
            external_pairs.add((edge.a,edge.b))
    if not hasattr(search,'lookahead_contact_order'):
        search.lookahead_contact_order=tuple(sorted(search.candidates,key=lambda e:(-search.priorities[e.id],e.id)))
    for edge in search.lookahead_contact_order:
        if len(internal) >= maximum:
            break
        if ((edge.a,edge.b) not in selected_pairs and (edge.a,edge.b) not in internal_pairs and
                state.groups[edge.a] == state.groups[edge.b] == merged_root):
            internal.append(edge)
            internal_pairs.add((edge.a,edge.b))
    # Multiple poses of one cycle are correlated probes. When possible, give
    # one check to closure and another to attaching a new physical component.
    return ([internal[0],*external[:maximum-1]] if internal and external else (internal+external)[:maximum])


def choose_children(candidates, future_scores, cap):
    """Keep objective leaders plus lookahead alternatives; never overwrite scores."""
    ranked = sorted(range(len(candidates)),key=lambda i:(-candidates[i].score,identity(candidates[i])))
    chosen = ranked[:max(1,cap//2)]
    for i in sorted(ranked,key=lambda i:(-future_scores[i],-candidates[i].score,identity(candidates[i]))):
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= cap:
            break
    return chosen[:cap]


def expand(search, source, budget, quota, lookahead_remaining, child_cap):
    config = search.config
    ordered = list(family_order(search,source))
    pool_limit = child_cap*config.joint_component_pool_factor
    base_limit = max(1,quota//2) if lookahead_remaining else quota
    candidates, incoming, seen, families = [], [], set(), set()
    checked = duplicates = 0
    before = Counter(search.rejections)
    for edge in ordered:
        if checked >= base_limit or len(candidates) >= pool_limit or not budget.take():
            break
        checked += 1
        families.add((edge.a,edge.b,search.strata[edge.id]))
        candidate = search.attempt(source,edge)
        if candidate is None:
            continue
        signature = (topology(candidate,search.edges),pose_signature(candidate,search.problem))
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature); candidates.append(candidate); incoming.append(edge)
        search.remember(candidate,'deferred_component_merge')
    future = [s.score for s in candidates]
    best_next = [None]*len(candidates)
    contacts = [next_contacts(search,s,s.groups[e.a],config.joint_lookahead_checks)
                if lookahead_remaining and checked < quota and len(s.components()) > 1 else []
                for s,e in zip(candidates,incoming)]
    lookahead = valid_next = 0
    # Give every deferred family its first chance before giving any a second.
    for depth in range(config.joint_lookahead_checks):
        for i, state in enumerate(candidates):
            if checked+lookahead >= quota or lookahead >= lookahead_remaining:
                break
            if depth >= len(contacts[i]):
                continue
            if not budget.take():
                break
            lookahead += 1
            extra = search.attempt(state,contacts[i][depth])
            if extra is not None:
                valid_next += 1
                search.remember(extra,'component_lookahead')
                if extra.score > future[i]:
                    future[i], best_next[i] = extra.score, extra
        if budget.reason != 'exhausted_candidates':
            break
    chosen = choose_children(candidates,future,child_cap)
    extras = [best_next[i] for i in chosen if best_next[i] is not None]
    return [candidates[i] for i in chosen], extras, {
        'base_attempts':checked,'lookahead_attempts':lookahead,'valid_lookahead':valid_next,
        'distinct_valid_children_compared':len(candidates),'children_selected':len(chosen),
        'duplicate_relative_poses_skipped':duplicates,'attachment_pose_support_families_attempted':len(families),
        'untested_merge_candidates':max(0,len(ordered)-checked),
        'base_prefix_truncated':checked < len(ordered),'rejections':dict(search.rejections-before),
        'selected': [{'incoming_geometry_id':incoming[i].id,'actual_score':candidates[i].score,
                      'lookahead_best_score':future[i],'lookahead_extension_retained':best_next[i] is not None}
                     for i in chosen]}


def solve(search):
    problem, config = search.problem, search.config
    budget = Budget(config)
    if problem.deadline is not None:
        reserve=config.joint_finalize_seconds+(config.refine_seconds if config.refine else 0.)
        budget.start=min(budget.start,problem.deadline-reserve-config.max_seconds)
    cap=max(4,config.joint_rebuild_width)
    lookahead_limit=min(config.joint_lookahead_proposals,config.max_proposals//5)
    used_lookahead = 0
    frontier=[singleton_state(problem)]; pool=list(frontier)
    # This policy deliberately has the same cold-only semantics as its runner.
    if problem.initial_states:
        raise ValueError('Support-aware component search requires cold states')
    started=time.perf_counter(); traces=[]; examples=[]; totals=Counter()
    try:
        for depth in range(max(0,len(problem.pieces)-1)):
            children=[]; extensions=[]; previous=budget.proposals
            quota=max(1,(config.max_proposals-budget.proposals)//max(1,(len(problem.pieces)-1-depth)*len(frontier)))
            depth_totals=Counter()
            for source in frontier:
                if len(source.components()) == 1:
                    children.append(source)
                    continue
                selected, extra, trace = expand(search,source,budget,quota,
                    lookahead_limit-used_lookahead,cap)
                used_lookahead += trace['lookahead_attempts']
                children.extend(selected); extensions.extend(extra)
                for key,value in trace.items():
                    if isinstance(value,(int,bool)):
                        depth_totals[key] += value
                if len(examples) < 24:
                    examples.append({'depth':depth+1,'parent_selected_geometry_ids':list(source.selected),**trace})
                if budget.reason != 'exhausted_candidates':
                    break
            if children:
                frontier=retain(children,config.beam_width,search)
            # Actual checked lookahead extensions can reach the final pool;
            # projected scores never become state objective scores.
            pool=retain([*pool,*frontier,*extensions],config.beam_width,search)
            totals.update(depth_totals)
            traces.append({'depth':depth+1,'quota_per_parent':quota,'proposals':budget.proposals-previous,
                'frontier':len(frontier),'best_actual_score':max(s.score for s in pool),**dict(depth_totals)})
            if not children or budget.reason != 'exhausted_candidates':
                break
    except KeyboardInterrupt:
        budget.reason='interrupted'
    search.phase_seconds['component_merges_and_lookahead']=time.perf_counter()-started
    finalists={identity(s):s for s in [*pool,*frontier,singleton_state(problem)]}
    return SearchResult(sorted(finalists.values(),key=lambda s:(-s.score,identity(s))),{
        'search_policy':config.joint_search_policy,'proposals':budget.proposals,
        'candidate_edges':len(problem.edges),'searched_edges':len(search.candidates),
        'assignment_nodes':search.assignment_nodes,'proposal_strata':search.shortlist_audit,
        'invalid_support_occurrences':len(search.support.invalid),'phase_seconds':search.phase_seconds,
        'component_beam':{'depths':traces,'valid_children_per_state':cap,
            'deferred_valid_pool_per_state':cap*config.joint_component_pool_factor,
            'lookahead_proposal_limit':lookahead_limit,'lookahead_proposals':used_lookahead,
            'counts':dict(totals),'parent_examples':examples,'parent_example_limit':24,
            'lookahead_influences_exploration_only':True,'second_contact_required':False,
            'evaluation_truth_opened':False},
        'complete_candidate_records':search.completed,'complete_candidate_record_limit':128,
        'retained_complete_layouts':sum(len(s.components()) == 1 for s in finalists.values()),
        'retained_layouts_are_complete_k_best':False,
        'pruning':{'selection':'deferred distinct poses, actual objective leaders and bounded next-contact alternatives',
            'exhaustion_means':'bounded prefixes and deferred pools, not exhaustive global search'},
        'evaluation_truth_opened':False},budget.reason,dict(search.rejections))
