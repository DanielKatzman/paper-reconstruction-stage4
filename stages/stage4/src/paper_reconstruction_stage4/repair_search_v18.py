"""Versioned bounded search with per-lane pruning diagnostics and shared physics."""
from collections import defaultdict
import time
from .repair_search import Node, key, extent_hypothesis, attempt
from .search.execution import clone
from .component_closure import shape_measurement


def fair_order(proposals,state):
    buckets=defaultdict(list)
    for proposal in proposals:
        a,b=proposal['pieces'];group=tuple(sorted((state.groups[a],state.groups[b])))
        if group[0]!=group[1]:buckets[group].append(proposal)
    for values in buckets.values():
        values.sort(key=lambda p:(not p['conditioned'],-p['score'],p['proposal_id']))
    ordered=sorted(buckets,key=lambda g:(not buckets[g][0]['conditioned'],-buckets[g][0]['score'],g))
    for depth in range(max(map(len,buckets.values()),default=0)):
        for group in ordered:
            if depth<len(buckets[group]):yield buckets[group][depth]


def search(context,source,proposals,deadline,maximum_checks=None,beam=24,area_bound=False,physical=None,on_complete=None):
    from .feasibility import Feasibility
    from collections import Counter
    started=time.perf_counter();physical=physical or Feasibility(context.problem,context.config)
    component_bound=(lambda state:context.component_bounds(state)['valid']) if area_bound else None
    pruning=Counter()
    extent=extent_hypothesis(context.problem);nodes=[Node(clone(source))];complete=[];checks=0;rejections=Counter()
    ordered=sorted(proposals,key=lambda p:(-p['score'],p['proposal_id']))
    depth=0
    while nodes and depth<len(source.components())-1:
        next_nodes=[];seen=set()
        for node in nodes:
            accepted=0
            for proposal in ordered:
                if node.state.groups[proposal['pieces'][0]]==node.state.groups[proposal['pieces'][1]]:continue
                if (maximum_checks is not None and checks>=maximum_checks) or time.perf_counter()>=deadline:break
                checks+=1;candidate,reason=attempt(context,node,proposal,physical,extent,component_bound)
                rejections[reason]+=1
                if candidate is None:continue
                identifier=key(candidate)
                if identifier in seen:continue
                seen.add(identifier);accepted+=1
                if len(candidate.state.components())==1:
                    shape=shape_measurement(context,candidate.state,256)
                    if shape['shape_gate_pass']:
                        complete.append((candidate,shape))
                        if on_complete is not None:
                            accepted_candidate = on_complete(candidate, shape)
                            if accepted_candidate is not None:
                                return accepted_candidate, {'policy': 'geometry-checked-component-ordering.v1', 'merge_checks': checks, 'maximum_merge_checks': None, 'budget_abstained': False, 'elapsed_seconds': time.perf_counter()-started, 'stop_reason': 'verified_construction', 'complete_shape_candidates': len(complete)}
                    else:rejections['complete_shape']+=1
                else:next_nodes.append(candidate)
                if accepted >= 16:
                    pruning['successor_limit_hits']+=1;break
            if (maximum_checks is not None and checks>=maximum_checks) or time.perf_counter()>=deadline:break
        # Protect distinct component partitions before filling by native score.
        next_nodes.sort(key=lambda n:(-n.value,key(n)));keep=[];partitions=set()
        for node in next_nodes:
            partition=tuple(tuple(g) for g in node.state.components())
            if partition in partitions:continue
            partitions.add(partition);keep.append(node)
            if len(keep)>=beam//2:break
        used={key(n) for n in keep}
        for node in next_nodes:
            if len(keep)>=beam:break
            k=key(node)
            if k not in used:keep.append(node);used.add(k)
        pruning['beam_nodes_dropped']+=max(0,len(next_nodes)-len(keep))
        pruning['peak_successors']=max(pruning['peak_successors'],len(next_nodes))
        nodes=keep;depth+=1
        if (maximum_checks is not None and checks>=maximum_checks) or time.perf_counter()>=deadline:break
    complete.sort(key=lambda n:(-n[1]['score'],-n[0].value,key(n[0])))
    return complete,{'policy':'legacy-order-native-component-beam.v4','pruning':dict(pruning),
        'component_bound_policy':'observed-total-area.v1' if area_bound else 'conditional-spanning-piece-extent.v1',
        'provisional_extent_enforced':not area_bound,'successor_limit':16,'source_groups':source.components(),
        'merge_checks':checks,'maximum_merge_checks':maximum_checks,'beam':beam,'completed_depth':depth,
        'complete_shape_candidates':len(complete),'rejections':dict(rejections),
        'elapsed_seconds':time.perf_counter()-started,'budget_abstained':(maximum_checks is not None and checks>=maximum_checks) or time.perf_counter()>=deadline,
        'provisional_page_sides_native_px':extent,'spanning_piece_hypothesis_not_universal':True}
