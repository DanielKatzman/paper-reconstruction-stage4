"""Bounded native-seam component merging; provisional extents are hypotheses."""
from dataclasses import dataclass,field
import time
import numpy as np
from paper_reconstruction_contracts import map_points
from .search.execution import clone
from .geometry import relative,regauge
from .native_bridge import pair_options
from .seam_poses import spread
from .outputs import digest
from .support import overlaps
from .component_closure import shape_measurement


@dataclass
class Node:
    state: object
    contacts: list=field(default_factory=list)
    value: float=0.


def key(node):
    return digest({'poses':{p:h.tolist() for p,h in sorted(node.state.poses.items())},
        'groups':node.state.groups,'contacts':[(v['pieces'],v['proposal_id']) for v in node.contacts]})


def extent_hypothesis(problem):
    side=0.;area=0.
    for piece in problem.pieces.values():
        y,x=np.nonzero(piece.mask);side=max(side,float(x.max()-x.min()+1),float(y.max()-y.min()+1));area+=len(x)
    return sorted([side,area/side])


def fits_extent(context,state,extent):
    # Conditional on this route's explicit spanning-piece hypothesis. It never
    # rejects an original solver state or asserts global impossibility.
    for group in state.components():
        xy=np.vstack([map_points(context.hulls[p],state.poses[p]) for p in group])
        wh=sorted(np.ptp(xy,axis=0)+1.)
        if any(a>b+2*context.config.contact_tolerance_px for a,b in zip(wh,extent)):return False
    return True


def attempt(context,node,proposal,physical,extent,component_bound=None):
    source=node.state;a,b=proposal['pieces']
    if source.groups[a]==source.groups[b]:return None,'already_connected'
    h=np.asarray(proposal['transform_b_to_a'],float)
    transform=source.poses[a]@h@np.linalg.inv(source.poses[b])
    moving={p for p in source.poses if source.groups[p]==source.groups[b]}
    candidate=clone(source)
    for p in moving:candidate.poses[p]=transform@source.poses[p]
    members=moving|{p for p in source.poses if source.groups[p]==source.groups[a]}
    for p in members:candidate.groups[p]=min(members)
    candidate.poses=regauge(candidate.poses,members)
    if component_bound is None:
        if not fits_extent(context,candidate,extent):return None,'provisional_extent'
    elif not component_bound(candidate):
        return None,'component_area_bound'
    occupied=[]
    for eid in source.selected:occupied.extend(context.support.arcs_for(source,eid))
    for contact in node.contacts:occupied.extend(contact['arcs'])
    options=pair_options(context.problem,a,b,candidate.poses,context.config,occupied)
    if not options:return None,'native_support'
    check=physical.state(candidate,final=False)
    if not check['valid']:return None,check['reason']
    arcs,metrics=options[0];length=min(v.length for v in arcs);condition=spread(arcs)
    value=length*(.1+min(condition,.35))/(1.+metrics['bidirectional_contact_p95_native_px']/4.)
    contact={'pieces':[a,b],'arcs':arcs,'proposal_id':proposal['proposal_id'],
        'proposed_transform_b_to_a':proposal['transform_b_to_a'],
        'source_groups':source.components(),'merged_groups':[
            sorted(p for p in source.poses if source.groups[p]==source.groups[a]),
            sorted(p for p in source.poses if source.groups[p]==source.groups[b])],
        'role':'merge','metrics':metrics,'spread':condition}
    return Node(candidate,[*node.contacts,contact],node.value+value),'accepted'


def search(context,source,proposals,deadline,maximum_checks=4000,beam=24,component_bound=None):
    from .feasibility import Feasibility
    from collections import Counter
    started=time.perf_counter();physical=Feasibility(context.problem,context.config)
    extent=extent_hypothesis(context.problem);nodes=[Node(clone(source))];complete=[];checks=0;rejections=Counter()
    ordered=sorted(proposals,key=lambda p:(-p['score'],p['proposal_id']))
    depth=0
    while nodes and depth<len(source.components())-1:
        next_nodes=[];seen=set()
        for node in nodes:
            accepted=0
            for proposal in ordered:
                if node.state.groups[proposal['pieces'][0]]==node.state.groups[proposal['pieces'][1]]:continue
                if checks>=maximum_checks or time.perf_counter()>=deadline:break
                checks+=1;candidate,reason=attempt(context,node,proposal,physical,extent,component_bound)
                rejections[reason]+=1
                if candidate is None:continue
                identifier=key(candidate)
                if identifier in seen:continue
                seen.add(identifier);accepted+=1
                if len(candidate.state.components())==1:
                    shape=shape_measurement(context,candidate.state,256)
                    if shape['shape_gate_pass']:complete.append((candidate,shape))
                    else:rejections['complete_shape']+=1
                else:next_nodes.append(candidate)
                if accepted>=16:break
            if checks>=maximum_checks or time.perf_counter()>=deadline:break
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
        nodes=keep;depth+=1
        if checks>=maximum_checks or time.perf_counter()>=deadline:break
    complete.sort(key=lambda n:(-n[1]['score'],-n[0].value,key(n[0])))
    return complete,{'policy':'native-seam-component-beam.v1','source_groups':source.components(),
        'merge_checks':checks,'maximum_merge_checks':maximum_checks,'beam':beam,'completed_depth':depth,
        'complete_shape_candidates':len(complete),'rejections':dict(rejections),
        'elapsed_seconds':time.perf_counter()-started,'budget_abstained':checks>=maximum_checks or time.perf_counter()>=deadline,
        'provisional_page_sides_native_px':extent,'spanning_piece_hypothesis_not_universal':True}
