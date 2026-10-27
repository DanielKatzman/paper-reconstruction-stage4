"""Observable uncertainty on selected bridges; deleting joins never refits poses."""
from collections import OrderedDict
from dataclasses import replace
import numpy as np
from .structural import jacobian_axes
from .support import overlaps


def cut_groups(state, edges, removed):
    adjacency = {p: set() for p in state.poses}
    for eid in state.selected:
        if eid != removed:
            edge = edges[eid]
            adjacency[edge.a].add(edge.b); adjacency[edge.b].add(edge.a)
    groups = {}
    for root in sorted(adjacency):
        if root in groups:
            continue
        todo = [root]
        while todo:
            pid = todo.pop()
            if pid in groups:
                continue
            groups[pid] = root
            todo.extend(adjacency[pid] - groups.keys())
    return groups


def without_join(state, edges, eid):
    return replace(state, groups=cut_groups(state, edges, eid),
        selected=tuple(e for e in state.selected if e != eid), tree=tuple(e for e in state.tree if e != eid),
        assignments={e:o for e,o in state.assignments.items() if e != eid},
        active_support={e:a for e,a in state.active_support.items() if e != eid})


def evaluate(scorer, state, base):
    # Exact pose bytes avoid reusing a support decision for rounded nearby poses.
    key = (tuple((p,state.groups[p],h.tobytes()) for p,h in sorted(state.poses.items())),
           tuple(sorted(state.selected)),tuple(sorted(state.assignments.items())),
           tuple(sorted((e,a['support_id']) for e,a in state.active_support.items())))
    if not hasattr(scorer, 'attachment_cache'):
        scorer.attachment_cache = OrderedDict()
    if key in scorer.attachment_cache:
        return scorer.attachment_cache[key]
    reports = []; total = 0.
    for eid in state.selected:
        edge = scorer.edges[eid]
        occurrence = next((o for o in edge.occurrences if o['occurrence_id'] == state.assignments.get(eid)), None)
        alignment = (occurrence or edge.raw).get('pose',{}).get('alignment',{})
        if not alignment.get('underconstrained',False):
            continue
        groups = cut_groups(state, scorer.edges, eid)
        if groups[edge.a] == groups[edge.b]:
            continue  # Selected support already closes an assigned cycle.
        witnesses = []
        for candidate in sorted(base['corroborating_unselected_pairs'],key=lambda c:(c['crop_probe_rms_px'],c['geometry_id'])):
            a,b = candidate['pieces']
            if state.groups[a] != state.groups[edge.a] or groups[a] == groups[b]:
                continue
            added = candidate['geometry_id']
            assignments, reason, nodes = scorer.support.assign(tuple(sorted((*state.selected,added))),state.poses,scorer.edges)
            compatible = assignments is not None and added in assignments
            if compatible:
                if hasattr(scorer.support,'last_active') and added in scorer.support.last_active:
                    from .active_support import record_arc
                    extra_arcs = tuple(record_arc(arc) for arc in scorer.support.last_active[added]['arcs'])
                else:
                    extra_arcs = scorer.support.arcs[assignments[added]]
                # The assignment search may have changed old occurrences or
                # trimmed old supports to make room. Credit only a contact
                # compatible with the supports occupied by this actual state.
                for selected in state.selected:
                    if selected not in state.assignments:
                        compatible = False; reason = 'unassigned_existing_support'; break
                    old_arcs = (scorer.support.arcs_for(state,selected) if hasattr(scorer.support,'arcs_for') else
                                scorer.support.arcs[state.assignments[selected]])
                    if any(overlaps(first,second,scorer.config) for first in extra_arcs for second in old_arcs):
                        compatible = False; reason = 'existing_state_support_occupied'; break
            witnesses.append({'geometry_id':added,'pieces':[a,b],'support_assignable':compatible,
                              'reason':reason,'assignment_nodes':nodes})
            if compatible or len(witnesses) >= 2:
                break
        corroborated = any(w['support_assignable'] for w in witnesses)
        profiles = [scorer.profiles[p] for p in (edge.a,edge.b)]
        confidence = min(p['axis_confidence'] for p in profiles)
        axis_cost = 0.; angle = None
        if confidence >= .5:
            axes = [jacobian_axes(state.poses[p],scorer.problem.pieces[p].shape,profile['angle_radians'])[0]
                    for p,profile in zip((edge.a,edge.b),profiles)]
            a,b = (np.arctan2(u[1],u[0]) for u in axes)
            angle = abs(float(.5*np.angle(np.exp(2j*(a-b)))))
            axis_cost = confidence*min((angle/np.deg2rad(scorer.config.structural_angle_sigma_degrees))**2,1.)
        discount = 1.-scorer.config.attachment_corroboration_discount if corroborated else 1.
        cost = discount*(scorer.config.attachment_uncertainty_cost+scorer.config.attachment_axis_weight*axis_cost)
        total += cost
        reports.append({'geometry_id':eid,'pieces':[edge.a,edge.b],'underconstrained':True,
            'shape_anisotropy':alignment.get('shape_anisotropy'),'fitted_overlap_length_query_px':alignment.get('fitted_overlap_length_query_px'),
            'axis_confidence':confidence,'axis_disagreement_degrees':None if angle is None else float(np.rad2deg(angle)),
            'assignable_corroboration':corroborated,'witnesses':witnesses,'uncertainty_cost':cost,
            'hard_rejection':False})
    value = {'cost':total,'bridges':reports,'confidence_calibrated':False,
             'policy':'conditioned-bridge-uncertainty.v1','independent_support_required_for_join':False}
    if len(scorer.attachment_cache) >= scorer.config.maximum_cache_entries:
        scorer.attachment_cache.popitem(last=False)
    scorer.attachment_cache[key] = value
    return value
