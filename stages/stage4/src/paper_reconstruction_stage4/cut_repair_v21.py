"""Move an observed unsupported cut using existing relative-pose hypotheses."""
from collections import Counter
import numpy as np
from .native_graph_v17 import quality
from .native_bridge import contour_arc, check_arcs
from .constraints import require_rigid
from .geometry import regauge
from .repair_search import Node
from .search.execution import clone
from .outputs import digest
from .page_geometry import fingerprint
from .component_closure import shape_measurement


def smaller_side(row):
    sides=[tuple(sorted(row['cut'])),tuple(sorted(row['other_side']))]
    return min(sides,key=lambda x:(len(x),x))


def cut_options(node,moving,proposals,tolerance):
    """A local ordering tier precedes wider modes; no fitted mode is removed.

    Distinct physical neighbors may corroborate the same group motion. This
    is an ordering heuristic only, never accepted support or confidence.
    """
    moving=set(moving); rows=[]; state=node.state
    for p in proposals:
        a,b=p['pieces']
        if (a in moving)==(b in moving):
            continue
        h=np.asarray(p['transform_b_to_a'],float)
        try:
            require_rigid(h)
            if not np.allclose(h[:2,:2],np.rint(h[:2,:2]),atol=1e-7,rtol=0.):
                continue
            inside,outside=(a,b) if a in moving else (b,a)
            desired=state.poses[b]@np.linalg.inv(h) if inside==a else state.poses[a]@h
            delta=desired@np.linalg.inv(state.poses[inside])
        except (ValueError,np.linalg.LinAlgError):
            continue
        shifted=max(float(np.linalg.norm((delta@state.poses[pid])[:2,2]-state.poses[pid][:2,2])) for pid in moving)
        rotation_changed=not np.allclose(delta[:2,:2],np.eye(2),atol=1e-7,rtol=0.)
        if not rotation_changed and shifted<=1e-6:
            continue
        rows.append({'proposal':p,'delta':delta,'outside':outside,'displacement_px':shifted,
            'rotation_changed':rotation_changed,'local_ordering_tier':not rotation_changed and shifted<=4.*tolerance})
    # Compare motions by their action on one common group anchor. Transform
    # translations alone depend on the current component's arbitrary gauge.
    anchor=min(moving)
    if rows:
        anchors=np.stack([(r['delta']@state.poses[anchor])[:2,2] for r in rows])
        rotations=np.stack([r['delta'][:2,:2].ravel() for r in rows])
        for i,row in enumerate(rows):
            compatible=(np.max(np.abs(rotations-rotations[i]),axis=1)<=1e-7)&(np.linalg.norm(anchors-anchors[i],axis=1)<=tolerance)
            row['distinct_neighbor_witnesses']=len({rows[j]['outside'] for j in np.flatnonzero(compatible)}-{row['outside']})
    return sorted(rows,key=lambda r:(not r['local_ordering_tier'],-r['distinct_neighbor_witnesses'],
        -quality(r['proposal']),r['displacement_px'],r['proposal']['proposal_id']))


def moved_node(context,source,node,moving,option):
    """Retain each side's internal tree and replace their single bridge.

    General group moves preserve relative poses inside both components. A
    non-tree cut is left to wider growth, rather than publishing a disconnected
    support forest with an incorrectly declared connected component.
    """
    if source.selected or any(len(g)!=1 for g in source.components()):
        raise ValueError('Cut repair requires an independent singleton source')
    moving=set(moving); stable=set(node.state.poses)-moving
    crossing=[r for r in node.contacts if (r['pieces'][0] in moving)!=(r['pieces'][1] in moving)]
    if not moving or not stable or len(crossing)!=1 or crossing[0]['role']!='merge':
        raise ValueError('Cut does not split a single independent tree bridge')
    p=option['proposal']; a,b=p['pieces']; h=np.asarray(p['transform_b_to_a'],float)
    require_rigid(h)
    if not np.allclose(h[:2,:2],np.rint(h[:2,:2]),atol=1e-7,rtol=0.):
        raise ValueError('Undeclared cut repair rotation')
    arcs=tuple(contour_arc(context.problem,r) for r in p['arcs'])
    if [arc.piece for arc in arcs]!=[a,b] or (a in moving)==(b in moving):
        raise ValueError('Repair proposal must cross its declared cut')
    metrics=check_arcs(context.problem,arcs,{a:np.eye(3),b:h},context.config)
    state=clone(node.state)
    for pid in moving:
        state.poses[pid]=option['delta']@state.poses[pid]
    state.poses=regauge(state.poses,set(state.poses))
    groups=sorted([sorted(moving),sorted(stable)],key=lambda g:g[0])
    contact={'pieces':[a,b],'arcs':arcs,'metrics':metrics,'proposal_id':p['proposal_id'],
        'proposed_transform_b_to_a':p['transform_b_to_a'],'source_groups':groups,
        'merged_groups':[next(g for g in groups if a in g),next(g for g in groups if b in g)],
        'role':'merge','spread':p['shape_spread_ratio']}
    contacts=[r for r in node.contacts if r is not crossing[0]]+[contact]
    return Node(state,contacts,node.value)


class CutJob:
    def __init__(self,source,node,moving,cut,proposal_revision):
        self.source=source; self.node=node; self.moving=tuple(moving); self.cut=cut
        self.revision=-1; self.pending=[]; self.tried=set(); self.pose_histories=set()
        self.created_revision=proposal_revision; self.counts=Counter(); self.records=[]

    def update(self,context):
        if self.revision!=context.proposal_revision:
            self.pending=[r for r in cut_options(self.node,self.moving,context.generated,context.config.cycle_tolerance_px)
                if r['proposal']['proposal_id'] not in self.tried]
            self.revision=context.proposal_revision

    def step(self,context,physical):
        self.update(context)
        if not self.pending:
            return None,False
        option=self.pending.pop(0); p=option['proposal']; self.tried.add(p['proposal_id'])
        self.counts['alternatives_examined']+=1
        record={'source_candidate_id':fingerprint(self.node.state),'moving_piece_ids':list(self.moving),
            'cut_pair':self.cut['pieces'],'proposal_id':p['proposal_id'],'pieces':p['pieces'],
            'displacement_native_px':option['displacement_px'],'rotation_changed':option['rotation_changed'],
            'distinct_neighbor_witnesses':option['distinct_neighbor_witnesses'],
            'local_ordering_tier':option['local_ordering_tier'],'proposal_bank_revision':self.revision,
            'accepted':False,'reference_used':False,'new_pose_fit_generated':False}
        try:
            node=moved_node(context,self.source,self.node,self.moving,option)
            key=digest([fingerprint(node.state),sorted(r['proposal_id'] for r in node.contacts)])
            if key in self.pose_histories:
                self.counts['duplicate_histories_avoided']+=1
                return None,True
            self.pose_histories.add(key)
            shape=shape_measurement(context,node.state,256); record['coarse_geometry']=shape
            if not shape['shape_gate_pass']:
                record['reason']='page_geometry'; return None,True
            check=physical.state(node.state,final=True)
            if not check['valid']:
                record['reason']=check['reason']; return None,True
            candidate,verdict=context.finalize(self.source,node,physical)
            verdict.update(lane='unsupported_cut_repair',cut_repair_proposal_id=p['proposal_id'])
            context.repair_audit['final_checks'].append(verdict)
            if candidate is None and 'diagnostic_state' in verdict:
                context.repair_audit['rejected_complete_candidates'].append(verdict)
            record.update(accepted=candidate is not None,verdict=verdict)
            if candidate is not None:
                self.counts['verified_completions']+=1
            return candidate,True
        except ValueError as exc:
            record['reason']=str(exc); return None,True
        finally:
            self.records.append(record)
