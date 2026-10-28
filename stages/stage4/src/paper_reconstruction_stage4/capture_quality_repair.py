"""Observed-only repair of supported complete capture assemblies.

Inputs are in-memory cold-search states and observed contours. This module has
no filesystem, previous-result or reference reader. Every changed pose must
independently pass the normal support, collision and complete-page checks.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from .geometry import map_points,relative
from .model import singleton_state
from .repair_search import Node
from .search.execution import clone
from .native_bridge import pair_options,check_arcs
from .native_partial import component_cuts
from .cut_repair_v21 import cut_options,moved_node
from .component_closure import shape_measurement
from .page import enclosing_rectangle
from .page_geometry import fingerprint
from .outputs import digest


def compact_area(points):
    """Same pixel-centre rectangle formula, vectorized for proposal triage."""
    hull=points[ConvexHull(points).vertices]
    edge=np.roll(hull,-1,axis=0)-hull
    angles=np.unique(np.mod(np.arctan2(edge[:,1],edge[:,0]),np.pi/2.))
    c,s=np.cos(angles),np.sin(angles)
    axes=np.stack([c,-s,s,c],axis=1).reshape(-1,2,2)
    xy=np.einsum('ni,aij->anj',hull,axes)
    return float(np.min(np.prod(xy.max(axis=1)-xy.min(axis=1)+1.,axis=1)))


def quality_met(shape):
    """Improvement stopping only; never a substitute for acceptance checks."""
    uncertainty=shape['raster_fraction_uncertainty_estimate']
    return bool(shape['shape_gate_pass'] and shape['unfilled_fraction']<=uncertainty and
        shape['overlap_fraction']<=uncertainty and shape['interior_hole_fraction']<=uncertainty)


def node_from_verified(context,state):
    if len(state.components())!=1:raise ValueError('Quality repair requires a complete component')
    ids=sorted(state.poses);parent={pid:pid for pid in ids}
    def root(pid):
        while parent[pid]!=pid:pid=parent[pid]
        return pid
    contacts=[]
    for eid in sorted(state.selected,key=lambda eid:(eid not in state.tree,eid)):
        edge=context.edges[eid];a,b=sorted((edge.a,edge.b))
        ra,rb=root(a),root(b)
        if ra==rb:continue
        arcs=tuple(sorted(context.support.arcs_for(state,eid),key=lambda arc:arc.piece))
        metrics=check_arcs(context.problem,arcs,state.poses,context.config)
        h=relative(state.poses,a,b)
        contacts.append({'pieces':[a,b],'arcs':arcs,'metrics':metrics,'role':'merge',
            'proposal_id':digest(['verified-in-memory-capture-contact.v1',eid,h.tolist()]),
            'proposed_transform_b_to_a':h.tolist(),'source_groups':[[pid] for pid in ids],
            'merged_groups':[[a],[b]]})
        parent[ra]=rb
    if len(contacts)!=len(ids)-1:raise ValueError('Verified contacts do not provide an independent spanning tree')
    raw=clone(state);raw.selected=();raw.tree=();raw.assignments={};raw.active_support={}
    return Node(raw,contacts)


def refitted_node(context,node,poses):
    state=clone(node.state);state.poses=poses;contacts=[]
    for original in node.contacts:
        a,b=original['pieces'];options=pair_options(context.problem,a,b,poses,context.config,[])
        if not options:raise ValueError('Refined pose lost its observed seam')
        arcs,metrics=options[0];h=relative(poses,a,b)
        contacts.append({**original,'arcs':arcs,'metrics':metrics,
            'proposal_id':digest(['observed-common-scale-rigid-refinement.v1',a,b,h.tolist()]),
            'proposed_transform_b_to_a':h.tolist()})
    return Node(state,contacts)


def verify(context,node,physical,deadline,audit):
    if time.perf_counter()>=deadline:return None,None
    shape=shape_measurement(context,node.state,768)
    if not shape['shape_gate_pass'] or not physical.state(node.state,final=True)['valid']:return None,None
    basis=singleton_state(context.problem);context.repair_deadline=deadline
    while time.perf_counter()<deadline:
        candidate,verdict=context.finalize(basis,node,physical)
        audit['support_replays']+=1
        if candidate is not None or not verdict.get('support_selection',{}).get('continuation_pending'):break
    else:return None,None
    context.repair_audit['final_checks'].append({**verdict,'lane':'supported_capture_quality_repair'})
    return candidate,verdict


def best_cut(context,node,rows,physical,deadline,audit,moving_only=None):
    """Rank all modes by observed compactness, then replay finite fine batches.

    Batches control memory/turn size only. A failed batch does not remove later
    modes. Geometry success, exhaustion of this local neighborhood, or the shared
    deadline ends work; no candidate-count limit declares an assembly impossible.
    """
    current=shape_measurement(context,node.state,768);ids=set(node.state.poses)
    clouds={pid:map_points(context.hulls[pid],h) for pid,h in node.state.poses.items()}
    pending=[];seen=set();maximum=(1.+context.config.page_component_area_slack)*sum(context.areas.values())
    for moving in ([moving_only] if moving_only is not None else component_cuts(node,ids)):
        moving=set(moving);stable=ids-moving
        if not moving or not stable:continue
        left=np.vstack([clouds[p] for p in sorted(moving)]);right=np.vstack([clouds[p] for p in sorted(stable)])
        left=left[ConvexHull(left).vertices];right=right[ConvexHull(right).vertices]
        for option in cut_options(node,moving,rows,context.config.cycle_tolerance_px,'continuous_rigid',corroboration=False):
            if time.perf_counter()>=deadline:
                audit['deadline_reached']=True;return None
            audit['cut_options_ranked']+=1
            key=(tuple(sorted(moving)),option['delta'].tobytes())
            if key in seen:continue
            seen.add(key)
            area=compact_area(np.vstack([map_points(left,option['delta']),right]))
            if area<=maximum+1e-6:pending.append((area,tuple(sorted(moving)),option['proposal']['proposal_id'],option))
    pending.sort(key=lambda row:(row[0],row[1],row[2]));batch=[]
    fine_batch=max(1,context.config.repair_beam//2)
    for _,moving,_,option in pending:
        if time.perf_counter()>=deadline:audit['deadline_reached']=True;break
        try:
            changed=moved_node(context,singleton_state(context.problem),node,moving,option)
            if not physical.state(changed.state,final=False)['valid']:continue
            shape=shape_measurement(context,changed.state,768);audit['fine_geometry_checks']+=1
            if not shape['shape_gate_pass'] or shape['score']<=current['score']+1e-5:continue
            batch.append((shape['score'],changed,moving,option['proposal']['proposal_id']))
        except ValueError:continue
        if len(batch)<fine_batch:continue
        result=_verify_batch(context,batch,physical,deadline,audit)
        if result is not None:return result
        batch=[]
    return _verify_batch(context,batch,physical,deadline,audit)


def _verify_batch(context,batch,physical,deadline,audit):
    for _,node,moving,proposal in sorted(batch,key=lambda row:(-row[0],row[3])):
        candidate,verdict=verify(context,node,physical,deadline,audit)
        if candidate is not None:
            audit['accepted_moves'].append({'moving_piece_ids':list(moving),'proposal_id':proposal,
                'geometry':verdict['geometry'],'reference_used':False,'independently_verified':True})
            return node,candidate,verdict
    return None


def polish(context,state,deadline):
    from .feasibility import Feasibility
    from .observed_border_angles import lines,angle_seeds
    from .seam_poses_rigid import generate
    from .observed_rigid_refinement import proposals
    started=time.perf_counter();physical=Feasibility(context.problem,context.config)
    initial=shape_measurement(context,state,768)
    audit={'policy':'observed-border-angles-and-supported-incumbent-repair.v1','reference_geometry_used':False,
        'previous_result_loaded':False,'source_candidate_id':fingerprint(state),'initial_geometry':initial,
        'relative_piece_scale':1.,'acceptance_thresholds_changed':False,'support_replays':0,
        'cut_options_ranked':0,'fine_geometry_checks':0,'accepted_moves':[],'rigid_refinement':[],
        'border_generation':[],'deadline_reached':False,'local_search_is_exhaustive':False,
        'local_neighborhood_exhausted':False}
    context.repair_audit.setdefault('supported_capture_quality_repair',[]).append(audit)
    context.progress('Capture quality repair: deriving rotations from observed borders')
    best=state;best_shape=initial
    try:
        if quality_met(initial):return best
        node=node_from_verified(context,state)
        if not hasattr(context,'capture_border_features'):
            context.capture_border_features={pid:lines(piece.contour['points_xy'],context.config.contact_tolerance_px)
                for pid,piece in context.problem.pieces.items()}
        if not hasattr(context,'capture_border_attempted'):context.capture_border_attempted=set()
        clouds={pid:map_points(context.hulls[pid],h) for pid,h in node.state.poses.items()}
        full=enclosing_rectangle(np.vstack(list(clouds.values())))[0]
        cuts=[moving for moving in component_cuts(node,set(clouds)) if len(moving)==1]
        cuts.sort(key=lambda moving:-(full-enclosing_rectangle(np.vstack([v for pid,v in clouds.items() if pid not in moving]))[0])/sum(context.areas[pid] for pid in moving))
        for moving in cuts:
            if time.perf_counter()>=deadline:break
            a,b=next(r['pieces'] for r in node.contacts if (r['pieces'][0] in moving)!=(r['pieces'][1] in moving))
            pair=(a,b)
            if pair in context.capture_border_attempted:continue
            angles=angle_seeds(context.capture_border_features[a],context.capture_border_features[b])
            context.capture_border_attempted.add(pair)
            if not angles:continue
            rows,fit=generate(context.problem,context.config,deadline,samples=384,peaks=12,
                angle_seeds={pair:angles},angle_source='observed-long-straight-contour-spans.v1')
            fit.update(reference_geometry_used=False,upstream_angle_claimed=False,
                target_moving_piece_ids=list(moving),observed_border_hypotheses_are_not_support=True)
            audit['border_generation'].append(fit)
            context.progress(f'Observed border rotation fitting: {len(angles)} angles, {len(rows)} verified native pose proposals')
            context._merge_generation(rows,fit,'observed_border_generation')
            changed=best_cut(context,node,rows,physical,deadline,audit,moving_only=moving)
            if changed is not None:
                node,best,verdict=changed;best_shape=verdict['geometry']
                break
        if time.perf_counter()<deadline and not quality_met(best_shape):
            context.progress('Capture quality repair: fitting shared-scale rigid poses')
            for poses,fit in proposals(context,node,deadline):
                try:
                    trial=refitted_node(context,node,poses)
                    shape=shape_measurement(context,trial.state,768)
                    if shape['score']<=best_shape['score']+1e-5:
                        audit['rigid_refinement'].append({**fit,'accepted':False});continue
                    candidate,verdict=verify(context,trial,physical,deadline,audit)
                    audit['rigid_refinement'].append({**fit,'accepted':candidate is not None})
                    if candidate is not None:node,best,best_shape=trial,candidate,verdict['geometry']
                except ValueError as exc:audit['rigid_refinement'].append({**fit,'accepted':False,'reason':str(exc)})
        histories={fingerprint(node.state)}
        context.progress('Capture quality repair: revisiting supported tree cuts using the measured pose bank')
        while time.perf_counter()<deadline and not quality_met(best_shape):
            changed=best_cut(context,node,context.generated,physical,deadline,audit)
            if changed is None:
                audit['local_neighborhood_exhausted']=time.perf_counter()<deadline
                break
            trial,candidate,verdict=changed;key=fingerprint(trial.state)
            if key in histories:break
            histories.add(key);node,best,best_shape=trial,candidate,verdict['geometry']
            context.progress(f"Verified capture geometry improvement: {best_shape['score']:.5f}")
        return best
    except ValueError as exc:
        audit['unavailable_reason']=str(exc);return best
    finally:
        audit.update(elapsed_seconds=time.perf_counter()-started,final_geometry=best_shape,
            final_candidate_id=fingerprint(best),geometry_quality_target_met=quality_met(best_shape),
            deadline_reached=audit['deadline_reached'] or time.perf_counter()>=deadline,
            verified_incumbent_preserved=True)
