"""Versioned consumer-derived native seam evidence, separate from Stage 3 fits.

Intervals are reconstructed from the authenticated native contour on replay.
No upstream occurrence, prediction, or bundle is rewritten.
"""
import numpy as np
from scipy.spatial import cKDTree
from paper_reconstruction_contracts import map_points
from .config import CLOSURE_POLICY
from .constraints import require_rigid
from .geometry import relative
from .model import Edge
from .outputs import digest
from .support import Arc, contact_error, overlaps
from .active_support import slice_arc

POLICY = 'stage4.native-component-bridge.v1'
MINIMUM_LENGTH = 48.
MINIMUM_PAIRS = 2
NORMAL_COSINE = -.8


def contour_arc(problem, value):
    if set(value) != {'piece', 'intervals', 'length'} or value['piece'] not in problem.pieces:
        raise ValueError('Invalid derived native arc inventory')
    pid = value['piece']; xy = np.asarray(problem.pieces[pid].contour['points_xy'],float)
    perimeter = float(np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1).sum())
    intervals = tuple(tuple(v) for v in value['intervals'])
    if not 1 <= len(intervals) <= 2 or any(len(v)!=2 or not np.isfinite(v).all() or not 0 <= v[0] < v[1] <= perimeter+1e-7 for v in intervals):
        raise ValueError('Derived native arc escapes contour')
    # Two intervals may only represent one contiguous circular seam.
    if len(intervals)==2 and (abs(intervals[0][1]-perimeter)>1e-7 or abs(intervals[1][0])>1e-7 or intervals[1][1]>=intervals[0][0]):
        raise ValueError('Derived native support is not contiguous')
    length = sum(b-a for a,b in intervals)
    if not np.isfinite(value['length']) or abs(length-value['length'])>1e-6 or length<MINIMUM_LENGTH:
        raise ValueError('Insufficient derived native seam length')
    return slice_arc(problem,Arc(pid,intervals,np.empty((0,2)),length))


def arc_value(arc):
    return {'piece':arc.piece,'intervals':[list(v) for v in arc.intervals],'length':arc.length}


def arc_from_indices(problem, pid, first, last):
    xy = np.asarray(problem.pieces[pid].contour['points_xy'],float); n=len(xy)
    cum=np.r_[0.,np.cumsum(np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1))]
    if not 0 <= first < n or not first < last < first+n:
        raise ValueError('Invalid circular native run')
    intervals=([(float(cum[first]),float(cum[last]))] if last<n else
               [(float(cum[first]),float(cum[-1])),(0.,float(cum[last-n]))])
    intervals=[v for v in intervals if v[1]>v[0]]
    return contour_arc(problem,{'piece':pid,'intervals':intervals,'length':sum(b-a for a,b in intervals)})


def runs(good):
    """Contiguous circular sample runs, including a run through contour zero."""
    n=len(good)
    if good.all() or not good.any():return []
    start=(int(np.flatnonzero(~good)[0])+1)%n
    ordered=np.roll(good,-start)
    changes=np.diff(np.r_[False,ordered,False].astype(int))
    return [((start+a)%n, (start+a)%n+b-a-1) for a,b in
            zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1),strict=True) if b-a>=4]


def available(problem, pid, occupied):
    xy=np.asarray(problem.pieces[pid].contour['points_xy'],float)
    s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    good=np.ones(len(xy),bool)
    for arc in occupied:
        if arc.piece==pid:
            for lo,hi in arc.intervals:good &= ~((s>=lo-2.) & (s<=hi+2.))
    return good


def normal_metrics(problem, arcs, poses):
    values=[]
    for a,b in (arcs,arcs[::-1]):
        h=relative(poses,b.piece,a.piece); mapped=map_points(a.points,h)
        nearest=cKDTree(b.points).query(mapped)[1]
        ia=cKDTree(problem.pieces[a.piece].contour['points_xy']).query(a.points)[1]
        ib=cKDTree(problem.pieces[b.piece].contour['points_xy']).query(b.points[nearest])[1]
        na=problem.pieces[a.piece].contour['inward_normal_xy'][ia]@h[:2,:2].T
        nb=problem.pieces[b.piece].contour['inward_normal_xy'][ib]
        cosine=np.sum(na*nb,axis=1)/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1),1e-12)
        values.extend(cosine.tolist())
    return {'normal_cosine_median':float(np.median(values)),
            'opposed_normal_fraction':float(np.mean(np.asarray(values)<NORMAL_COSINE))}


def check_arcs(problem, arcs, poses, config, occupied=()):
    error=contact_error(arcs,poses); normals=normal_metrics(problem,arcs,poses)
    if error>config.contact_tolerance_px or normals['opposed_normal_fraction']<.85:
        raise ValueError('Derived bridge contact distance/interior normals failed')
    if any(overlaps(a,b,config) for a in arcs for b in occupied):
        raise ValueError('Derived bridge reuses occupied native support')
    if min(a.length for a in arcs)/max(a.length for a in arcs)<.75:
        raise ValueError('Derived seam lengths do not correspond')
    return {'bidirectional_contact_p95_native_px':error,**normals}


def pair_options(problem, a, b, poses, config, occupied):
    """Extract contiguous, monotone, opposing contacts from full native contours."""
    pa,pb=(problem.pieces[p].contour for p in (a,b))
    xa,xb=(map_points(problem.pieces[p].contour['points_xy'],poses[p]) for p in (a,b))
    distance,nearest=cKDTree(xb).query(xa)
    na=pa['inward_normal_xy']@poses[a][:2,:2].T
    nb=pb['inward_normal_xy']@poses[b][:2,:2].T
    cosine=np.sum(na*nb[nearest],axis=1)/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nb[nearest],axis=1),1e-12)
    good=(distance<=config.contact_tolerance_px)&(cosine<NORMAL_COSINE)
    good &= available(problem,a,occupied)&available(problem,b,occupied)[nearest]
    options=[]; n=len(xa); m=len(xb)
    for first,last in runs(good):
        indices=np.arange(first,last+1)%n
        # Unwrap the nearest target positions before checking monotonicity.
        target=np.unwrap(nearest[indices]*2*np.pi/m)*m/(2*np.pi)
        steps=np.diff(target); monotone=float(max(np.mean(steps<=.01),np.mean(steps>=-.01)))
        if monotone<.9:continue
        lo,hi=int(round(target.min())),int(round(target.max()))
        if hi<=lo or hi-lo>=m:continue
        try:
            arcs=(arc_from_indices(problem,a,first,last),arc_from_indices(problem,b,lo%m,lo%m+hi-lo))
            metrics=check_arcs(problem,arcs,poses,config,occupied)
        except ValueError:continue
        options.append((arcs,{**metrics,'monotone_fraction':monotone}))
    options.sort(key=lambda item:(-min(a.length for a in item[0]),digest([arc_value(a) for a in item[0]])))
    return options[:4]


def make_record(problem, arcs, poses, source_id, proposal_id, source_groups):
    a,b=sorted(arc.piece for arc in arcs)
    arcs=tuple(sorted(arcs,key=lambda arc:arc.piece))
    value={'policy':POLICY,'input_manifest_sha256':problem.manifest['manifest_sha256'],
        'pieces':[a,b],'arcs':[arc_value(arc) for arc in arcs],
        'transform_b_to_a':relative(poses,a,b).tolist(),
        'source_candidate_id':source_id,'placement_proposal_id':proposal_id,
        'source_groups':source_groups,'producer':'stage4_observed_native_contours',
        'upstream_pose_or_occurrence_claimed':False}
    return {**value,'bridge_id':digest(value)}


def validate_record(value):
    if value.get('policy')=='stage4.native-component-bridge.v3':
        from .native_bridge_v3 import validate_record as validate_v3
        return validate_v3(value)
    if value.get('policy')=='stage4.native-component-bridge.v2':
        from .native_bridge_v2 import validate_record as validate_v2
        return validate_v2(value)
    keys={'policy','input_manifest_sha256','pieces','arcs','transform_b_to_a','source_candidate_id',
          'placement_proposal_id','source_groups','producer','upstream_pose_or_occurrence_claimed','bridge_id'}
    if set(value)!=keys or value['policy']!=POLICY or value['producer']!='stage4_observed_native_contours' or value['upstream_pose_or_occurrence_claimed'] is not False:
        raise ValueError('Invalid consumer-derived bridge policy/provenance')
    if value['bridge_id']!=digest({k:v for k,v in value.items() if k!='bridge_id'}):
        raise ValueError('Changed consumer-derived bridge identity')
    a,b=value['pieces'];groups=value['source_groups']
    if not a<b or len(value['arcs'])!=2 or [arc['piece'] for arc in value['arcs']]!=[a,b]:
        raise ValueError('Invalid consumer-derived bridge pair')
    flat=[p for group in groups for p in group]
    if len(groups)!=2 or any(not g or g!=sorted(set(g)) for g in groups) or len(flat)!=len(set(flat)) or not any(a in g for g in groups) or not any(b in g for g in groups) or any(a in g and b in g for g in groups):
        raise ValueError('Derived bridge does not cross two observed components')
    require_rigid(value['transform_b_to_a'])


def record_arcs(problem, value):
    validate_record(value)
    if value['input_manifest_sha256']!=problem.manifest['manifest_sha256'] or sorted(p for g in value['source_groups'] for p in g)!=sorted(problem.pieces):
        raise ValueError('Derived bridge input binding differs')
    return tuple(contour_arc(problem,a) for a in value['arcs'])


def edge_from_record(problem, value, config):
    if config.component_closure_policy!=CLOSURE_POLICY or not problem.constraint_profile:
        raise ValueError('Consumer-derived bridge requires the explicit bound closure profile')
    arcs=record_arcs(problem,value);a,b=value['pieces'];oid=digest({'policy':value['policy'],'bridge_id':value['bridge_id']})
    if value['policy']=='stage4.native-component-bridge.v3':
        from .config import REPAIR_V18_POLICY
        if config.native_repair_policy!=REPAIR_V18_POLICY:
            raise ValueError('Page-boundary bridge requires the V18 development policy')
    if value['policy'] in {'stage4.native-component-bridge.v2','stage4.native-component-bridge.v3'} and config.native_repair_policy=='disabled.v1':
        raise ValueError('Multi-component native bridge requires the native repair policy')
    conditioned=False
    if value['policy'] in {'stage4.native-component-bridge.v2','stage4.native-component-bridge.v3'}:
        from .seam_poses import spread
        conditioned=min(v.length for v in arcs)>=96. and spread(arcs)>=.02
    pose={'span_a':None,'span_b':None,'alignment':{'underconstrained':not conditioned}}
    occurrence={'occurrence_id':oid,'query_key':'native-contour:'+a,'target_key':'native-contour:'+b,
                'pose':pose,'native_bridge':value}
    # Neutral edge gain; no learned Stage 3 confidence is fabricated.
    edge=Edge(value['bridge_id'],a,b,np.asarray(value['transform_b_to_a'],float),config.join_cost,[occurrence],
              {'score_method':value['policy'],'score_variant':'observed_native_seam' if value['policy'].endswith(('.v2','.v3')) else 'observed_component_extent','pose':pose,'native_bridge':value})
    return edge,arcs


def install_result_bridges(problem, result, config):
    """Restore only explicitly declared evidence; independently verify on use."""
    values={}
    for value in result.get('search',{}).get('native_bridge_evidence',[]):
        prior=values.setdefault(value['bridge_id'],value)
        if prior!=value:raise ValueError('Conflicting derived evidence identity')
    for layout in result['layouts']:
        for join in layout['selected_joins']:
            if 'native_bridge' in join:
                value=join['native_bridge']; prior=values.setdefault(value['bridge_id'],value)
                if prior!=value:raise ValueError('Conflicting derived evidence identity')
    existing={e.id:e for e in problem.edges}
    for value in values.values():
        edge,_=edge_from_record(problem,value,config)
        if edge.id in existing:
            if existing[edge.id].raw.get('native_bridge')!=value:raise ValueError('Derived bridge replaces upstream evidence')
        else:problem.edges.append(edge);existing[edge.id]=edge
