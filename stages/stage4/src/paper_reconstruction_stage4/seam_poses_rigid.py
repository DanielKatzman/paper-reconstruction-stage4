"""Fixed-scale native contour fitting conditioned on observed rotation modes.

Only rotation angles are taken from immutable input evidence. Translation and
contact support are recomputed from full observed contours. No old assembly,
reference coordinates, scale fitting or perspective transform enters here.
"""
from collections import defaultdict
import numpy as np
from .constraints import require_rigid
from .outputs import digest
from .native_bridge import pair_options,arc_value
from .seam_poses import translation_modes,refine_translation,spread

POLICY='native-contour-observed-rigid-angles.v1'
MEASUREMENT='native-contour-rigid-angle-votes.v1'


def angle_modes(problem):
    rows=defaultdict(list)
    for edge in sorted(problem.edges,key=lambda e:(-e.weight,e.id)):
        if 'native_bridge' in edge.raw:continue
        h=np.asarray(edge.transform,float);require_rigid(h)
        a,b=edge.a,edge.b
        if a>b:a,b=b,a;h=np.linalg.inv(h)
        rows[a,b].append((float(np.arctan2(h[1,0],h[0,0])),edge.id))
    result={};step=np.deg2rad(.25)
    from itertools import combinations
    for pair in combinations(sorted(problem.pieces),2):
        choices=[]
        for angle,source in [*rows[pair],*((i*np.pi/2.,None) for i in range(4))]:
            if any(abs(np.arctan2(np.sin(angle-old['angle']),np.cos(angle-old['angle'])))<step for old in choices):continue
            choices.append({'angle':angle,'source_geometry_id':source})
        result[pair]=choices
    return result


def prepare(data,a,b,peaks,cache,angles):
    _,_,sa,na=data[a];xb,nb,sb,snb=data[b];rotations=[]
    for index,seed in enumerate(angles[a,b]):
        angle=seed['angle'];key=('rigid',b,angle)
        if key not in cache:
            c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s],[s,c]])
            cache[key]=(r,xb@r.T,nb@r.T,sb@r.T,snb@r.T)
        r,rb,rn,rsb,rsnb=cache[key]
        general=translation_modes(sa,rsb,na,rsnb,peaks)
        curved=[]
        for t,v in translation_modes(sa,rsb,na,rsnb,2,curved_only=True):
            if not any(np.linalg.norm(t-old[0])<6. for old in [*general,*curved]):curved.append((t,v))
        rotations.append({'quarter':index,'angle':angle,'source_geometry_id':seed['source_geometry_id'],
            'observed_border_seed':seed.get('observed_border_seed'),
            'r':r,'rb':rb,'rn':rn,'general':general,'curved':curved})
    return rotations


def fit(problem,config,physical,data,a,b,value):
    rotation,_,_,translation,votes=value;xa,na,_,_=data[a]
    h=np.eye(3);h[:2,:2]=rotation['r'];h[:2,2]=refine_translation(xa,rotation['rb'],na,rotation['rn'],translation)
    require_rigid(h);rows=[]
    for arcs,metrics in pair_options(problem,a,b,{a:np.eye(3),b:h},config,[]):
        length=min(v.length for v in arcs);condition=spread(arcs)
        if length<64. or not physical.pair(a,b,h,final=False)['valid']:continue
        row={'policy':MEASUREMENT,'pieces':[a,b],'rotation_angle_radians':rotation['angle'],
            'angle_seed_geometry_id':rotation['source_geometry_id'],'translation_refitted_from_native_contours':True,
            'transform_b_to_a':h.tolist(),'arcs':[arc_value(v) for v in arcs],
            'native_length_px':length,'shape_spread_ratio':condition,'conditioned':condition>=.05,
            'native_metrics':metrics,'score':float(length*(.1+min(condition,.35))/(1.+metrics['bidirectional_contact_p95_native_px']/4.)),
            'vote_strength':votes}
        if rotation.get('observed_border_seed') is not None:
            row['rotation_seed_source']='observed-long-straight-contour-spans.v1'
            row['observed_border_seed']=rotation['observed_border_seed']
        row['proposal_id']=digest(row);rows.append(row)
    return rows


def retain(rows):
    kept=[]
    for row in sorted(rows,key=lambda r:(not r['conditioned'],-r['score'],r['proposal_id'])):
        h=np.asarray(row['transform_b_to_a'])
        if any(np.max(np.abs(h[:2,:2]-np.asarray(old['transform_b_to_a'])[:2,:2]))<1e-6
               and np.linalg.norm(h[:2,2]-np.asarray(old['transform_b_to_a'])[:2,2])<4. for old in kept):continue
        kept.append(row)
    return kept


def generate(problem,config,deadline,maximum_checks=None,samples=256,peaks=4,frontiers=(),angle_seeds=None,
             angle_source='targeted-authenticated-archive-angles'):
    from .seam_poses_v18 import generate as scheduled
    angles=angle_modes(problem) if angle_seeds is None else angle_seeds
    proposals,audit=scheduled(problem,config,deadline,maximum_checks,samples,peaks,frontiers,
        _rotation_provider=lambda data,a,b,p,cache:prepare(data,a,b,p,cache,angles),
        _fit_provider=fit,_retain_provider=retain,_pairs=None if angle_seeds is None else angles)
    audit.update(policy=POLICY,measurement_policy=MEASUREMENT,
        rotation_mode=('observed-working-evidence-angles-plus-quarter-basins' if angle_seeds is None else
            angle_source),relative_piece_scale=1.,
        rotation_seed_bin_degrees=.25,pose_translation_inherited=False,reference_coordinates_used=False,
        angle_coverage_is_exhaustive=False,angle_seeds=[{'pieces':list(pair),'seeds':seeds} for pair,seeds in angles.items()])
    return proposals,audit
