"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
from collections import defaultdict
import numpy as np
from .alignment import apply_transform
from .pose_geometry import transform_distance

def curve(inputs,owner,count):
    # Keep the established feature contract (128/256/512); verification uses
    # deterministic lower-resolution copies of the observable global512 curve.
    source=inputs.features(owner,{'samples':512});grid=np.linspace(0,len(source['points_xy_crop'])-1,count)
    native=np.arange(len(source['points_xy_crop']))
    xy=np.column_stack([np.interp(grid,native,source['points_xy_crop'][:,i]) for i in range(2)])
    normals=np.column_stack([np.interp(grid,native,source['inward_normal_xy_crop'][:,i]) for i in range(2)])
    lengths=np.linalg.norm(normals,axis=1);bad=lengths<1e-12
    if bad.any():normals[bad]=source['inward_normal_xy_crop'][np.rint(grid[bad]).astype(int)]
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    return {'points_xy_crop':xy,'inward_normal_xy_crop':normals,'observed_length_px':source['observed_length_px']}


def diversity_key(pose):
    fit=pose['alignment'];a,b=fit['query_arc_fraction'],fit['target_arc_fraction']
    fraction=min(a[1]-a[0],b[1]-b[0]);length=0 if fraction<=.125 else 1 if fraction<=.25 else 2 if fraction<=.5 else 3
    return length,pose['windows'][0] is not None,min(3,int(2*(a[0]+a[1]))),min(3,int(2*(b[0]+b[1])))


def diverse(values,limit,score,probes=None,trace=None):
    """Round-robin overlap scales, locations and source type before filling slots."""
    buckets=defaultdict(list)
    for value in values:buckets[diversity_key(value)].append(value)
    for values_ in buckets.values():values_.sort(key=lambda v:(-score(v),str(v['proposal_id'])))
    selected=[];selected_xy=[];visited=0
    while buckets and len(selected)<limit:
        progressed=False
        for length in range(4):
            tier_selected=False
            keys=sorted([k for k in buckets if k[0]==length],key=lambda k:(-score(buckets[k][0]),k))
            for key in keys:
                value=buckets[key].pop(0)
                visited+=1
                if not buckets[key]:del buckets[key]
                if probes is not None:
                    xy=apply_transform(probes,value['alignment']['transform'])
                    distances=np.max(np.linalg.norm(np.asarray(selected_xy)-xy,axis=2),axis=1) if selected_xy else np.array([])
                    duplicates=np.flatnonzero(distances<2.)
                    if len(duplicates):
                        if trace is not None:
                            other=int(duplicates[0]);trace({'value':value,'reason':'transform_duplicate','visited_order':visited,
                                'duplicate_of':selected[other],'duplicate_distance_px':float(distances[other])})
                        continue
                    selected_xy.append(xy)
                selected.append(value);progressed=True;tier_selected=True;break
            if tier_selected and trace is not None:
                # Emit only for the selection made in this overlap-length tier.
                trace({'value':value,'reason':'retained','visited_order':visited,'solver_slot':len(selected)})
            if len(selected)>=limit:break
        if not progressed and not buckets:break
    if trace is not None:
        for values_ in buckets.values():
            for value in values_:trace({'value':value,'reason':'quota_not_visited','visited_order':None})
    return selected
