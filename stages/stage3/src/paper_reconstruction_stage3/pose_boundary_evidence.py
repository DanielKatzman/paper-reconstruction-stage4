"""Normal-compatible point-to-polyline support and conservative CPU refinement."""
import numpy as np

from .alignment import apply_transform,fit_transforms
from .pose_geometry import transform_distance


from .exact_acceleration import correspondences


def transformed(feature,matrix):
    if not np.array_equal(np.asarray(matrix)[2],[0.,0.,1.]):
        from .projective import warp_normals
        native=np.asarray(feature['points_xy_crop']);points=apply_transform(native,matrix)
        scale=float(np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))/max(np.sum(np.linalg.norm(np.diff(native,axis=0),axis=1)),1e-12))
        return points,warp_normals(native,feature['inward_normal_xy_crop'],matrix),scale
    linear=np.asarray(matrix)[:2,:2];scale=float(np.sqrt(np.linalg.det(linear)))
    return apply_transform(feature['points_xy_crop'],matrix),feature['inward_normal_xy_crop']@(linear/scale).T,scale


def fit_range(pose,side,total):
    span=pose['span_a' if side==0 else 'span_b'];fraction=pose['alignment']['query_arc_fraction' if side==0 else 'target_arc_fraction']
    start,end=(0.,total) if span is None else (span['start_px'],min(total,span['end_px']))
    return start+fraction[0]*(end-start),start+fraction[1]*(end-start)


def evidence(first,second,matrix,pose,policy,*,parity=None):
    a=first['points_xy_crop'].astype(float);an=first['inward_normal_xy_crop'].astype(float)
    b,bn,scale=transformed(second,matrix);da,ja,ua,_=correspondences(a,an,b,bn)
    db,jb,ub,_=correspondences(b,bn,a,an)
    use_a=np.ones(len(a),bool);use_b=np.ones(len(b),bool)
    if parity is not None:use_a=np.arange(len(a))%2==parity;use_b=np.arange(len(b))%2==parity
    ina=(da<=policy['support_distance_px'])&use_a;inb=(db<=policy['support_distance_px'])&use_b
    la=float(first['observed_length_px'][0]);lb=float(second['observed_length_px'][0])
    support=min(la*ina.sum()/max(use_a.sum(),1),lb*scale*inb.sum()/max(use_b.sum(),1))
    coverage=min(ina.sum()/max(use_a.sum(),1),inb.sum()/max(use_b.sum(),1))
    errors=np.concatenate((da[ina],db[inb]));rms=float(np.sqrt(np.mean(errors**2))) if len(errors) else None
    sa=np.linspace(0,la,len(a));sb=np.linspace(0,lb,len(b));ra,rb=fit_range(pose,0,la),fit_range(pose,1,lb)
    outside_a=(sa<ra[0])|(sa>ra[1]);outside_b=(sb<rb[0])|(sb>rb[1])
    match_b=(ja+ua)/(len(b)-1)*lb;match_a=(jb+ub)/(len(a)-1)*la
    extra_a=ina&outside_a&((match_b<rb[0])|(match_b>rb[1]))
    extra_b=inb&outside_b&((match_a<ra[0])|(match_a>ra[1]))
    extra=min(la*extra_a.sum()/max(use_a.sum(),1),lb*scale*extra_b.sum()/max(use_b.sum(),1))
    precision=float(np.exp(-rms/4.)) if rms is not None else 0.
    score=.4*support/(support+128.)+.25*coverage+.25*precision+.1*extra/(extra+64.)
    credible=support>=policy['minimum_support_length_px'] and min(ina.sum(),inb.sum())>=4 and score>=policy['minimum_observable_score']
    return {'score':float(score),'supported_length_query_px':float(support),'symmetric_coverage':float(coverage),
        'matched_rms_query_px':rms,'support_beyond_proposal_query_px':float(extra),'matched_samples':[int(ina.sum()),int(inb.sum())],
        'credible':bool(credible),'scope':'Observable full-candidate curves; normal-compatible segment matches. No saved anchors.'}


def refine(first,second,pose,policy,base_config,probes):
    initial=np.asarray(pose['alignment']['transform'],float);current=initial.copy()
    best=evidence(first,second,current,pose,policy,parity=1);before=best;accepted=0;last_pairs=0
    a=first['points_xy_crop'].astype(float);an=first['inward_normal_xy_crop'].astype(float)
    native_b=second['points_xy_crop'].astype(float)
    for _ in range(policy['refinement_iterations']):
        b,bn,_=transformed(second,current)
        da,ja,ua,_=correspondences(a,an,b,bn);db,jb,ub,pa=correspondences(b,bn,a,an)
        ia=(np.arange(len(a))%2==0)&(da<=policy['refinement_distance_px'])
        ib=(np.arange(len(b))%2==0)&(db<=policy['refinement_distance_px'])
        qb=native_b[ja]*(1-ua[:,None])+native_b[ja+1]*ua[:,None]
        x=np.concatenate((a[ia],pa[ib]));y=np.concatenate((qb[ia],native_b[ib]));errors=np.concatenate((da[ia],db[ib]))
        if len(x)<8:break
        keep=errors<=max(float(np.quantile(errors,.8)),1e-6);x,y=x[keep],y[keep]
        if len(x)<8:break
        candidate=fit_transforms(x[None],y[None],similarity=base_config['family']=='similarity',
            scale_bounds=(base_config['minimum_scale'],base_config['maximum_scale']))[0][0]
        if transform_distance(candidate,initial,probes)>policy['maximum_refinement_displacement_px']:break
        value=evidence(first,second,candidate,pose,policy,parity=1)
        if value['score']<=best['score']+1e-6 or value['supported_length_query_px']<.9*before['supported_length_query_px']:break
        current=candidate;best=value;accepted+=1;last_pairs=len(x)
    return current,{'accepted_iterations':accepted,'fit_correspondences':last_pairs,'before_heldout':before,'after_heldout':best,
        'maximum_probe_displacement_px':transform_distance(current,initial,probes),
        'scope':'Even observable samples fit; odd samples validate. Nearby curve samples are correlated, not an independent test set.'}
