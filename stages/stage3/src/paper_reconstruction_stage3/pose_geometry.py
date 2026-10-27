"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
import numpy as np
from .alignment import _subarcs, apply_transform, collision_fraction, fit_transforms
from .contracts import require
from .windows import observable_windows

def span_for(inputs,owner,window):
    if window is None:return None
    if owner not in inputs.pose_windows:
        _,p=inputs.gallery[owner];contour=inputs.contours[p['entry']['observation_key']]
        inputs.pose_windows[owner]={v['window_id']:v for v in observable_windows(contour,p['candidate'],length_px=256.)}
    require(window in inputs.pose_windows[owner],'Unknown original observable window')
    return inputs.pose_windows[owner][window]


def transform_distance(a,b,points):
    return float(np.max(np.linalg.norm(apply_transform(points,a)-apply_transform(points,b),axis=1)))


def hypotheses(first,second,fit_config,config,prepared=None,owners=None):
    ar,a,an,al=prepared[0] if prepared is not None else _subarcs(first,fit_config)
    br,b,bn,bl=prepared[1] if prepared is not None else _subarcs(second,fit_config,True)
    if not len(a) or not len(b):return []
    ia,ib=np.meshgrid(np.arange(len(a)),np.arange(len(b)),indexing='ij');ia,ib=ia.ravel(),ib.ravel()
    matrices,fitted,scales,rotations=fit_transforms(a[ia],b[ib],similarity=config['family']=='similarity',
        scale_bounds=(fit_config.minimum_scale,fit_config.maximum_scale))
    error=np.linalg.norm(fitted-a[ia],axis=2);rms=np.sqrt(np.mean(error**2,axis=1))
    length=np.maximum(np.minimum(al[ia],bl[ib]*scales),1e-6)
    transformed=np.einsum('hni,hij->hnj',bn[ib],rotations)
    normals=np.clip(np.mean((1+np.sum(an[ia]*transformed,axis=2))/2,axis=1),0.,1.)
    length_penalty=np.abs(np.log(np.maximum(al[ia],1e-6)/np.maximum(bl[ib]*scales,1e-6)))
    objective=rms/length/.03+2*normals+.25*length_penalty
    from .unit_scale import initial_trace
    initial_trace(owners,matrices,rms,length,normals,config)
    order=np.lexsort((ib,ia,-length,objective));selected=[]
    probe=second['points_xy_crop'][np.linspace(0,len(second['points_xy_crop'])-1,5).round().astype(int)]
    for i in order[:24]:
        if rms[i]/length[i]>config['maximum_normalized_residual'] or normals[i]>config['maximum_normal_penalty']:continue
        matrix=matrices[i]
        if any(transform_distance(matrix,s['transform'],probe)<2. for s in selected):continue
        eig=np.linalg.eigvalsh(np.cov(a[ia[i]].T));anisotropy=float(eig[0]/max(eig[-1],1e-12))
        selected.append({'status':'fitted','transform':matrix.tolist(),'family':config['family'],
            'transform_coordinates':'target crop -> query crop','query_arc_fraction':ar[ia[i]],'target_arc_fraction':br[ib[i]],
            'scale':float(scales[i]),'fitted_overlap_length_query_px':float(length[i]),'rms_residual_query_px':float(rms[i]),
            'normalized_residual':float(rms[i]/length[i]),'inward_normal_penalty':float(normals[i]),
            'geometry_score':float(np.exp(-min(float(objective[i]),700.))),
            'shape_anisotropy':anisotropy,'underconstrained':anisotropy<.005,'pose_confidence_calibrated':False})
        if len(selected)==config['alternatives_per_parent']:break
    return selected


def fit_parent(parent,inputs,config):
    q,t=parent['query'],parent['target'];candidates=[];tested=0
    for windows in parent['windows']:
        wa,wb=windows;samples=512 if wa is None else 256;profile={'samples':samples}
        sa,sb=span_for(inputs,q,wa),span_for(inputs,t,wb)
        a,b=inputs.features(q,profile,sa),inputs.features(t,profile,sb)
        prepared=(inputs.subarcs(q,profile,sa),inputs.subarcs(t,profile,sb,True))
        for fit in hypotheses(a,b,inputs.config,config,prepared,owners=(q,t)):
            candidates.append({'query':q,'target':t,'windows':windows,'samples':samples,'span_a':sa,'span_b':sb,'alignment':fit})
        tested+=1
    candidates.sort(key=lambda v:(-v['alignment']['geometry_score'],-v['alignment']['fitted_overlap_length_query_px'],str(v['windows'])))
    selected=[];probe=inputs.features(t,{'samples':128})['points_xy_crop'][::32]
    # Mask checks are reserved for the strongest bounded alternatives, after
    # subarc geometry has been scored without labels.
    for value in candidates[:24]:
        fit=value['alignment']
        if any(transform_distance(fit['transform'],v['alignment']['transform'],probe)<2. for v in selected):continue
        collision=collision_fraction(inputs.mask(q),inputs.mask(t),fit['transform'],config['collision_grid'],
            (inputs.mask_samples(q),inputs.mask_samples(t)))
        if collision>config['maximum_pair_collision']:continue
        fit['mask_collision_proxy']=collision;selected.append(value)
        if len(selected)==config['alternatives_per_parent']:break
    return {'query':q,'target':t,'input_hypotheses':tested,'poses':selected}
