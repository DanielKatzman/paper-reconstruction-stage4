"""Actual-placement RGB continuation profiles; masks and texture can abstain."""
import numpy as np

from .alignment import apply_transform,translated_transform
from .features import bilinear
from .pose_boundary_evidence import correspondences,transformed
from .pose_observability_geometry import tangent
from .visual_preparation import prepare_source


def visual_profile(first,second,pose,pixels_a,pixels_b,correction,policy,_preparation=None,_query_owner=None):
    """No labels or learned weights. All shifts compare identical valid samples.

    Pair nearest normal-compatible boundaries under the proposed transform,
    then sample each piece inward. This preserves tangential association while
    geometry separately measures the gap. Fixed query samples are used for all
    offsets. RGB high-pass signatures remove constant brightness offsets.
    """
    rgb_a,mask_a=pixels_a;rgb_b,mask_b=pixels_b
    source=prepare_source(first,pixels_a,policy) if _preparation is None else _preparation.source(_query_owner,first,pixels_a,policy)
    a,an,direction,depths,ia,va=(source[k] for k in ('a','an','direction','depths','ia','va'))
    matrix=np.asarray(pose['alignment']['transform'],float);b,bn,_=transformed(second,matrix)
    center_match=correspondences(a,an,b,bn);distance=center_match[0]
    target=[];valid=[]
    for offset in policy['visual_offsets_query_px']:
        moved=translated_transform(matrix,offset*direction)
        if offset==0 and moved.tobytes(order='C')==matrix.tobytes(order='C'):
            mb,mbn=b,bn;da,ja,u,projected=center_match
            if _preparation is not None:_preparation.center_reuses+=1
        else:
            mb,mbn,_=transformed(second,moved);da,ja,u,projected=correspondences(a,an,mb,mbn)
            if offset==0 and _preparation is not None:_preparation.center_byte_mismatch_fallbacks+=1
        inward=mbn[ja]*(1-u[:,None])+mbn[ja+1]*u[:,None]
        inward/=np.maximum(np.linalg.norm(inward,axis=1,keepdims=True),1e-12)
        bands=projected[:,None]+inward[:,None]*depths[None,:,None]
        native=apply_transform(bands.reshape(-1,2),np.linalg.inv(moved)).reshape(bands.shape)
        target.append(bilinear(rgb_b,native).mean(axis=1))
        valid.append((bilinear(mask_b,native)>=policy['visual_mask_validity']).all(axis=1)&np.isfinite(da))
    common=(distance<=correction['support_distance_px'])&va&np.logical_and.reduce(valid)
    radius=policy['visual_highpass_radius'];width=2*radius+1
    common=np.convolve(common.astype(int),np.ones(width,dtype=int),mode='same')==width
    def highpass(values):
        smooth=np.column_stack([np.convolve(values[:,c],np.ones(width)/width,mode='same') for c in range(3)])
        return (values-smooth)[common].astype(float)
    fa=highpass(ia);fb=[highpass(v) for v in target]
    count=int(common.sum());length=float(first['observed_length_px'][0])*count/len(a)
    energy_a=float(np.sqrt(np.mean(fa**2))) if count else 0.
    energy_b=[float(np.sqrt(np.mean(v**2))) if count else 0. for v in fb]
    support=count>=policy['visual_minimum_samples'] and length>=policy['visual_minimum_length_query_px']
    textured=energy_a>=policy['visual_minimum_texture_rms'] and min(energy_b,default=0.)>=policy['visual_minimum_texture_rms']
    informative=bool(support and textured);rows=[]
    for offset,v,energy in zip(policy['visual_offsets_query_px'],fb,energy_b):
        den=float(np.linalg.norm(fa)*np.linalg.norm(v));correlation=float(np.clip(np.sum(fa*v)/den,-1,1)) if den>1e-12 else None
        rows.append({'offset_query_px':offset,'correlation':correlation,'target_texture_rms':energy})
    center=next(v for v in rows if v['offset_query_px']==0);best=None;near=[];span=None
    if informative:
        best=max(rows,key=lambda v:(v['correlation'],-abs(v['offset_query_px']),-v['offset_query_px']))
        near=[v['offset_query_px'] for v in rows if v['correlation']>=best['correlation']-policy['visual_score_slack']]
        span=float(max(near)-min(near))
    boundary=bool(near and (min(near)==min(policy['visual_offsets_query_px']) or max(near)==max(policy['visual_offsets_query_px'])))
    off=bool(best and abs(best['offset_query_px'])>=4 and best['correlation']>center['correlation']+policy['visual_score_slack'])
    ambiguous=not informative or span>=policy['visual_ambiguous_width_query_px'] or boundary or off
    adjustment=policy['visual_weight_limit']*center['correlation'] if informative else 0.
    return {'informative':informative,'status':'informative' if informative else 'insufficient_support' if not support else 'insufficient_texture',
        'common_samples':count,'common_length_query_px':length,'common_fraction':count/len(a),'query_texture_rms':energy_a,
        'central_correlation':center['correlation'] if informative else None,'profile':rows,'direction_query_xy':direction.tolist(),
        'near_best_offsets_query_px':near,'near_best_width_query_px':span,'best_offset_query_px':best['offset_query_px'] if best else None,
        'best_at_scan_boundary':boundary,'better_off_center':off,'visually_ambiguous':bool(ambiguous),
        'bounded_weight_adjustment':float(adjustment),'confidence_calibrated':False,'validation_truth_opened':False,
        'central_boundary_rms_query_px':float(np.sqrt(np.mean(distance[common]**2))) if count else None,
        'scope':'Heuristic RGB continuation between boundaries paired under the actual transform. Own inward bands preserve tangential association; geometric gaps are checked separately. Not learned compatibility or proof of a correct join.'}
