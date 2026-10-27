"""Bounded sliding overlap fits from observed curves only; no label dependencies."""
import numpy as np
from .alignment import fit_transforms
from .contracts import digest
from .pose_correction_geometry import diverse
from .unit_scale import native_parameters, refinement_parameters


def intervals(length, minimum, fractions, divisions):
    """A fixed position grid, including both ends at every retained overlap scale."""
    values=[]
    for fraction in fractions:
        if fraction*length<minimum:continue
        count=max(1,int(np.ceil((1-fraction)*divisions)))
        for start in np.linspace(0.,1-fraction,count+1) if fraction<1 else [0.]:
            values.append((float(start),float(fraction)))
    return values


def fits(first,second,parameters,base):
    """Batched proper-rotation fits; target traversal is reversed exactly once."""
    parameters=np.asarray(parameters,float).reshape(-1,4)
    if not len(parameters):return []
    grid=np.linspace(0.,1.,base['alignment_samples']);points=[];normals=[]
    for side,feature in enumerate((first,second)):
        start=parameters[:,2*side];length=parameters[:,2*side+1]
        u=start[:,None]+length[:,None]*(grid if side==0 else 1-grid)
        native=np.linspace(0.,1.,len(feature['points_xy_crop']))
        xy=np.stack([np.interp(u,native,feature['points_xy_crop'][:,k]) for k in range(2)],axis=-1)
        no=np.stack([np.interp(u,native,feature['inward_normal_xy_crop'][:,k]) for k in range(2)],axis=-1)
        no/=np.maximum(np.linalg.norm(no,axis=-1,keepdims=True),1e-12)
        points.append(xy);normals.append(no)
    matrices,fitted,scales,rotations=fit_transforms(*points,similarity=base['family']=='similarity',
        scale_bounds=(base['minimum_scale'],base['maximum_scale']))
    rms=np.sqrt(np.mean(np.sum((fitted-points[0])**2,axis=2),axis=1))
    la=parameters[:,1]*float(first['observed_length_px'][0]);lb=parameters[:,3]*float(second['observed_length_px'][0])
    length=np.maximum(np.minimum(la,lb*scales),1e-6)
    rotated=np.einsum('hni,hij->hnj',normals[1],rotations)
    penalty=np.clip(np.mean((1+np.sum(normals[0]*rotated,axis=2))/2,axis=1),0.,1.)
    objective=rms/length/.03+2*penalty+.25*np.abs(np.log(np.maximum(la,1e-6)/np.maximum(lb*scales,1e-6)))
    eig=np.linalg.eigvalsh(np.einsum('hni,hnj->hij',points[0]-points[0].mean(axis=1,keepdims=True),points[0]-points[0].mean(axis=1,keepdims=True)))
    out=[]
    for i,h in enumerate(matrices):
        p=parameters[i];anisotropy=float(eig[i,0]/max(eig[i,1],1e-12))
        fit={'status':'fitted','family':base['family'],'transform':h.tolist(),'transform_coordinates':'target crop -> query crop',
            'query_arc_fraction':[float(p[0]),float(p[0]+p[1])],'target_arc_fraction':[float(p[2]),float(p[2]+p[3])],
            'scale':float(scales[i]),'geometry_score':float(np.exp(-min(float(objective[i]),700.))),
            'normalized_residual':float(rms[i]/length[i]),'inward_normal_penalty':float(penalty[i]),
            'rms_residual_query_px':float(rms[i]),'fitted_overlap_length_query_px':float(length[i]),
            'shape_anisotropy':anisotropy,'underconstrained':anisotropy<.005,'pose_confidence_calibrated':False}
        out.append({'windows':[None,None],'samples':512,'span_a':None,'span_b':None,'alignment':fit,
            'proposal_id':digest(h.tolist())[:24],'_parameters':p.tolist(),'_objective':float(objective[i])})
    return out


def sliding_fits(first,second,base,configuration):
    lengths=[float(f['observed_length_px'][0]) for f in (first,second)]
    parameters=native_parameters(first,second,base,configuration)
    coarse=[];chunk=configuration['overlap_fit_batch']
    for start in range(0,len(parameters),chunk):coarse.extend(fits(first,second,parameters[start:start+chunk],base))
    # Diverse starts are chosen by observable geometry before any evaluator sees fits.
    probes=second['points_xy_crop'][::64]
    seeds=diverse(coarse,configuration['overlap_refinement_seeds'],lambda p:p['alignment']['geometry_score'],probes)
    refined=[];current=seeds
    for step in configuration['overlap_refinement_steps']:
        for _ in range(2):
            parameters,owners=refinement_parameters(current,lengths,base['minimum_fit_length_px'],step)
            values=fits(first,second,parameters,base);refined.extend(values)
            next_=list(current)
            for pose,owner in zip(values,owners):
                if pose['_objective']<next_[owner]['_objective']-1e-12:next_[owner]=pose
            current=next_
    return coarse,refined,{'coarse_fits':len(coarse),'refinement_trials':len(refined),'refinement_starts':len(seeds),
        'position_divisions':configuration['overlap_position_divisions'],'overlap_scales':list(configuration['overlap_fractions']),
        'fit_batch':chunk,'reference_anchors_used':False}
