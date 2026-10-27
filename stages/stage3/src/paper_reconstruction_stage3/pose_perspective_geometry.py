"""Bounded multi-start similarity and regularized projective contour refinement."""
import numpy as np
from .alignment import apply_transform,fit_transforms
from .pose_boundary_evidence import correspondences,transformed
from .projective import fit_homography,warp_check,grid


def blocks(n,count):return np.minimum(np.arange(n)*count//n,count-1)


def distances(first,second,matrix):
    a=np.asarray(first['points_xy_crop']);an=np.asarray(first['inward_normal_xy_crop'])
    b,bn,_=transformed(second,matrix)
    da,ja,ua,pa=correspondences(a,an,b,bn);db,jb,ub,pb=correspondences(b,bn,a,an)
    return da,db,pa,pb,ja,jb


def validate_fixed(first,second,matrix,fixed,policy):
    da,db,_,_,_,_=distances(first,second,matrix);scores=[];support=[];groups=[]
    for d,use in zip((da,db),fixed):
        scores.extend(np.exp(-d[use]/4.));support.extend(d[use]<=policy['refinement_distance_px'])
        identity=blocks(len(d),policy['validation_blocks'])
        groups.extend(float(np.exp(-d[use&(identity==i)]/4.).mean()) for i in range(policy['validation_blocks']) if np.any(use&(identity==i)))
    return {'score':float(np.mean(scores)) if scores else 0.,'support_fraction':float(np.mean(support)) if support else 0.,
        'block_scores':groups,'samples':[int(x.sum()) for x in fixed]}


from .unit_scale import refine_rigid as refine
