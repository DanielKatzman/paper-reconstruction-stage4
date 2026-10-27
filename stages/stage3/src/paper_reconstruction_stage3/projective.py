"""Observable projective math; no saved labels, reference fits or model imports."""
import numpy as np
from .alignment import apply_transform


def jacobians(points, matrix):
    points=np.asarray(points,float);h=np.asarray(matrix,float)
    den=points@h[2,:2]+h[2,2];mapped=apply_transform(points,h)
    return (h[None,:2,:2]-mapped[:,:,None]*h[None,2,:2])/den[:,None,None]


def warp_normals(points,normals,matrix):
    value=np.einsum('ni,nij->nj',np.asarray(normals),np.linalg.inv(jacobians(points,matrix)))
    return value/np.maximum(np.linalg.norm(value,axis=1,keepdims=True),1e-12)


def grid(shape):
    h,w=shape
    x,y=np.meshgrid(np.linspace(0,w-1,3),np.linspace(0,h-1,3))
    return np.column_stack((x.ravel(),y.ravel()))


def warp_check(matrix,source_shape,target_shape,policy):
    """Check both complete crop rectangles, not just the fitted seam."""
    h=np.asarray(matrix,float);stats=[]
    if h.shape!=(3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-12:return {'valid':False,'reason':'singular_warp'}
    for value,shape in ((h,source_shape),(np.linalg.inv(h),target_shape)):
        points=grid(shape);den=points@value[2,:2]+value[2,2]
        if np.min(den)*np.max(den)<=0 or min(abs(den))/max(abs(den))<policy['minimum_denominator_ratio']:
            return {'valid':False,'reason':'crop_horizon_or_extreme_perspective'}
        jac=jacobians(points,value);det=np.linalg.det(jac)
        if np.any(det<=0):return {'valid':False,'reason':'orientation_reversal'}
        singular=np.linalg.svd(jac,compute_uv=False)
        if singular.min()<policy['minimum_local_scale'] or singular.max()>policy['maximum_local_scale']:
            return {'valid':False,'reason':'local_scale_out_of_bounds'}
        ratio=singular[:,0]/singular[:,1]
        if ratio.max()>policy['maximum_local_anisotropy']:return {'valid':False,'reason':'excessive_anisotropy'}
        stats.append({'minimum_scale':float(singular.min()),'maximum_scale':float(singular.max()),
            'maximum_anisotropy':float(ratio.max()),'denominator_ratio':float(min(abs(den))/max(abs(den)))})
    return {'valid':True,'reason':'bounded_positive_warp','forward':stats[0],'inverse':stats[1]}


def fit_homography(a,b,prior,policy):
    """Normalized ridge fit toward the current observable seed, not an oracle fit."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<policy['minimum_fit_pairs']:return None,{'reason':'insufficient_fit_pairs'}
    def normalize(x):
        center=x.mean(0);scale=max(float(np.sqrt(np.mean(np.sum((x-center)**2,axis=1)))),1e-12)
        t=np.array([[1/scale,0,-center[0]/scale],[0,1/scale,-center[1]/scale],[0,0,1.]])
        return (x-center)/scale,t
    aa,ta=normalize(a);bb,tb=normalize(b)
    anisotropy=[]
    for points in (aa,bb):
        eigen=np.linalg.eigvalsh(np.cov(points.T));anisotropy.append(float(eigen[0]/max(eigen[-1],1e-12)))
    if min(anisotropy)<policy['minimum_fit_anisotropy']:return None,{'reason':'nearly_straight_fit_support','anisotropy':anisotropy}
    x,y=bb.T;u,v=aa.T;n=len(a);design=np.zeros((2*n,8))
    design[::2,:3]=np.c_[x,y,np.ones(n)];design[::2,6:]=-np.c_[u*x,u*y]
    design[1::2,3:6]=np.c_[x,y,np.ones(n)];design[1::2,6:]=-np.c_[v*x,v*y]
    condition=float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design)<8 or condition>policy['maximum_fit_condition']:return None,{'reason':'ill_conditioned_fit','condition':condition}
    p=ta@np.asarray(prior)@np.linalg.inv(tb)
    if abs(p[2,2])<1e-10:return None,{'reason':'unstable_prior_gauge'}
    p=(p/p[2,2]).ravel()[:8];ridge=policy['projective_ridge']*len(a)
    coef=np.linalg.lstsq(np.vstack((design,np.eye(8)*np.sqrt(ridge))),np.r_[aa.ravel(),p*np.sqrt(ridge)],rcond=None)[0]
    h=np.linalg.inv(ta)@np.r_[coef,1.].reshape(3,3)@tb
    if abs(h[2,2])<1e-10:return None,{'reason':'unstable_fitted_gauge'}
    return h/h[2,2],{'reason':'fitted','condition':condition,'anisotropy':anisotropy,'fit_pairs':len(a),'ridge':ridge}
