"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
import numpy as np

def tangent(feature):
    xy=np.asarray(feature['points_xy_crop']);delta=np.diff(xy,axis=0)
    direction=np.sum(delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12),axis=0)
    if np.linalg.norm(direction)<1e-8:
        _,vectors=np.linalg.eigh(np.cov(xy.T));direction=vectors[:,-1]
    direction=direction/max(np.linalg.norm(direction),1e-12)
    if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    return direction
