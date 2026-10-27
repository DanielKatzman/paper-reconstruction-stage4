"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
import numpy as np

def observed_risk(inputs):
    """Global spacing and predicted uncertain/transition mass, without labels."""
    spacing=[];uncertainty=[]
    for owner in range(len(inputs.gallery)):
        f=inputs.features(owner,{'samples':128});spacing.append(float(f['observed_length_px'][0])/511.)
        uncertainty.append(float(np.clip(f['probabilities'][:,2:].sum(1),0,1).mean()))
    return np.asarray(spacing),np.asarray(uncertainty)
