"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
import numpy as np

def fitted_evidence(fit):
    """The original eight evidence values, independent of strip resampling."""
    if fit['status'] != 'fitted': return np.zeros(8, np.float32)
    return np.array([fit["normalized_residual"],fit["inward_normal_penalty"],fit["length_ratio_penalty"],fit["scale"],
        np.diff(fit["query_arc_fraction"])[0],np.diff(fit["target_arc_fraction"])[0],float(fit["underconstrained"]),fit["score"]],np.float32)


def observable_pair_evidence(a,b):
    """Reuse the baseline's observable partial rigid fit; no label-driven join."""
    from .alignment import partial_fit
    from .baselines import BaselineConfig
    def fit_input(x):return {"points_xy_crop":x["points"],"inward_normal_xy_crop":x["normals"],"observed_length_px":x["observed_length"]}
    fit=partial_fit(fit_input(a),fit_input(b),BaselineConfig(),family="rigid")
    if fit["status"]!="fitted":return np.zeros(8,np.float32),np.zeros((8,*a["strip"].shape[1:]),np.float32) if "strip" in a else None
    evidence=fitted_evidence(fit)
    joined=None
    if "strip" in a:
        def resample(strip,interval,reverse=False):
            n=strip.shape[1];positions=np.linspace(*interval,n)
            if reverse:positions=positions[::-1]
            native=np.linspace(0,1,n);out=np.empty_like(strip)
            for ch in range(strip.shape[0]):
                for depth in range(strip.shape[2]):out[ch,:,depth]=np.interp(positions,native,strip[ch,:,depth])
            return out
        # Two masked sides, aligned by the fitted observable subarcs. Eight channels
        # preserve validity independently; compatibility must still be learned.
        joined=np.concatenate((resample(a["strip"],fit["query_arc_fraction"]),resample(b["strip"],fit["target_arc_fraction"],True)),0)
    return evidence,joined
