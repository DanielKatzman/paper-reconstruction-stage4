"""Homogeneous math; warp guards adapted from the archived Stage 3 control."""
import numpy as np
from functools import lru_cache
from paper_reconstruction_contracts import map_points


def normalize(value):
    h = np.asarray(value, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12 or abs(h[2, 2]) < 1e-12:
        raise ValueError("Invalid/singular homogeneous transform")
    return h / h[2, 2]


@lru_cache(maxsize=256)
def _probe_grid(shape, grid):
    height, width = shape
    x, y = np.meshgrid(np.linspace(0, width - 1, grid), np.linspace(0, height - 1, grid))
    result = np.column_stack((x.ravel(), y.ravel()))
    result.flags.writeable = False
    return result


def probes(shape, grid=3):
    # Callers retain independent writable arrays; the expensive static grid is
    # immutable and bounded by shape/resolution, never by a rounded transform.
    return _probe_grid(tuple(shape), grid).copy()


def jacobians(points, matrix):
    h = normalize(matrix)
    den = points @ h[2, :2] + h[2, 2]
    mapped = map_points(points, h)
    return (h[None, :2, :2] - mapped[:, :, None] * h[None, 2, :2]) / den[:, None, None]


def warp_check(matrix, source_shape, target_shape, config, grid=5):
    if config.page_policy != 'disabled.v1':
        from .constraints import rigid_check
        check = rigid_check(matrix)
        if not check['valid']:
            return check
    try:
        h = normalize(matrix)
        stats = []
        for value, shape in ((h, source_shape), (np.linalg.inv(h), target_shape)):
            # An exactly affine warp has constant denominator/Jacobian across
            # the crop. Evaluate the same guards once instead of 25 identical
            # SVDs. Every projective/non-unit-denominator case keeps the full
            # sampled path; there is no approximate-affine threshold.
            if (getattr(config,'joint_execution_policy','baseline.v1') == 'exact-reuse.v1' and
                    value[2,0] == 0. and value[2,1] == 0. and value[2,2] == 1.):
                jac = value[:2,:2]
                if np.linalg.det(jac) <= 0:
                    return {"valid": False, "reason": "orientation_reversal"}
                singular = np.linalg.svd(jac,compute_uv=False)
                if singular.min() < config.minimum_local_scale or singular.max() > config.maximum_local_scale:
                    return {"valid": False, "reason": "local_scale_out_of_bounds"}
                anisotropy = singular[0]/singular[1]
                if anisotropy > config.maximum_local_anisotropy:
                    return {"valid": False, "reason": "excessive_anisotropy"}
                stats.append({"minimum_scale":float(singular.min()),"maximum_scale":float(singular.max()),
                              "maximum_anisotropy":float(anisotropy)})
                continue
            points = probes(shape, grid)
            den = points @ value[2, :2] + value[2, 2]
            if np.min(den) * np.max(den) <= 0 or np.min(np.abs(den)) / np.max(np.abs(den)) < config.minimum_denominator_ratio:
                return {"valid": False, "reason": "crop_horizon_or_extreme_perspective"}
            jac = jacobians(points, value)
            if np.any(np.linalg.det(jac) <= 0):
                return {"valid": False, "reason": "orientation_reversal"}
            singular = np.linalg.svd(jac, compute_uv=False)
            if singular.min() < config.minimum_local_scale or singular.max() > config.maximum_local_scale:
                return {"valid": False, "reason": "local_scale_out_of_bounds"}
            anisotropy = singular[:, 0] / singular[:, 1]
            if anisotropy.max() > config.maximum_local_anisotropy:
                return {"valid": False, "reason": "excessive_anisotropy"}
            stats.append({"minimum_scale": float(singular.min()), "maximum_scale": float(singular.max()),
                          "maximum_anisotropy": float(anisotropy.max())})
        return {"valid": True, "reason": "bounded_positive_warp", "sampling_grid": grid, "directions": stats}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return {"valid": False, "reason": "singular_or_nonfinite_warp"}


def relative(poses, a, b):
    return normalize(np.linalg.inv(poses[a]) @ poses[b])


def pose_error(first, second, shape):
    return float(np.max(np.linalg.norm(map_points(probes(shape), first) - map_points(probes(shape), second), axis=1)))


def regauge(poses, members):
    root = min(members)
    gauge = np.linalg.inv(poses[root])
    return {p: (normalize(gauge @ h) if p in members else h) for p, h in poses.items()}
