"""Observable partial fitting. Transforms map the target crop into the query crop."""
from __future__ import annotations

import numpy as np

from .contracts import require


def apply_transform(points, matrix):
    points, matrix = np.asarray(points), np.asarray(matrix)
    numerator = points @ matrix[:2, :2].T + matrix[:2, 2]
    # Preserve the exact legacy affine arithmetic, including float operation order.
    if np.array_equal(matrix[2], [0., 0., 1.]): return numerator
    denominator = points @ matrix[2, :2] + matrix[2, 2]
    require(np.isfinite(denominator).all() and np.all(np.abs(denominator) > 1e-10),
            'Transform crosses a projective horizon')
    return numerator / denominator[..., None]


def translated_transform(matrix, offset):
    """Translate the output frame; adding only h13/h23 is wrong for homographies."""
    moved = np.asarray(matrix, float).copy()
    moved[:2] += np.asarray(offset)[:, None] * moved[2]
    return moved


def fit_transforms(first, second, *, similarity=False, scale_bounds=(.5, 2.0)):
    """Batched proper-rotation Procrustes. Reflections are never allowed."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[-1] == 2 and a.shape[1] >= 2,
            "Invalid point correspondences for fitting")
    ac, bc = a.mean(axis=1), b.mean(axis=1)
    aa, bb = a - ac[:, None], b - bc[:, None]
    covariance = np.einsum("hni,hnj->hij", bb, aa)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
    correction = np.ones((len(a), 2)); correction[:, 1] = sign
    rotation = (u * correction[:, None, :]) @ vt
    scale = np.ones(len(a))
    if similarity:
        numerator = (singular * correction).sum(axis=1)
        denominator = np.sum(bb * bb, axis=(1, 2))
        scale = np.clip(numerator / np.maximum(denominator, 1e-12), *scale_bounds)
    linear = scale[:, None, None] * rotation
    translation = ac - np.einsum("hi,hij->hj", bc, linear)
    matrices = np.broadcast_to(np.eye(3), (len(a), 3, 3)).copy()
    matrices[:, :2, :2] = linear.transpose(0, 2, 1)
    matrices[:, :2, 2] = translation
    fitted = np.einsum("hni,hij->hnj", b, linear) + translation[:, None]
    return matrices, fitted, scale, rotation


def _subarcs(features, config, reverse=False):
    length = float(features["observed_length_px"][0])
    xy = features["points_xy_crop"].astype(np.float64)
    normals = features["inward_normal_xy_crop"].astype(np.float64)
    if reverse: xy, normals = xy[::-1], normals[::-1]
    intervals, positions, normal_values, lengths = [], [], [], []
    for fraction in config.alignment_fractions:
        if fraction * length < config.minimum_fit_length_px: continue
        for start in sorted(set((0., (1. - fraction) / 2, 1. - fraction))):
            grid = np.linspace(start, start + fraction, config.alignment_samples)
            native = np.linspace(0, 1, len(xy))
            points = np.column_stack([np.interp(grid, native, xy[:, i]) for i in range(2)])
            normal = np.column_stack([np.interp(grid, native, normals[:, i]) for i in range(2)])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            # Range is always in original traversal, including for the reversed target.
            interval = [1 - start - fraction, 1 - start] if reverse else [start, start + fraction]
            intervals.append(interval); positions.append(points); normal_values.append(normal); lengths.append(fraction * length)
    return intervals, np.asarray(positions), np.asarray(normal_values), np.asarray(lengths)


def mask_samples(mask, grid):
    h, w = mask.shape
    x, y = np.meshgrid(np.linspace(0, w - 1, grid), np.linspace(0, h - 1, grid))
    points = np.column_stack((x.ravel(), y.ravel()))
    paper = mask[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)] >= 128
    return points, paper


def collision_fraction(first_mask, second_mask, matrix, grid=24, prepared=None):
    """Sparse mask overlap proxy, sampled in both crops; not exact polygon collision."""
    if not grid: return None
    def direction(mask, other, transform, sampled):
        points, paper = sampled if sampled is not None else mask_samples(mask, grid)
        mapped = apply_transform(points, transform); xx, yy = np.rint(mapped).astype(np.int64).T
        inside = (xx >= 0) & (xx < other.shape[1]) & (yy >= 0) & (yy < other.shape[0])
        hits = np.zeros(len(points), bool);hits[inside] = other[yy[inside], xx[inside]] >= 128
        return float(np.sum(paper & hits) / np.sum(paper)) if paper.any() else 0.0
    sampled = prepared or (None, None)
    return max(direction(first_mask, second_mask, np.linalg.inv(matrix), sampled[0]), direction(second_mask, first_mask, matrix, sampled[1]))


def partial_fit(first, second, config, *, family="rigid", masks=None, prepared=None, prepared_masks=None):
    require(family in {"rigid", "similarity"}, "Unknown analytic transform family")
    ar, a, an, al = prepared[0] if prepared is not None else _subarcs(first, config)
    br, b, bn, bl = prepared[1] if prepared is not None else _subarcs(second, config, reverse=True)
    if not len(a) or not len(b):
        return {"status": "insufficient_observed_length", "score": 0.0, "transform": None,
                "family": family, "direction": "target_reversed", "underconstrained": True}
    ia, ib = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij");ia, ib = ia.ravel(), ib.ravel()
    matrices, fitted, scales, rotations = fit_transforms(a[ia], b[ib], similarity=family == "similarity",
                                                       scale_bounds=(config.minimum_scale, config.maximum_scale))
    error = np.linalg.norm(fitted - a[ia], axis=2)
    rms = np.sqrt(np.mean(error ** 2, axis=1))
    denominator = np.maximum(np.minimum(al[ia], bl[ib] * scales), 1e-6)
    transformed_normals = np.einsum("hni,hij->hnj", bn[ib], rotations)
    normal_penalty = np.mean((1 + np.sum(an[ia] * transformed_normals, axis=2)) / 2, axis=1)
    length_penalty = np.abs(np.log(np.maximum(al[ia], 1e-6) / np.maximum(bl[ib] * scales, 1e-6)))
    objective = rms / denominator / .03 + 2 * normal_penalty + .25 * length_penalty
    # Ties favor the longer supported arc. No seam ID chooses the fit.
    chosen = min(range(len(objective)), key=lambda i: (float(objective[i]), -float(denominator[i]), int(ia[i]), int(ib[i])))
    matrix = matrices[chosen]
    eigenvalues = np.linalg.eigvalsh(np.cov(a[ia[chosen]].T))
    anisotropy = float(eigenvalues[0] / max(eigenvalues[-1], 1e-12))
    collision = collision_fraction(*masks, matrix, config.collision_grid, prepared_masks) if masks is not None else None
    cost = float(objective[chosen]) + (2 * collision if collision is not None else 0)
    return {"status": "fitted", "score": float(np.exp(-min(max(cost, 0), 700))), "family": family,
            "transform": matrix.tolist(), "transform_coordinates": "target crop -> query crop",
            "direction": "target_reversed", "scale": float(scales[chosen]),
            "query_arc_fraction": ar[ia[chosen]], "target_arc_fraction": br[ib[chosen]],
            "fitted_overlap_length_query_px": float(denominator[chosen]), "rms_residual_query_px": float(rms[chosen]),
            "normalized_residual": float(rms[chosen] / denominator[chosen]),
            "inward_normal_penalty": float(normal_penalty[chosen]), "length_ratio_penalty": float(length_penalty[chosen]),
            "mask_collision_proxy": collision, "shape_anisotropy": anisotropy,
            "underconstrained": bool(anisotropy < .005), "hypotheses_tested": len(objective),
            "confidence_is_calibrated": False}
