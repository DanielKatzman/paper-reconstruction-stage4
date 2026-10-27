"""Observable-only fragment representations shared by research and production."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .contracts import require
from .data import ObservableContour

GEOMETRY_CHANNELS = ["x", "y", "tangent_x", "tangent_y", "inward_x", "inward_y",
                     "curvature_times_length", "corner_strength_times_length", "thickness_over_length"]
PROBABILITY_CHANNELS = ["seam", "page_border", "uncertain", "transition"]


@dataclass(frozen=True)
class FeatureConfig:
    samples: int = 256
    strip_depth: int = 33
    context_size: int = 0

    def validate(self):
        require(type(self.samples) is int and self.samples in {128, 256, 512}, "Feature samples must be 128, 256, or 512")
        require(type(self.strip_depth) is int and self.strip_depth in {33, 65, 97}, "Strip depth must be 33, 65, or 97")
        require(type(self.context_size) is int and self.context_size in {0, 128, 256}, "Context size must be 0, 128, or 256")

    def manifest(self):
        self.validate()
        return {"schema_version": "stage3.observable-features.v1", **asdict(self),
                "geometry_channels": GEOMETRY_CHANNELS, "probability_channels": PROBABILITY_CHANNELS,
                "strip_channels": ["masked_r", "masked_g", "masked_b", "mask_validity"],
                "strip_axis_0": "candidate traversal", "strip_axis_1": "inward offset in crop pixels",
                "normalization": "center and observed open-polyline length; first observed tangent frame",
                "rank_and_truth_in_features": False}


def _unit(values):
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    require(np.all(lengths > 1e-7), "Cannot normalize a degenerate resampled direction")
    return values / lengths


def resampling_grid(points, samples: int, *, arc_start_px=0.0, arc_end_px=None):
    """The common observable arc grid; supervision never chooses sample locations."""
    FeatureConfig(samples=samples).validate()
    xy = np.asarray(points, dtype=np.float64)
    require(xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all(), "Invalid fragment points")
    require(len(xy) >= 2, "A fragment needs at least two points")
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(distance) > 1e-8))
    require(np.count_nonzero(keep) >= 2 and distance[-1] > 1e-6, "Fragment has zero observed length")
    end = float(distance[-1]) if arc_end_px is None else float(arc_end_px)
    start = float(arc_start_px)
    require(np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= distance[-1] + 1e-7,
            "Window arc range must lie inside the observed open candidate")
    target = np.linspace(start, min(end, float(distance[-1])), samples)
    return distance, keep, target


def native_sample_indices(points, samples: int, *, arc_start_px=0.0, arc_end_px=None) -> np.ndarray:
    """Nearest native vertices on the same arc grid used by observable features.

    This is a diagnostic mapping, not a change to linear feature interpolation.
    Ties use the earlier retained vertex; duplicate/zero-length vertices are omitted.
    """
    distance, keep, target = resampling_grid(points, samples, arc_start_px=arc_start_px, arc_end_px=arc_end_px)
    arc = distance[keep]
    middle = (arc[:-1] + arc[1:]) / 2
    return np.flatnonzero(keep)[np.searchsorted(middle, target, side="left")]


def resample_fragment(contour: ObservableContour, candidate: dict, samples: int = 256, *, window=None) -> dict:
    fragment = contour.fragment(candidate)
    arguments = {} if window is None else {"arc_start_px": window["start_px"], "arc_end_px": window["end_px"]}
    distance, keep, target = resampling_grid(fragment["points_xy"], samples, **arguments)
    result = {}
    for name, values in fragment.items():
        values = values[keep]
        result[name] = (np.interp(target, distance[keep], values) if values.ndim == 1 else
                        np.column_stack([np.interp(target, distance[keep], values[:, i]) for i in range(values.shape[1])]))
    for name in ("tangent_xy", "inward_normal_xy"):
        if window is not None:
            # At an exact antipodal interpolation point there is no unique
            # average direction. Use the same observable nearest-vertex tie
            # rule as the window grid instead of creating a zero/NaN frame.
            bad = np.linalg.norm(result[name], axis=1) <= 1e-7
            if np.any(bad):
                native = native_sample_indices(fragment["points_xy"], samples, **arguments)
                result[name][bad] = fragment[name][native[bad]]
        result[name] = _unit(result[name])
    length = float(target[-1] - target[0])
    result["observed_length_px"] = length
    result["sample_arc_fraction"] = (target - target[0]) / length
    if window is not None:
        result["parent_observed_length_px"] = float(distance[-1])
        result["window_arc_range_px"] = np.array([target[0], target[-1]])
    return result


def masked_pixels(contour: ObservableContour) -> tuple[np.ndarray, np.ndarray]:
    require(contour.image_path is not None and contour.mask_path is not None, "Appearance requires RGB and mask paths")
    with Image.open(contour.image_path) as opened, Image.open(contour.mask_path) as mask:
        require(opened.size == mask.size, "Appearance RGB/mask sizes disagree")
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        alpha = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return rgb * alpha[:, :, None], alpha


def bilinear(pixels: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Zero-padded bilinear sampling; outside pixels never become valid."""
    height, width = pixels.shape[:2]
    x, y = coordinates[..., 0], coordinates[..., 1]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    trailing = pixels.shape[2:]
    result = np.zeros(x.shape + trailing, dtype=np.float32)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = x0 + dx, y0 + dy
        weight = (1 - np.abs(x - xx)) * (1 - np.abs(y - yy))
        inside = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < height)
        sampled = pixels[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)]
        factor = weight * inside
        result += sampled * (factor[..., None] if trailing else factor)
    return result


def _context(rgb, mask, size):
    height, width = mask.shape
    scale = min(size / width, size / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    result = np.zeros((size, size, 4), dtype=np.float32)
    source = np.concatenate((rgb, mask[..., None]), axis=-1)
    x, y = (size - w) // 2, (size - h) // 2
    for channel in range(4):
        resized = Image.fromarray(source[:, :, channel]).resize((w, h), Image.Resampling.BILINEAR)
        result[y:y+h, x:x+w, channel] = np.asarray(resized)
    return np.clip(result, 0, 1)


def fragment_features(contour: ObservableContour, candidate: dict, config: FeatureConfig = FeatureConfig(),
                      *, reverse: bool = False, appearance: bool = True, pixels=None, window=None) -> dict[str, np.ndarray]:
    """Returns only observable numeric blocks; no canonical IDs or labels."""
    config.validate()
    sampled = resample_fragment(contour, candidate, config.samples, window=window)
    length = sampled["observed_length_px"]
    xy, tangent, normal = sampled["points_xy"], sampled["tangent_xy"], sampled["inward_normal_xy"]
    curvature = sampled["curvature_per_px"]
    probabilities = np.column_stack((sampled["role_probabilities"], sampled["transition_probabilities"]))
    corner, thickness = sampled["corner_strength"], sampled["inward_thickness_px"]
    if reverse:
        xy, tangent, normal = xy[::-1], -tangent[::-1], normal[::-1]
        curvature, probabilities = -curvature[::-1], probabilities[::-1]
        corner, thickness = corner[::-1], thickness[::-1]
    first = tangent[0]
    basis = np.array([first, [-first[1], first[0]]])
    geometry = np.column_stack(((xy - xy.mean(axis=0)) @ basis.T / length,
                                tangent @ basis.T, normal @ basis.T,
                                curvature * length, corner * length, thickness / length))
    result = {"geometry": geometry.astype(np.float32), "probabilities": probabilities.astype(np.float32),
              "points_xy_crop": xy.astype(np.float32), "tangent_xy_crop": tangent.astype(np.float32),
              "inward_normal_xy_crop": normal.astype(np.float32),
              "observed_length_px": np.array([length], dtype=np.float32),
              "sample_arc_fraction": np.linspace(0, 1, config.samples, dtype=np.float32)}
    if window is not None:
        result["parent_observed_length_px"] = np.array([sampled["parent_observed_length_px"]], dtype=np.float32)
        result["window_arc_range_px"] = sampled["window_arc_range_px"].astype(np.float32)
    if appearance:
        rgb, mask = pixels if pixels is not None else masked_pixels(contour)
        offsets = np.arange(config.strip_depth, dtype=np.float32)
        coords = xy[:, None, :] + normal[:, None, :] * offsets[None, :, None]
        colors = np.clip(bilinear(rgb, coords), 0, 1)
        valid = np.clip(bilinear(mask, coords), 0, 1)
        # Stop after leaving the first inward paper run; a remote lobe is not
        # neighboring texture. Offset zero retains its subpixel boundary mask.
        continuous = np.ones(valid.shape, dtype=bool)
        continuous[:, 1:] = np.logical_and.accumulate(valid[:, 1:] >= 0.5, axis=1)
        colors *= continuous[..., None]
        valid *= continuous
        result["strip"] = np.concatenate((colors, valid[..., None]), axis=-1).astype(np.float32)
        if config.context_size:
            result["context"] = _context(rgb, mask, config.context_size)
    require(all(np.isfinite(v).all() for v in result.values()), "Nonfinite observable features")
    return result
