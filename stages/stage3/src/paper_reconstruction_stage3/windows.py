"""Observable open-arc windows. This module has no label or truth dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .contracts import digest, require
from .features import resampling_grid

SAMPLES = (128, 256, 512)


@dataclass(frozen=True)
class WindowConfig:
    lengths_px: tuple = (64.0, 128.0, 256.0)
    stride_fraction: float = 0.5

    def validate(self):
        require(1 <= len(self.lengths_px) <= 8 and len(set(self.lengths_px)) == len(self.lengths_px),
                "Choose one to eight distinct observable window lengths")
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and x > 0 for x in self.lengths_px),
                "Window lengths must be finite positive observed pixel lengths")
        require(len({f"{x:g}" for x in self.lengths_px}) == len(self.lengths_px),
                "Window lengths must have distinct variant names; choose clearly separated scales")
        require(isinstance(self.stride_fraction, (int, float)) and not isinstance(self.stride_fraction, bool)
                and np.isfinite(self.stride_fraction) and 0 < self.stride_fraction <= 1,
                "Window stride fraction must be in (0,1]")

    def manifest(self):
        self.validate()
        return {"schema_version": "stage3.observable-windows.v1", **asdict(self),
                "coordinates": "Observed open-candidate arc length in crop pixels",
                "selection": "Fixed lengths and stride, with a final tail window; no truth or labels",
                "short_candidate": "One unpadded full-span window per requested length",
                "parent_hypotheses": "Preserved separately; no parent/split merging",
                "model_feature_metadata": ["parent_observed_length_px", "window_arc_range_px"]}


def windows_for_length(parent_length, length_px, stride_fraction=0.5):
    WindowConfig((length_px,), stride_fraction).validate()
    require(np.isfinite(parent_length) and parent_length > 1e-6, "Candidate has zero observed length")
    length = min(float(parent_length), float(length_px))
    tail = float(parent_length) - length
    step = length * stride_fraction
    count = int(math.floor(tail / step)) + 1
    starts = [i * step for i in range(count)]
    if tail - starts[-1] > 1e-8:
        starts.append(tail)
    else:
        starts[-1] = tail
    return [{"window_id": digest({"start_px": start, "end_px": start + length,
                                   "requested_length_px": float(length_px)})[:20],
             "start_px": start, "end_px": start + length, "requested_length_px": float(length_px)}
            for start in starts]


def observable_windows(contour, candidate, *, length_px=64.0, stride_fraction=0.5):
    points = contour.fragment(candidate)["points_xy"]
    length = float(resampling_grid(points, 128)[0][-1])
    return windows_for_length(length, length_px, stride_fraction)


def variants(config):
    config.validate()
    result = {f"global_{n}": {"samples": n, "length_px": None} for n in SAMPLES}
    for length in config.lengths_px:
        for n in SAMPLES:
            result[f"window_{length:g}_{n}"] = {"samples": n, "length_px": float(length)}
    return result


def feature_bytes(samples, strip_depth, context_size=0, *, window=False):
    """Exact uncompressed numeric array bytes; excludes NPZ/JSON/container overhead."""
    geometry = (samples * 20 + 1 + (3 if window else 0)) * 4
    appearance = samples * strip_depth * 4 * 4 + context_size * context_size * 4 * 4
    return {"geometry_only": geometry, "geometry_and_appearance": geometry + appearance}
