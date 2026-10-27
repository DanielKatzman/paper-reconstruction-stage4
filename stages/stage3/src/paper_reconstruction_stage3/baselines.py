"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict, dataclass
import time
import numpy as np
from .contracts import require
from .data import load_contour
from .features import FeatureConfig, fragment_features
from .windows import WindowConfig, variants

@dataclass(frozen=True)
class BaselineConfig:
    variants: tuple = ("global_128", "global_256", "global_512", "window_256_128",
                       "window_256_256", "window_128_128", "window_64_128")
    descriptor_modes: tuple = ("shape", "shape_prob")
    alignment_variants: tuple = ("global_512", "window_256_256")
    ks: tuple = (1, 5, 10, 20, 50, 100)
    descriptor_bins: int = 8
    search_block: int = 128
    cache_mib: int = 256
    feature_cache_entries: int = 32
    rerank_pool: int = 20
    export_topk: int = 5
    alignment_samples: int = 16
    alignment_fractions: tuple = (1.0, .5, .25, .125)
    minimum_fit_length_px: float = 12.0
    minimum_scale: float = .5
    maximum_scale: float = 2.0
    collision_grid: int = 24
    exhaustive_puzzles: int = 3
    exhaustive_max_candidates: int = 96
    pr_bins: int = 1024
    score_thresholds: tuple = (.2, .4, .6, .8, .9)

    def validate(self, windows=WindowConfig()):
        names = variants(windows)
        for field in ("variants", "descriptor_modes", "ks", "alignment_fractions", "score_thresholds"):
            values = getattr(self, field)
            require(isinstance(values, (list, tuple)) and values and len(set(values)) == len(values), "Invalid/duplicate baseline " + field)
        require(set(self.variants).issubset(names), "Unknown representation variant")
        require(set(self.descriptor_modes).issubset({"shape", "shape_prob"}), "Unknown descriptor mode")
        require(set(self.alignment_variants).issubset(self.variants) and "shape" in self.descriptor_modes,
                "Alignment variants require the shape descriptor and an evaluated variant")
        for name, lo, hi in (("descriptor_bins", 4, 32), ("search_block", 1, 512), ("cache_mib", 0, 1048576),
                            ("feature_cache_entries", 1, 256), ("rerank_pool", 1, 100), ("export_topk", 0, 100),
                            ("alignment_samples", 8, 64), ("collision_grid", 0, 64), ("exhaustive_puzzles", 0, 100),
                            ("exhaustive_max_candidates", 2, 256), ("pr_bins", 32, 4096)):
            require(type(getattr(self, name)) is int and lo <= getattr(self, name) <= hi, "Invalid baseline " + name)
        require(all(type(k) is int and 1 <= k <= 1000 for k in self.ks), "Invalid K budgets")
        require(all(isinstance(x, (float, int)) and not isinstance(x, bool) and np.isfinite(x) and 0 < x <= 1
                    for x in self.alignment_fractions), "Invalid fit fractions")
        require(all(isinstance(x, (float, int)) and not isinstance(x, bool) and np.isfinite(x) and 0 <= x <= 1
                    for x in self.score_thresholds), "Invalid score thresholds")
        require(all(isinstance(x, (float, int)) and not isinstance(x, bool) and np.isfinite(x) and x > 0
                    for x in (self.minimum_fit_length_px, self.minimum_scale, self.maximum_scale))
                and self.minimum_scale <= 1 <= self.maximum_scale, "Invalid observable fit length/scale bounds")

    def manifest(self):
        return {"schema_version": "stage3.baseline-config.v1", **asdict(self)}


def descriptor(features, bins=8, *, reverse=False, probabilities=False):
    """First-tangent frame, normalized shape/turning, optional predicted roles."""
    xy = features["points_xy_crop"].astype(np.float64)
    tangent = features["tangent_xy_crop"].astype(np.float64)
    curvature = features["geometry"][:, 6].astype(np.float64)
    roles = features["probabilities"].astype(np.float64)
    if reverse:
        xy, tangent, curvature, roles = xy[::-1], -tangent[::-1], -curvature[::-1], roles[::-1]
    frame = np.array([tangent[0], [-tangent[0, 1], tangent[0, 0]]])
    length = float(features["observed_length_px"][0])
    positions = (xy - xy.mean(axis=0)) @ frame.T / length
    directions = tangent @ frame.T
    grid = np.linspace(0, len(xy) - 1, bins)
    def sample(a):
        return np.column_stack([np.interp(grid, np.arange(len(a)), a[:, i]) for i in range(a.shape[1])]).ravel()
    result = np.concatenate((3 * sample(positions), .5 * sample(directions),
                             .25 * np.interp(grid, np.arange(len(xy)), np.tanh(curvature / np.pi)),
                             [np.linalg.norm(xy[-1] - xy[0]) / length]))
    if probabilities:
        result = np.concatenate((result, .25 * sample(roles)))
    require(np.isfinite(result).all(), "Nonfinite analytic descriptor")
    return result.astype(np.float32)


class ObservableInputs:
    """One gallery in RAM; bounded features, no RGB strips or labels."""
    def __init__(self, gallery, config):
        self.gallery, self.config = gallery, config
        self.contours, self.feature_cache, self.masks = {}, OrderedDict(), {}
        self.feature_constructions = 0
        self.subarc_cache, self.mask_sample_cache = OrderedDict(), {}
        self.alignment_preparation_seconds = self.fit_seconds = 0.
        for _, payload in gallery:
            e = payload["entry"]
            if e["observation_key"] not in self.contours:
                self.contours[e["observation_key"]] = load_contour({**e, "candidates": []})

    def features(self, owner, profile, span=None):
        identity = (owner, profile["samples"], None if span is None else (span["start_px"], span["end_px"]))
        if identity not in self.feature_cache:
            _, payload = self.gallery[owner]
            contour = self.contours[payload["entry"]["observation_key"]]
            self.feature_cache[identity] = fragment_features(contour, payload["candidate"],
                FeatureConfig(profile["samples"]), appearance=False, window=span)
            self.feature_constructions += 1
            while len(self.feature_cache) > self.config.feature_cache_entries:
                self.feature_cache.popitem(last=False)
        self.feature_cache.move_to_end(identity)
        return self.feature_cache[identity]

    def subarcs(self, owner, profile, span, reverse=False):
        from .alignment import _subarcs
        identity = (owner, profile["samples"], None if span is None else (span["start_px"], span["end_px"]), reverse)
        if identity not in self.subarc_cache:
            tick = time.perf_counter()
            self.subarc_cache[identity] = _subarcs(self.features(owner, profile, span), self.config, reverse)
            self.alignment_preparation_seconds += time.perf_counter() - tick
            while len(self.subarc_cache) > 2 * self.config.feature_cache_entries: self.subarc_cache.popitem(last=False)
        self.subarc_cache.move_to_end(identity)
        return self.subarc_cache[identity]

    def mask_samples(self, owner):
        from .alignment import mask_samples
        key = self.gallery[owner][1]["entry"]["observation_key"]
        if key not in self.mask_sample_cache:
            tick = time.perf_counter()
            self.mask_sample_cache[key] = mask_samples(self.mask(owner), self.config.collision_grid)
            self.alignment_preparation_seconds += time.perf_counter() - tick
        return self.mask_sample_cache[key]

    def mask(self, owner):
        from PIL import Image
        e = self.gallery[owner][1]["entry"]
        key = e["observation_key"]
        if key not in self.masks:
            require(e.get("mask_path"), "Observable mask required for collision sampling")
            with Image.open(e["mask_path"]) as image:
                self.masks[key] = np.asarray(image.convert("L"), dtype=np.uint8)
        return self.masks[key]
