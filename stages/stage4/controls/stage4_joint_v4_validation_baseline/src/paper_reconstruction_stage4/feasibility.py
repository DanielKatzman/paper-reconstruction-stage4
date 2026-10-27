from collections import OrderedDict
import numpy as np
from scipy.ndimage import binary_erosion
from paper_reconstruction_contracts import map_points
from .geometry import normalize, probes, relative, warp_check


class Feasibility:
    def __init__(self, problem, config):
        self.problem, self.config = problem, config
        self.masks, self.samples = {}, {}
        self.cache = OrderedDict()
        self.cache_hits = 0
        for pid, piece in problem.pieces.items():
            eroded = binary_erosion(piece.mask, iterations=config.edge_erosion_px) if config.edge_erosion_px else piece.mask
            # A thin physical piece must not disappear from collision tests.
            self.masks[pid] = eroded if eroded.any() else piece.mask
            y, x = np.nonzero(self.masks[pid])
            points = np.column_stack((x, y)).astype(float)
            limit = config.collision_grid ** 2
            self.samples[pid] = points[np.linspace(0, len(points) - 1, min(limit, len(points))).astype(int)]

    def _direction(self, source, target, transform, final):
        mask = self.masks[source]
        if final:
            yy, xx = np.nonzero(mask)
            count = len(xx)
            stride = max(1, int(np.ceil(count / self.config.final_collision_samples)))
            points = np.column_stack((xx[::stride], yy[::stride])).astype(float)
        else:
            points, stride = self.samples[source], None
            count = int(mask.sum())
        mapped = np.rint(map_points(points, transform))
        # Reject pathological coordinates before converting them to integers.
        if not np.isfinite(mapped).all() or np.max(np.abs(mapped), initial=0) > 1e12:
            raise ValueError("Unbounded mask mapping")
        x, y = mapped.astype(np.int64).T
        other = self.masks[target]
        valid = (x >= 0) & (y >= 0) & (x < other.shape[1]) & (y < other.shape[0])
        hits = np.zeros(len(points), bool)
        hits[valid] = other[y[valid], x[valid]]
        return float(hits.mean()), {"samples": len(points), "interior_pixels": count, "stride": stride}

    def pair(self, a, b, h, final=False):
        h = normalize(h)
        # Exact transform bytes avoid rounded-key reuse near a collision threshold.
        key = (a, b, h.tobytes(), final)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        warp = warp_check(h, self.problem.pieces[b].shape, self.problem.pieces[a].shape, self.config)
        if not warp["valid"]:
            return {"valid": False, "reason": warp["reason"], "warp": warp}
        first_box = probes(self.problem.pieces[a].shape, 2)
        other_box = map_points(probes(self.problem.pieces[b].shape, 2), h)
        separate = np.any(first_box.max(axis=0) < other_box.min(axis=0)) or np.any(other_box.max(axis=0) < first_box.min(axis=0))
        if separate:
            result = {"valid": True, "reason": "separate_crop_boxes", "overlap": 0., "warp": warp}
        else:
            ba, ba_sampling = self._direction(b, a, h, final)
            ab, ab_sampling = self._direction(a, b, np.linalg.inv(h), final)
            overlap = max(ab, ba)
            result = {"valid": overlap <= self.config.maximum_assembly_collision, "reason": "interior_collision" if overlap > self.config.maximum_assembly_collision else "checked",
                      "overlap": overlap, "directional_overlap": [ab, ba], "sampling": [ab_sampling, ba_sampling], "warp": warp}
        if len(self.cache) >= self.config.maximum_cache_entries:
            self.cache.popitem(last=False)
        self.cache[key] = result
        return result

    def state(self, state, final=False):
        pairs = []
        for component in state.components():
            for index, a in enumerate(component):
                # Validate even the gauge root, including isolated pieces.
                check = warp_check(state.poses[a], self.problem.pieces[a].shape,
                                   self.problem.pieces[min(component)].shape, self.config)
                if not check["valid"]:
                    return {"valid": False, "reason": check["reason"], "piece": a}
                for b in component[index + 1:]:
                    report = self.pair(a, b, relative(state.poses, a, b), final)
                    pairs.append({"a": a, "b": b, **report})
                    if not report["valid"]:
                        return {"valid": False, "reason": report["reason"], "pairs": pairs}
        return {"valid": True, "reason": "checked", "pairs": pairs,
                "fidelity": "bounded_native_interior_sampling" if final else "coarse_native_interior_sampling",
                "edge_erosion_px": self.config.edge_erosion_px, "maximum_overlap": max((p["overlap"] for p in pairs), default=0.)}
