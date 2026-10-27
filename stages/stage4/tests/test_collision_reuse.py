from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility


class CollisionReuseTests(unittest.TestCase):
    def test_native_collision_matches_uncached_sampling_at_both_fidelities(self):
        rng = np.random.default_rng(77)
        masks = {"a": rng.random((91, 79)) > .3, "b": rng.random((91, 79)) > .6}
        # Include a very thin fragment, which exercises the erosion fallback.
        masks["thin"] = np.zeros((91, 79), bool)
        masks["thin"][10:80, 35] = True
        problem = SimpleNamespace(pieces={p: SimpleNamespace(mask=m, shape=m.shape) for p, m in masks.items()})
        config = replace(SolverConfig(), edge_erosion_px=2, collision_grid=7, final_collision_samples=37)
        f = Feasibility(problem, config)
        for final in (False, True):
            for source in masks:
                for tx, ty, projective in ((0., 0., 0.), (11.3, -2.7, .0003), (-9.8, 17., -.0002)):
                    h = np.array([[1., .03, tx], [-.02, 1., ty], [projective, 0., 1.]])
                    mask = f.masks[source]
                    yy, xx = np.nonzero(mask)
                    if final:
                        stride = max(1, int(np.ceil(len(xx)/config.final_collision_samples)))
                        points = np.column_stack((xx[::stride], yy[::stride])).astype(float)
                    else:
                        stride = None
                        all_points = np.column_stack((xx, yy)).astype(float)
                        points = all_points[np.linspace(0, len(all_points)-1, min(config.collision_grid**2, len(all_points))).astype(int)]
                    mapped = np.rint(map_points(points, h)).astype(np.int64)
                    x, y = mapped.T
                    other = f.masks["b"]
                    valid = (x >= 0) & (y >= 0) & (x < other.shape[1]) & (y < other.shape[0])
                    hits = np.zeros(len(points), bool)
                    hits[valid] = other[y[valid], x[valid]]
                    actual, metadata = f._direction(source, "b", h, final)
                    self.assertEqual(actual, float(hits.mean()))
                    self.assertEqual(metadata, {"samples": len(points), "interior_pixels": len(xx), "stride": stride})
