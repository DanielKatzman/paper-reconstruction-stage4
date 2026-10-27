"""Exact CPU evidence acceleration, candidate v1; no new search/scoring policy.

This file is copied into an isolated runtime as ``exact_acceleration.py``.
The frozen producer remains unchanged until the candidate is qualified.
"""
from collections import OrderedDict
from copy import deepcopy
import numpy as np


def correspondences(a, an, b, bn):
    """Same continuous normal-compatible segment search, in bounded row tiles.

    Two scalar coordinate arrays replace N*M*2 temporaries. Every segment is
    still tested, normals are interpolated/normalized exactly as before, and
    argmin keeps the first segment on ties (including the all-incompatible row).
    No vertex approximation, lower resolution, precision change or spatial cap.
    """
    a, an, b, bn = (np.asarray(x) for x in (a, an, b, bn))
    if len(b) < 2:
        raise ValueError('attempt to get argmin of an empty sequence')
    v = b[1:]-b[:-1]
    den = np.maximum(np.sum(v*v, axis=1), 1e-12)
    distances = []; indices = []; fractions = []; projections = []
    # Keep peak temporary storage bounded and the active tile cache-friendly.
    for start in range(0, len(a), 64):
        aa = a[start:start+64]; nn = an[start:start+64]
        dx = aa[:, 0, None]-b[None, :-1, 0]
        dy = aa[:, 1, None]-b[None, :-1, 1]
        dot = dx*v[None, :, 0]+dy*v[None, :, 1]
        # np.sum over two components starts at +0; retain its signed-zero rule.
        dot = dot+0.
        u = np.clip(dot/den[None], 0., 1.)
        px = b[None, :-1, 0]+u*v[None, :, 0]
        py = b[None, :-1, 1]+u*v[None, :, 1]
        nx = bn[None, :-1, 0]*(1-u)+bn[None, 1:, 0]*u
        ny = bn[None, :-1, 1]*(1-u)+bn[None, 1:, 1]*u
        norm = np.maximum(np.sqrt(nx*nx+ny*ny), 1e-12)
        nx /= norm; ny /= norm
        compatible = (nn[:, 0, None]*nx+nn[:, 1, None]*ny+0.) <= -.5
        cost = (aa[:, 0, None]-px)**2+(aa[:, 1, None]-py)**2
        cost = np.where(compatible, cost, np.inf)
        j = np.argmin(cost, axis=1); i = np.arange(len(aa))
        distances.append(np.sqrt(cost[i, j])); indices.append(j)
        fractions.append(u[i, j]); projections.append(np.column_stack((px[i, j], py[i, j])))
    if not len(a):
        # Match the reference output shape/dtype for the empty-query case.
        dtype = np.result_type(a, b)
        if np.issubdtype(dtype, np.integer):
            dtype = np.dtype(np.float64)  # true division in the reference
        return (np.empty(0, dtype=dtype), np.empty(0, dtype=np.intp),
                np.empty(0, dtype=dtype), np.empty((0, 2), dtype=dtype))
    return tuple(np.concatenate(values) for values in (distances, indices, fractions, projections))


class VisualCache:
    """Bounded cache scoped to ONE immutable physical-pair inference block.

    The owner must keep observations, 512-sample candidate curves, RGB/masks and
    correction/visual configuration fixed throughout the block. The actual
    visual function uses query/target candidates and the full directed matrix;
    it does not read pose support/windows. Broad support evidence is NOT cached.
    """
    def __init__(self, maximum=256):
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError('Visual cache capacity must be a positive integer')
        self.maximum = maximum
        self.values = OrderedDict()
        self.hits = self.misses = 0

    def evaluate(self, q, t, pose, compute):
        matrix = np.asarray(pose['alignment']['transform'], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError('Visual cache requires a finite full directed 3x3 matrix')
        # Do not round, normalize, canonicalize direction or conflate candidates.
        key = (q, t, matrix.tobytes(order='C'))
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return deepcopy(self.values[key])
        self.misses += 1
        result = compute()
        self.values[key] = deepcopy(result)
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return result

    def summary(self):
        return {'version': 'directed-visual-cache.v1', 'hits': self.hits, 'misses': self.misses,
                'entries': len(self.values), 'maximum_entries': self.maximum,
                'scope': 'one physical pair, fixed inputs/configuration; exact directed candidate/matrix keys'}
