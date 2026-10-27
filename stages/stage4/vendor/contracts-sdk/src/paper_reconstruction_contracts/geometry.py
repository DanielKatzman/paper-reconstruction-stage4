"""Native homogeneous coordinates for independent Stage 4 implementations."""
import numpy as np
from .common import normalized_matrix, require


def map_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all(), 'Finite N x 2 native points required')
    h = normalized_matrix(matrix)
    mapped = np.column_stack((points, np.ones(len(points)))) @ h.T
    require((np.abs(mapped[:, 2]) > 1e-12).all(), 'Projective pole at native point')
    return mapped[:, :2]/mapped[:, 2:]


def compose(second_to_first, third_to_second):
    return normalized_matrix(normalized_matrix(second_to_first) @ normalized_matrix(third_to_second))
