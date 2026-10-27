"""Replayable exterior-edge evidence for a sliding native graph bridge.

The uncertain side cannot define its own rectangle. All four lines must have
measured, unoccupied support on the other side of the graph cut. Two orthogonal
exterior arcs on the uncertain side then constrain its remaining sliding mode.
No format, DPI, reference placement, or learned score enters this measurement.
"""
import numpy as np
from paper_reconstruction_contracts import map_points
from .native_bridge import available, runs, arc_from_indices, arc_value
from .outputs import digest
from .page import enclosing_rectangle
from .constraints import require_rigid

POLICY = 'observed-page-boundary-cut-witness.v1'


def measure(problem, poses, cut, occupied, config, axes=None):
    ids = set(problem.pieces); cut = set(cut); other = ids-cut
    if set(poses) != ids or not cut or not other:
        raise ValueError('Page boundary witness requires the complete input inventory')
    if len(cut) > len(other):
        cut, other = other, cut
    for h in poses.values():
        require_rigid(h)
    clouds = {p: map_points(problem.pieces[p].contour['points_xy'], poses[p]) for p in sorted(ids)}
    if axes is None:
        axes = enclosing_rectangle(np.vstack(list(clouds.values())))[1]
    axes = np.asarray(axes, float)
    if axes.shape != (2, 2) or not np.isfinite(axes).all() or not np.allclose(axes.T@axes, np.eye(2), atol=1e-7) or np.linalg.det(axes) < .999999:
        raise ValueError('Invalid page-boundary coordinate axes')
    projected = {p: clouds[p]@axes for p in sorted(ids)}
    body = np.vstack([projected[p] for p in sorted(other)])
    lo, hi = body.min(axis=0), body.max(axis=0)
    tolerance = config.contact_tolerance_px
    if np.min(hi-lo) <= 2*tolerance:
        raise ValueError('Degenerate independent page frame')
    # The cut may not expand the independently observed rectangle.
    leaf = np.vstack([projected[p] for p in sorted(cut)])
    if np.any(leaf.min(axis=0) < lo-tolerance) or np.any(leaf.max(axis=0) > hi+tolerance):
        raise ValueError('Uncertain component escapes the independent page frame')
    rows = []
    for side in range(4):
        dimension, high = divmod(side, 2)
        line = hi[dimension] if high else lo[dimension]
        for p in sorted(ids):
            contour = problem.pieces[p].contour
            normals = np.asarray(contour['inward_normal_xy'])@poses[p][:2, :2].T@axes
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            good = (np.abs(projected[p][:, dimension]-line) <= tolerance)
            good &= (normals[:, dimension]*(-1. if high else 1.)) >= .9
            good &= available(problem, p, occupied)
            for first, last in runs(good):
                try:
                    arc = arc_from_indices(problem, p, first, last)
                except ValueError:
                    continue
                if arc.length < 96.:
                    continue
                # All replayed arc samples, not just contour endpoints, must
                # support the same real line and a substantial straight span.
                xy = map_points(arc.points, poses[p])@axes
                if np.max(np.abs(xy[:, dimension]-line)) > tolerance or np.ptp(xy[:, 1-dimension]) < 96.:
                    continue
                rows.append({'side': side, 'arc': arc_value(arc), 'cut_side': p in cut})
    body_sides = {r['side'] for r in rows if not r['cut_side']}
    cut_sides = {r['side'] for r in rows if r['cut_side']}
    if body_sides != set(range(4)):
        raise ValueError('Other pieces do not independently support all four page edges')
    if not any(a//2 != b//2 for a in cut_sides for b in cut_sides):
        raise ValueError('Uncertain component lacks two orthogonal exterior-edge witnesses')
    value = {'policy': POLICY, 'input_manifest_sha256': problem.manifest['manifest_sha256'],
        'cut': sorted(cut), 'other_side': sorted(other), 'axes': axes.tolist(),
        'body_line_low': lo.tolist(), 'body_line_high': hi.tolist(),
        'contact_tolerance_px': tolerance, 'minimum_external_span_px': 96.,
        'observed_external_arcs': rows, 'page_format_or_dpi_used': False}
    return {**value, 'witness_id': digest(value)}


def verify(problem, poses, cut, occupied, config, witness):
    if witness.get('witness_id') != digest({k: v for k, v in witness.items() if k != 'witness_id'}):
        raise ValueError('Changed page-boundary witness identity')
    expected = measure(problem, poses, cut, occupied, config, witness.get('axes'))
    if witness != expected:
        raise ValueError('Page-boundary witness differs from independent native replay')
    return True
