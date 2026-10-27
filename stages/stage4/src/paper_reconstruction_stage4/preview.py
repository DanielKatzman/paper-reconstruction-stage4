"""Bounded raster preview; component packing never changes reconstruction poses."""
import math
import numpy as np
from PIL import Image, ImageDraw
from paper_reconstruction_contracts import map_points
from .geometry import normalize, probes


def render(problem, state, path, max_side=1600):
    components = state.components()
    bounds = []
    for component in components:
        corners = np.vstack([map_points(probes(problem.pieces[p].shape, 2), state.poses[p]) for p in component])
        bounds.append((corners.min(axis=0), corners.max(axis=0)))
    widths = [float(high[0] - low[0] + 1) for low, high in bounds]
    heights = [float(high[1] - low[1] + 1) for low, high in bounds]
    total_area = sum(w * h for w, h in zip(widths, heights))
    scale = min(6., (max_side - 120) / max(max(widths), max(heights), math.sqrt(max(total_area, 1)) * 1.5))
    # Pack at display resolution, then downscale once more if needed.
    row_width = max(600., math.sqrt(max(total_area, 1)) * scale * 1.5)
    packed, x, y, row_h = {}, 20., 45., 0.
    for component, (low, high), width, height in zip(components, bounds, widths, heights, strict=True):
        w, h = width * scale, height * scale
        if x > 20 and x + w + 20 > row_width:
            x, y, row_h = 20., y + row_h + 45., 0.
        packed[min(component)] = np.array([[scale, 0., x - scale * low[0]], [0., scale, y - scale * low[1]], [0., 0., 1.]])
        x += w + 35.
        row_h = max(row_h, h)
    width = int(max(row_width + 40, x + 20))
    height = int(y + row_h + 50)
    shrink = min(1., max_side / max(width, height))
    display_scale = np.diag([shrink, shrink, 1.])
    packed = {p: display_scale @ h for p, h in packed.items()}
    canvas = Image.new("RGBA", (max(1, int(width * shrink)), max(1, int(height * shrink))), "#e8edf2")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{len(state.poses)} pieces / {len(components)} components - observable candidate", fill="black")
    labels = []
    for pid, piece in sorted(problem.pieces.items()):
        display = packed[state.groups[pid]] @ state.poses[pid]
        inverse = normalize(np.linalg.inv(display))
        with Image.open(piece.image_path) as original:
            layer = original.convert("RGBA")
        layer.putalpha(Image.fromarray(piece.mask.astype(np.uint8) * 255))
        coefficients = tuple(inverse.ravel()[:8])
        warped = layer.transform(canvas.size, Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC)
        canvas.alpha_composite(warped)
        center = map_points(np.array([[piece.shape[1] / 2, piece.shape[0] / 2]]), display)[0]
        labels.append((tuple(center), pid))
    draw = ImageDraw.Draw(canvas)
    from .support import occurrence_arcs
    edges = {e.id: e for e in problem.edges}
    for eid, oid in sorted(state.assignments.items()):
        occurrence = next(o for o in edges[eid].occurrences if o["occurrence_id"] == oid)
        for arc in occurrence_arcs(problem, occurrence):
            display = packed[state.groups[arc.piece]] @ state.poses[arc.piece]
            points = map_points(arc.points, display)
            draw.line([tuple(point) for point in points], fill="#999999" if eid in state.active_support else "#14804a", width=1 if eid in state.active_support else 2)
        for arc in state.active_support.get(eid, {}).get("arcs", []):
            display = packed[state.groups[arc["piece"]]] @ state.poses[arc["piece"]]
            points = map_points(arc["points"], display)
            draw.line([tuple(point) for point in points], fill="#14804a", width=2)
    for center, label in labels:
        draw.text(center, label[:35], fill="#cc2030", stroke_width=1, stroke_fill="white")
    canvas.convert("RGB").save(path)
    return {root: h.tolist() for root, h in packed.items()}
