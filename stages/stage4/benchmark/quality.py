"""Post-prediction measurements; structural review stays distinct from precision."""
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
from paper_reconstruction_contracts import map_points


def quantiles(values):
    values = np.asarray(values, float)
    return {"p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)), "maximum": float(values.max())} if values.size else {"p50": None, "p95": None, "maximum": None}


def selected_path(joins, a, b):
    adjacency = {}
    for j in joins:
        h = np.asarray(j["source_transform_b_to_a"], float)
        adjacency.setdefault(j["piece_a"], []).append((j["piece_b"], h, j["geometry_id"]))
        adjacency.setdefault(j["piece_b"], []).append((j["piece_a"], np.linalg.inv(h), j["geometry_id"]))
    todo, visited = deque([(a, np.eye(3), [])]), {a}
    while todo:
        current, transform, ids = todo.popleft()
        if current == b:
            return {"geometry_ids": ids, "transform_b_to_a": transform}
        for neighbor, h, eid in sorted(adjacency.get(current, []), key=lambda x: (x[0], x[2])):
            if neighbor not in visited:
                visited.add(neighbor)
                todo.append((neighbor, transform @ h, [*ids, eid]))
    return None


def contact_path_diagnostics(layout, contacts, error_function):
    rows = {}
    joins = layout["selected_joins"]
    direct = {(j["piece_a"], j["piece_b"]): j for j in joins}
    for c in contacts:
        pair = (c["piece_a"], c["piece_b"])
        path = selected_path(joins, *pair)
        join = direct.get(pair)
        rows[pair] = {"explicitly_selected": join is not None,
                      "selected_direct_source_p95_px": error_function(join["source_transform_b_to_a"], c) if join else None,
                      "selected_path_geometry_ids": path["geometry_ids"] if path else None,
                      "selected_path_source_p95_px": error_function(path["transform_b_to_a"], c) if path else None,
                      "counterfactual_support_compatibility": None,
                      "counterfactual_support_reason": "Not tested: independent accurate direct poses can have incompatible supports or global constraints."}
    return rows


def assembly_quality(result, reference, evaluation):
    layout = result["layouts"][0]
    pieces = {p["piece_id"]: p for p in layout["pieces"]}
    expected = {(c["piece_a"], c["piece_b"]) for c in reference["contacts"]}
    selected = {(j["piece_a"], j["piece_b"]) for j in layout["selected_joins"]}
    component = min(layout["components"], key=lambda c: (-len(c), min(c)))
    anchor = min(component)
    poses = {p: np.asarray(v["native_to_component"], float) for p, v in pieces.items()}
    ref = {p: np.asarray(v["native_to_reference"], float) for p, v in reference["pieces"].items()}
    align = ref[anchor] @ np.linalg.inv(poses[anchor])
    diagonal = float(np.linalg.norm(reference["reference_size_hw"])) if reference.get("reference_size_hw") else None
    rows, displacements = [], []
    for pid, piece in sorted(pieces.items()):
        if pid not in component:
            rows.append({"piece_id": pid, "p95_reference_px": None, "reason": "Separate component: no measured inter-component pose"})
            continue
        probes = np.asarray(piece["native_probes"], float)
        pred = align @ poses[pid]
        try:
            errors = np.linalg.norm(map_points(probes, pred)-map_points(probes, ref[pid]), axis=1)
            center = probes.mean(axis=0)
            basis = np.array([center, center+[1., 0.]])
            u = np.diff(map_points(basis, pred), axis=0)[0]
            v = np.diff(map_points(basis, ref[pid]), axis=0)[0]
            if not np.isfinite(errors).all() or np.linalg.norm(v) == 0:
                raise ValueError("Invalid common-gauge warp")
            angle = float(np.degrees(np.arctan2(v[0]*u[1]-v[1]*u[0], np.dot(u, v))))
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"piece_id": pid, "p95_reference_px": None, "reason": str(exc)})
            continue
        rows.append({"piece_id": pid, "anchor_piece": pid == anchor, "p95_reference_px": float(np.quantile(errors, .95)),
                     "p95_fraction_reference_diagonal": float(np.quantile(errors, .95))/diagonal if diagonal else None,
                     "local_orientation_error_degrees": angle, "local_scale_ratio": float(np.linalg.norm(u)/np.linalg.norm(v))})
        if pid != anchor:
            displacements.extend(errors.tolist())
    curve = [{"tolerance_native_a_px": t, "passing_contacts": sum(c["p95_native_a_px"] is not None and c["p95_native_a_px"] <= t for c in evaluation["contacts"]),
              "total_contacts": len(evaluation["contacts"]), "diagnostic_only": True} for t in (4, 8, 16, 32, 48)]
    return {"structural_visual_assessment": None, "structural_visual_reason": "Requires review of the primary preview, orientations, gaps, overlap and drift; no new pass threshold has been chosen.",
            "physical_inventory_count": len(pieces), "inventory_coverage": len(pieces)/len(reference["pieces"]),
            "component_sizes": sorted((len(c) for c in layout["components"]), reverse=True),
            "all_pieces_connected": layout["all_pieces_connected"], "selected_neighbor_identity_precision": len(selected & expected)/len(selected) if selected else None,
            "selected_true_neighbor_pairs": len(selected & expected), "selected_physical_pairs": len(selected),
            "neighbor_identity_warning": "Neighbor identity and connectivity alone do not establish correct orientation/placement. A selected spanning tree need not explicitly contain every true contact.",
            "common_gauge": {"anchor_piece": anchor, "scope_piece_ids": component, "alignment": "One shared transform using the smallest ID in the largest component; no per-piece fitting.",
                             "other_components_aligned": False, "non_anchor_probe_errors_reference_px": quantiles(displacements),
                             "whole_page_error_available": len(component) == len(pieces), "per_piece": rows},
            "contact_p95_error_distribution_native_a_px": quantiles([c["p95_native_a_px"] for c in evaluation["contacts"] if c["p95_native_a_px"] is not None]),
            "unresolved_contacts": sum(c["p95_native_a_px"] is None for c in evaluation["contacts"]),
            "contact_tolerance_sensitivity": curve, "primary_layout_index": 0}


def placement_overlay(result, reference, inputs, stage3, target):
    """Evaluation illustration: one shared gauge for the largest component only."""
    from .common import read_json
    source = inputs / "evaluation_only/reference_page.jpg"
    if not source.is_file():
        return None
    with Image.open(source) as img:
        canvas = img.convert("RGB")
    canvas = Image.blend(canvas, Image.new("RGB", canvas.size, "white"), .65)
    sy, sx = canvas.height/reference["reference_size_hw"][0], canvas.width/reference["reference_size_hw"][1]
    draw = ImageDraw.Draw(canvas)
    for seam in reference["seams"]:
        draw.line([(x*sx, y*sy) for x, y in seam["points_reference"]], fill="#dc5966", width=1)
    layout = result["layouts"][0]
    pieces = {p["piece_id"]: p for p in layout["pieces"]}
    component = min(layout["components"], key=lambda c: (-len(c), min(c)))
    anchor = min(component)
    align = np.asarray(reference["pieces"][anchor]["native_to_reference"]) @ np.linalg.inv(pieces[anchor]["native_to_component"])
    manifest = read_json(stage3 / "bundle.json")
    clipped, omitted = [], []
    for pid in component:
        contour = (stage3 / manifest["pieces"][pid]["contour"]).resolve()
        if not contour.is_relative_to(stage3.resolve()):
            raise ValueError("Contour path escaped the verified evidence bundle")
        with np.load(contour, allow_pickle=False) as arrays:
            points = arrays["points_xy"].copy()
        try:
            points = map_points(points, align @ np.asarray(pieces[pid]["native_to_component"]))
        except ValueError:
            omitted.append(pid)
            continue
        xy = [(float(x*sx), float(y*sy)) for x, y in points]
        bounded = np.clip(xy, [-8*canvas.width, -8*canvas.height], [9*canvas.width, 9*canvas.height])
        if not np.array_equal(bounded, xy):
            clipped.append(pid)
        xy = [tuple(p) for p in bounded]
        draw.line([*xy, xy[0]], fill="#007eac", width=2)
        draw.text(tuple(np.mean(xy, axis=0)), pid, fill="#00375b", stroke_width=1, stroke_fill="white")
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return {"path": target.name, "scope_piece_ids": component, "anchor_piece": anchor, "clipped_for_display_piece_ids": clipped, "omitted_invalid_reference_warp_piece_ids": omitted,
            "caption": "Evaluation only: cyan predicted contours, red known cuts, faded source page. One shared anchor; separate components are omitted."}
