"""Explicit observable toy layouts; contains no Stage 3 runtime dependency."""
import gzip
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from paper_reconstruction_contracts.common import digest
from paper_reconstruction_stage4.outputs import atomic_json, canonical, file_hash


def translation(x, y=0):
    return np.array([[1., 0., float(x)], [0., 1., float(y)], [0., 0., 1.]])


def rectangle_contour():
    xy = np.array([(x, 2) for x in range(2, 30)] + [(29, y) for y in range(3, 30)] +
                  [(x, 29) for x in range(28, 1, -1)] + [(2, y) for y in range(28, 2, -1)], dtype=float)
    delta = np.roll(xy, -1, axis=0) - xy
    tangent = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    return {"points_xy": xy, "arc_length_px": np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))],
            "tangent_xy": tangent, "inward_normal_xy": np.column_stack((-tangent[:, 1], tangent[:, 0])),
            "curvature_per_px": np.zeros(len(xy)), "corner_strength": np.zeros(len(xy)),
            "inward_thickness_px": np.ones(len(xy)) * 10,
            "role_probabilities": np.tile([1., 0., 0.], (len(xy), 1)), "transition_probabilities": np.zeros(len(xy))}


def build_fixture(root, rows=2, cols=2, isolated=False, weight=.9, drift=0., alternatives=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)
    ids = [f"piece_{i:02}" for i in range(rows * cols)]
    if isolated:
        ids.append("piece_isolated")
    pieces, assets, candidates, poses = {}, {}, [], {}
    contour = rectangle_contour()
    xy = contour["points_xy"]
    sides = {"top": (0, 27, False), "right": (27, 54, False), "bottom": (54, 81, False), "left": (81, 0, True)}
    candidate_lookup = {}

    def seal(payload, suffix):
        import hashlib
        name = "assets/" + hashlib.sha256(payload).hexdigest() + suffix
        (root / name).write_bytes(payload)
        assets[name] = {"path": name, "size": len(payload), "sha256": file_hash(root / name)}
        return name

    for i, pid in enumerate(ids):
        mask = np.zeros((32, 32), np.uint8)
        mask[2:30, 2:30] = 255
        image = Image.new("RGB", (32, 32), (230, 220 - i * 8, 200))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_path = seal(buf.getvalue(), ".png")
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, format="PNG")
        mask_path = seal(buf.getvalue(), ".png")
        buf = io.BytesIO()
        np.savez_compressed(buf, **contour)
        contour_path = seal(buf.getvalue(), ".npz")
        pieces[pid] = {"piece_id": pid, "observation_key": pid + "/view0", "size_hw": [32, 32],
                       "image": image_path, "mask": mask_path, "contour": contour_path}
        poses[pid] = translation(27 * (i % cols), 27 * (i // cols)).tolist()
        if pid == "piece_isolated":
            continue
        for side, (start, end, wraps) in sides.items():
            key = pid + "/" + side
            candidate_lookup[pid, side] = (key, len(candidates))
            candidates.append({"candidate_key": key, "piece_id": pid, "observation_key": pid + "/view0",
                               "candidate": {"candidate_id": side, "start_index": start, "end_index": end, "wraps": wraps,
                                             "length_points": (end - start) % len(xy) + 1}})
    groups, edges, contacts = [], [], []
    row_index = 0
    for i, a in enumerate(ids[:rows * cols]):
        r, c = divmod(i, cols)
        neighbors = []
        if c + 1 < cols:
            neighbors.append((ids[i + 1], "right", "left", translation(27)))
        if r + 1 < rows:
            neighbors.append((ids[i + cols], "bottom", "top", translation(0, 27)))
        for b, side_a, side_b, matrix in neighbors:
            contact_points = xy[[81, 90, 100, 0]] if side_b == "left" else xy[[0, 8, 18, 27]]
            contacts.append({"piece_a": a, "piece_b": b, "points_b": contact_points.tolist()})
            matrices = [matrix.copy()]
            if drift and not groups:
                matrices[0][0, 2] += drift
            if alternatives:
                wrong = matrix.copy()
                wrong[0, 2] += 80
                matrices.append(wrong)
            for transform in matrices:
                key_a, index_a = candidate_lookup[a, side_a]
                key_b, index_b = candidate_lookup[b, side_b]
                hid = digest([a, b, pieces[a]["observation_key"], pieces[b]["observation_key"], transform.tolist()])
                oid = digest(["fixture_observable", row_index, row_index])
                pose = {"query": index_a, "target": index_b, "windows": [None, None], "span_a": None, "span_b": None,
                        "alignment": {"transform": transform.tolist(), "family": "similarity", "geometry_score": weight,
                                      "query_arc_fraction": [0., 1.], "target_arc_fraction": [0., 1.]}}
                occurrence = {"occurrence_id": oid, "variant": "fixture_observable", "row_index": row_index, "record_id": row_index,
                              "query_key": key_a, "target_key": key_b, "query_observation": pieces[a]["observation_key"],
                              "target_observation": pieces[b]["observation_key"], "query_piece": a, "target_piece": b,
                              "pose": pose, "original_canonical_transform_b_to_a": transform.tolist()}
                group = {"hypothesis_id": hid, "piece_a": a, "piece_b": b, "observation_a": pieces[a]["observation_key"],
                         "observation_b": pieces[b]["observation_key"], "transform_b_to_a": transform.tolist(), "occurrences": [occurrence]}
                groups.append(group)
                edges.append({k: group[k] for k in group if k != "occurrences"} | {"weight": weight, "score_method": "fixture",
                              "score_variant": "fixture_observable", "representative_occurrence": oid, "occurrence_ids": [oid], "pose": pose})
                row_index += 1
    pair_counts = {}
    for edge in edges:
        pair = (edge["piece_a"], edge["piece_b"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    pool = {"schema_version": "stage3.working-pool.v1", "maximum_per_physical_pair": 24,
            "selection": "explicit observable toy geometry", "scores_are_calibrated_probabilities": False,
            "pairs": [{"pieces": list(pair), "available_geometries": count, "working_geometries": count, "resource_truncated": False}
                      for pair, count in sorted(pair_counts.items())], "edges": edges}
    atomic_json(root / "working_pool.json", pool)
    (root / "hypotheses.jsonl.gz").write_bytes(gzip.compress(b"".join(canonical(group) + b"\n" for group in groups), mtime=0))

    def receipt(name):
        return {"path": name, "size": (root / name).stat().st_size, "sha256": file_hash(root / name)}
    manifest = {"schema_version": "stage3.assembly-evidence.v2", "algorithm": "stage4.explicit-synthetic-fixture",
                "identity": digest([rows, cols, isolated, weight, drift, alternatives]), "puzzle_id": "known_grid_fixture",
                "pieces": pieces, "candidates": candidates, "assets": assets, "archive": receipt("hypotheses.jsonl.gz"),
                "working_pool": receipt("working_pool.json"), "counts": {"pieces": len(pieces), "candidates": len(candidates),
                "geometries": len(groups), "occurrences": len(groups), "working_edges": len(edges)},
                "coordinate_frame": "native crop pixels; x right, y down; B to A",
                "occurrence_contract": "Explicit known-layout fixture with complete native supports",
                "provenance": {"kind": "synthetic_solver_fixture"}, "configuration": {},
                "correction_scorer": {"enabled": False, "reason": "synthetic fixture"}, "training_performed": False,
                "optimizer_steps": 0, "validation_truth_opened": False, "protected_truth_opened": False,
                "scores_are_calibrated_probabilities": False, "global_assembly_performed": False,
                "capacity": {"demonstrated_reliable_piece_capacity": None, "baseline_target_pieces": 20, "milestone_pieces": 30}}
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(root / "bundle.json", manifest)
    reference = {"schema_version": "stage4.evaluation-reference.v1", "split": "synthetic_debug",
                 "input_manifest_sha256": manifest["manifest_sha256"], "pieces": {p: {"native_to_reference": poses[p]} for p in pieces},
                 "contacts": contacts}
    return reference
