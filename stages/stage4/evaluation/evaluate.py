"""Explicit offline evaluation. Never imported by the serving solver.

Reference inputs are supplied by the caller only after immutable predictions
exist. This process does not edit those predictions or select another layout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.geometry import normalize
from paper_reconstruction_stage4.outputs import atomic_json, file_hash, read_result


def evaluate(result, reference, tolerance=4.):
    if reference.get("schema_version") != "stage4.evaluation-reference.v1":
        raise ValueError("Unsupported explicit evaluation reference")
    layout = result["layouts"][0]
    pieces = {p["piece_id"]: p for p in layout["pieces"]}
    if set(pieces) != set(reference["pieces"]):
        raise ValueError("Reference and prediction inventories differ")
    if reference.get("input_manifest_sha256") != result["input"]["manifest_sha256"]:
        raise ValueError("Reference is not bound to this observable input")
    reference_poses = {p: normalize(v["native_to_reference"]) for p, v in reference["pieces"].items()}
    predicted = {p: normalize(v["native_to_component"]) for p, v in pieces.items()}
    pairs = []
    unresolved_pairs = 0
    for a in sorted(pieces):
        for b in sorted(pieces):
            if a == b:
                continue
            if pieces[a]["component_id"] != pieces[b]["component_id"]:
                unresolved_pairs += 1
                pairs.append({"piece_a": a, "piece_b": b, "p95_native_a_px": None, "passed": False})
                continue
            points = np.asarray(pieces[b]["native_probes"], float)
            actual = map_points(points, np.linalg.inv(predicted[a]) @ predicted[b])
            expected = map_points(points, np.linalg.inv(reference_poses[a]) @ reference_poses[b])
            error = float(np.quantile(np.linalg.norm(actual - expected, axis=1), .95))
            pairs.append({"piece_a": a, "piece_b": b, "p95_native_a_px": error, "passed": error <= tolerance})
    contacts, expected_pairs = [], set()
    for contact in reference["contacts"]:
        a, b = contact["piece_a"], contact["piece_b"]
        if a not in pieces or b not in pieces or not a < b or (a, b) in expected_pairs:
            raise ValueError("Invalid/duplicate canonical reference contact")
        expected_pairs.add((a, b))
        points = np.asarray(contact["points_b"], float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Invalid saved contact probe set")
        if pieces[a]["component_id"] != pieces[b]["component_id"]:
            error = None
        else:
            actual = map_points(points, np.linalg.inv(predicted[a]) @ predicted[b])
            expected = (np.asarray(contact["points_a"], float) if "points_a" in contact else
                        map_points(points, np.linalg.inv(reference_poses[a]) @ reference_poses[b]))
            if expected.shape != actual.shape or not np.isfinite(expected).all():
                raise ValueError("Invalid saved native-A contact anchors")
            error = float(np.quantile(np.linalg.norm(actual - expected, axis=1), .95))
        contacts.append({"piece_a": a, "piece_b": b, "p95_native_a_px": error,
                         "passed": error is not None and error <= tolerance})
    selected_pairs = {(j["piece_a"], j["piece_b"]) for j in layout["selected_joins"]}
    false = sorted(selected_pairs - expected_pairs)
    missing = sorted(expected_pairs - selected_pairs)
    passing_contacts = sum(c["passed"] for c in contacts)
    correct_joins = sum(c["passed"] and (c["piece_a"], c["piece_b"]) in selected_pairs for c in contacts)
    correct_component_size = 1 if pieces else 0
    # Connected components in the graph of correct selected joins are a
    # transparent contact-based metric, not an oracle of unselected layouts.
    adjacency = {p: set() for p in pieces}
    for c in contacts:
        if c["passed"] and (c["piece_a"], c["piece_b"]) in selected_pairs:
            adjacency[c["piece_a"]].add(c["piece_b"])
            adjacency[c["piece_b"]].add(c["piece_a"])
    remaining = set(pieces)
    while remaining:
        todo, found = [min(remaining)], set()
        while todo:
            p = todo.pop()
            if p not in found:
                found.add(p)
                todo.extend(adjacency[p] - found)
        remaining -= found
        correct_component_size = max(correct_component_size, len(found))
    strict = (layout["all_pieces_connected"] and not false and layout["feasibility"]["valid"]
              and all(c["passed"] for c in contacts) and all(p["passed"] for p in pairs))
    return {"schema_version": "stage4.evaluation-result.v1", "prediction_run_id": result["run_id"],
            "prediction_layout_sha256": layout["layout_sha256"], "reference_split": reference.get("split", "unspecified"),
            "tolerance_native_px": tolerance, "strict_whole_page_pass": bool(strict),
            "expected_contacts": len(contacts), "passing_implied_contacts": passing_contacts,
            "implied_contact_recall": passing_contacts / len(contacts) if contacts else None,
            "selected_joins": len(selected_pairs), "correct_selected_joins": correct_joins,
            "false_neighbor_joins": false, "expected_contacts_not_explicitly_selected": missing,
            "selected_join_precision": correct_joins / len(selected_pairs) if selected_pairs else None,
            "largest_component_from_correct_selected_contacts": correct_component_size,
            "unresolved_ordered_pairs": unresolved_pairs,
            "maximum_pairwise_p95_native_px": None if unresolved_pairs else max((p["p95_native_a_px"] for p in pairs), default=0.),
            "contacts": contacts, "ordered_pairs": pairs, "accuracy_is_calibration": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = read_result(args.result)  # Verify commitment before opening truth.
        prediction_root = args.result.resolve()
        if prediction_root.is_file():
            prediction_root = prediction_root.parent
        if args.output.resolve().is_relative_to(prediction_root):
            raise ValueError("Evaluation output must be separate from predictions")
        if args.output.exists():
            raise ValueError("Evaluation output already exists")
        reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
        if reference.get("split") in {"holdout", "test", "confirmation"} and not args.allow_confirmation:
            raise ValueError("Confirmation evaluation requires explicit --allow-confirmation")
        report = evaluate(result, reference)
        report["reference_sha256"] = file_hash(args.reference)
        atomic_json(args.output, report)
        print(json.dumps({k: report[k] for k in ("strict_whole_page_pass", "expected_contacts", "passing_implied_contacts", "selected_join_precision")}))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "failed", "detail": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
