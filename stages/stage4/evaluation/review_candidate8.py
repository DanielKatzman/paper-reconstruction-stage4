"""Offline evidence attribution for the completed fixed eight-piece pilot.

Uses truth to identify diagnostic alternatives, never to select a serving result.
Original evidence, prediction, config and automated diagnosis remain untouched.
"""
import argparse
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
import csv
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from paper_reconstruction_contracts import load_bundle, map_points
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.geometry import normalize
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import Edge, State
from paper_reconstruction_stage4.outputs import read_result
from paper_reconstruction_stage4.search.common import objective
from paper_reconstruction_stage4.support import SupportIndex
from benchmark.common import WORKSPACE, atomic_json, file_hash, read_json
from benchmark.diagnostics import pose_error
from benchmark.quality import placement_overlay, quantiles

CASE = "puzzle_000174_0f1f6d83b42a"


def occurrence_score(occurrence):
    # Mirrors the documented frozen exporter priority; validated against every
    # retained SDK edge below. No producer package is imported.
    for method in ("mlp_start", "alpha_fused", "cnn_budget1", "reference"):
        value = occurrence.get("methods", {}).get(method, {}).get("weight")
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return float(occurrence["pose"]["alignment"].get("geometry_score", 0.))


def representative(group):
    candidates = [o for o in group["occurrences"] if o["variant"] == "source_control"] or group["occurrences"]
    return min(candidates, key=lambda o: (-occurrence_score(o), o["occurrence_id"]))


def components(pieces, pairs):
    adjacency = {p: set() for p in pieces}
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    todo, result = set(pieces), []
    while todo:
        stack, found = [min(todo)], set()
        while stack:
            p = stack.pop()
            if p not in found:
                found.add(p)
                stack.extend(adjacency[p]-found)
        result.append(sorted(found))
        todo -= found
    return sorted(result, key=lambda c: (-len(c), c))


def minimum_tree(pieces, rows):
    groups = {p: p for p in pieces}
    selected = []
    for row in sorted(rows, key=lambda r: (r["contact_p95_px"], r["geometry_id"])):
        a, b = row["pair"]
        if groups[a] == groups[b]:
            continue
        old, new = groups[b], groups[a]
        groups = {p: new if g == old else g for p, g in groups.items()}
        selected.append(row)
    return selected


def tree_state(problem, edges):
    adjacency = defaultdict(list)
    for e in edges:
        adjacency[e.a].append((e.b, e.transform))
        adjacency[e.b].append((e.a, np.linalg.inv(e.transform)))
    poses, groups = {}, {}
    for root in sorted(problem.pieces):
        if root in poses:
            continue
        poses[root], groups[root] = np.eye(3), root
        queue = deque([root])
        while queue:
            p = queue.popleft()
            for neighbor, h in adjacency[p]:
                if neighbor not in poses:
                    poses[neighbor] = normalize(poses[p] @ h)
                    groups[neighbor] = root
                    queue.append(neighbor)
    return State(poses, groups, tuple(e.id for e in edges), tuple(e.id for e in edges))


def check_tree(problem, config, reference, chosen, primary_score):
    edges = {e.id: e for e in chosen}
    subset = replace(problem, edges=chosen)
    state = tree_state(subset, chosen)
    support = SupportIndex(subset, config)
    assignments, reason, nodes = support.assign(state.selected, state.poses, edges)
    physical = Feasibility(subset, config).state(state, final=True)
    state.score = objective(state, subset, edges, config)
    align = np.asarray(reference["pieces"]["piece_001"]["native_to_reference"])
    displacements, per_piece = [], []
    for p, piece in sorted(subset.pieces.items()):
        if state.groups[p] != state.groups["piece_001"]:
            per_piece.append({"piece_id": p, "p95_reference_px": None, "reason": "Separate component"})
            continue
        # One common anchor; the root pose is the identity, never per-piece fit.
        contour = piece.contour["points_xy"]
        points = contour[np.linspace(0, len(contour)-1, 65).astype(int)]
        pred = map_points(points, align @ state.poses[p])
        target = map_points(points, reference["pieces"][p]["native_to_reference"])
        errors = np.linalg.norm(pred-target, axis=1)
        center = np.array([piece.shape[1]/2, piece.shape[0]/2])
        basis = np.array([center, center+[1., 0.]])
        u = np.diff(map_points(basis, align @ state.poses[p]), axis=0)[0]
        v = np.diff(map_points(basis, reference["pieces"][p]["native_to_reference"]), axis=0)[0]
        angle = float(np.degrees(np.arctan2(v[0]*u[1]-v[1]*u[0], np.dot(u, v))))
        per_piece.append({"piece_id": p, "p95_reference_px": float(np.quantile(errors, .95)), "local_orientation_error_degrees": angle})
        if p != "piece_001":
            displacements.extend(errors.tolist())
    contact_errors = []
    for c in reference["contacts"]:
        a, b = c["piece_a"], c["piece_b"]
        error = pose_error(np.linalg.inv(state.poses[a]) @ state.poses[b], c) if state.groups[a] == state.groups[b] else None
        contact_errors.append({"pair": [a, b], "p95_native_a_px": error})
    return {"geometry_ids": list(state.selected), "objective_score": state.score, "objective_delta_from_primary": state.score-primary_score,
            "component_sizes": [len(c) for c in state.components()], "support_status": reason, "support_assignment_nodes": nodes,
            "assigned_occurrence_ids": assignments, "support_invalid_occurrences": support.invalid,
            "physical_valid": physical["valid"], "physical_reason": physical["reason"],
            "maximum_overlap": physical.get("maximum_overlap"),
            "failed_physical_pairs": [{k: x.get(k) for k in ("a", "b", "reason", "overlap")} for x in physical.get("pairs", []) if not x["valid"]],
            "common_anchor_contour_errors_reference_px": quantiles(displacements), "per_piece_contour_errors": per_piece,
            "contact_errors": contact_errors,
            "all_current_constraints_pass": physical["valid"] and assignments is not None,
            "interpretation": "Truth-assisted diagnostic only. A failing chosen tree does not prove every alternative is infeasible. No serving prediction or success claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=WORKSPACE / "outputs/candidate8_exact_v1")
    args = parser.parse_args()
    start = time.perf_counter()
    run = args.run.resolve()
    progress, request = read_json(run / "progress.json"), read_json(run / "request.json")
    if progress["phase"] != "finished" or request["case_ids"] != [CASE]:
        raise ValueError("Requires the stopped, completed fixed eight-piece pilot")
    case = run / "cases" / CASE
    result = read_result(case / "stage4")  # Verify prediction before truth.
    immutable = [case / "stage4" / p for p in ("manifest.json", "result.json", "preview.png")]
    immutable += [case / "stage3_profile/evidence/bundle.json", run / "request.json", run / "stage4_config.json"]
    before = {str(p): file_hash(p) for p in immutable}
    input_root = WORKSPACE / "benchmarks/development_v2/cases" / CASE
    reference = read_json(input_root / "evaluation_only/reference_geometry.json")
    contacts = {(c["piece_a"], c["piece_b"]): c for c in reference["contacts"]}
    config = SolverConfig.from_dict(read_json(run / "stage4_config.json"))
    problem = load_problem(case / "stage3_profile/evidence", config)
    if problem.manifest["manifest_sha256"] != result["input"]["manifest_sha256"]:
        raise ValueError("Prediction and evidence differ")
    bundle = load_bundle(problem.root, full=True)
    groups = {g["hypothesis_id"]: g for g in bundle.hypotheses()}
    working = {e.id: e for e in problem.edges}
    rows, best_archive, best_working, per_pair = [], {}, {}, defaultdict(list)
    all_edges = dict(working)
    for gid, g in groups.items():
        rep = representative(g)
        score = occurrence_score(rep)
        if gid in working:
            assert abs(working[gid].weight-score) < 1e-12
        else:
            all_edges[gid] = Edge(gid, g["piece_a"], g["piece_b"], np.asarray(g["transform_b_to_a"]), score, g["occurrences"])
        pair = (g["piece_a"], g["piece_b"])
        if pair not in contacts:
            continue
        error = pose_error(g["transform_b_to_a"], contacts[pair])
        if error is None:
            continue
        row = {"pair": list(pair), "geometry_id": gid, "contact_p95_px": error, "in_working_pool": gid in working,
               "weight": score, "representative_occurrence_id": rep["occurrence_id"], "underconstrained": rep["pose"]["alignment"].get("underconstrained")}
        rows.append(row)
        per_pair[pair].append(row)
        for store, included in ((best_archive, True), (best_working, gid in working)):
            if included and (pair not in store or (error, gid) < (store[pair]["contact_p95_px"], store[pair]["geometry_id"])):
                store[pair] = row
    for pair, values in per_pair.items():
        for rank, row in enumerate(sorted((r for r in values if r["in_working_pool"]), key=lambda r: (-r["weight"], r["geometry_id"])), 1):
            row["weight_rank_in_pair_pool"] = rank
    sensitivity = []
    for tolerance in (4, 8, 16, 32, 48):
        entry = {"tolerance_px": tolerance, "diagnostic_only_not_acceptance_threshold": True}
        for name, best in (("archive", best_archive), ("working", best_working)):
            pairs = [pair for pair, row in best.items() if row["contact_p95_px"] <= tolerance]
            entry[name] = {"true_pairs_with_pose": len(pairs), "components": components(problem.pieces, pairs)}
        sensitivity.append(entry)
    primary = result["layouts"][0]
    checks, pair_rows = {}, []
    checks["primary_tree_reconstruction"] = check_tree(problem, config, reference, [working[j["geometry_id"]] for j in primary["selected_joins"]], primary["objective_score"])
    assert abs(checks["primary_tree_reconstruction"]["objective_delta_from_primary"]) < 1e-10
    for name, best in (("archive", best_archive), ("working", best_working)):
        tree = minimum_tree(problem.pieces, best.values())
        checks[name+"_minimum_error_tree"] = {"bottleneck_contact_px": max(r["contact_p95_px"] for r in tree),
            **check_tree(problem, config, reference, [all_edges[r["geometry_id"]] for r in tree], primary["objective_score"])}
    # Test specific pool omissions as bounded counterfactuals, without expanding
    # the actual bundle or using these truth-selected additions in serving.
    for pairs, name in [([("piece_003", "piece_008")], "working_plus_archive_3_8"),
                        ([("piece_003", "piece_008"), ("piece_004", "piece_005")], "working_plus_archive_3_8_and_4_5")]:
        augmented = dict(best_working)
        augmented.update({pair: best_archive[pair] for pair in pairs})
        tree = minimum_tree(problem.pieces, augmented.values())
        checks[name] = check_tree(problem, config, reference, [all_edges[r["geometry_id"]] for r in tree], primary["objective_score"])
    for pair in sorted(contacts):
        values = per_pair[pair]
        passing = sorted((v for v in values if v["in_working_pool"] and v["contact_p95_px"] <= 4), key=lambda r: (-r["weight"], r["geometry_id"]))
        single = check_tree(problem, config, reference, [all_edges[best_working[pair]["geometry_id"]]], primary["objective_score"])
        pair_rows.append({"pair": list(pair), "archive_best": best_archive[pair], "working_best": best_working[pair],
                          "highest_scoring_working_pose_at_4px": passing[0] if passing else None,
                          "best_working_pose_independent_support_and_collision": {k:single[k] for k in ("support_status", "physical_valid", "all_current_constraints_pass")}})
    # Isolate the two clear selected-pair choices without a second beam solve.
    for pair in (("piece_001", "piece_005"), ("piece_002", "piece_007")):
        replacement = best_working[pair]
        chosen = [all_edges[replacement["geometry_id"]] if (j["piece_a"], j["piece_b"]) == pair else working[j["geometry_id"]] for j in primary["selected_joins"]]
        checks["replace_"+"_".join(pair)] = check_tree(problem, config, reference, chosen, primary["objective_score"])
    # Verify native reference correspondence arithmetic independently of scores.
    consistency = [pose_error(np.linalg.inv(reference["pieces"][a]["native_to_reference"]) @ reference["pieces"][b]["native_to_reference"], c) for (a, b), c in contacts.items()]
    assert all(file_hash(Path(p)) == sha for p, sha in before.items())
    output = run / "review"
    output.mkdir(exist_ok=True)
    for name in ("archive_minimum_error_tree", "working_plus_archive_3_8"):
        state = tree_state(problem, [all_edges[eid] for eid in checks[name]["geometry_ids"]])
        illustration = {"layouts": [{"components": state.components(), "pieces": [{"piece_id": p, "native_to_component": h.tolist()} for p, h in state.poses.items()]}]}
        overlay = placement_overlay(illustration, reference, input_root, problem.root, output / ("truth_assisted_"+name+".png"))
        checks[name]["truth_assisted_overlay"] = overlay
    with (output / "true_contact_geometries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair", "geometry_id", "contact_p95_px", "in_working_pool", "weight", "weight_rank_in_pair_pool", "representative_occurrence_id", "underconstrained"])
        writer.writeheader()
        writer.writerows(rows)
    audit = {"schema_version": "stage4.offline-candidate-review.v1", "scope": "Truth-assisted attribution only; no producer/beam rerun or new primary prediction",
             "run_id": result["run_id"], "input_manifest_sha256": problem.manifest["manifest_sha256"], "primary_layout_sha256": primary["layout_sha256"],
             "review_script_sha256": file_hash(Path(__file__)), "input_files_before_and_after_unchanged": before,
             "reference_self_consistency_max_px": max(consistency), "pair_evidence": pair_rows, "graph_sensitivity": sensitivity,
             "counterfactuals": checks, "primary_objective_score": primary["objective_score"], "elapsed_seconds": time.perf_counter()-start}
    atomic_json(output / "evidence_review.json", audit)
    print(__import__("json").dumps({"review": str(output / "evidence_review.json"), "reference_self_consistency_max_px": max(consistency),
        "graph_sensitivity": sensitivity, "counterfactuals": {k: {f:v.get(f) for f in ("bottleneck_contact_px", "objective_score", "objective_delta_from_primary", "support_status", "physical_valid", "failed_physical_pairs", "all_current_constraints_pass", "common_anchor_contour_errors_reference_px")} for k,v in checks.items()}, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
