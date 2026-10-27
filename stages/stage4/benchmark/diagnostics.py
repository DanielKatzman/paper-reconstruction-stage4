"""Post-prediction diagnosis. This module may open isolated evaluation truth.

No output from this process is passed back into any serving stage.
"""
import argparse
from collections import Counter
from pathlib import Path
import traceback
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from paper_reconstruction_contracts import load_bundle, map_points
from paper_reconstruction_stage4.outputs import read_result
from evaluation.evaluate import evaluate
from .common import atomic_json, digest, file_hash, inventory, load_cohort, read_json
from .paths import artifact_paths

TOLERANCE = 4.


def reference_preview(inputs, reference, target):
    source = inputs / "evaluation_only/reference_page.jpg"
    if not source.is_file():
        return None
    with Image.open(source) as image:
        image = image.convert("RGB")
    sy = image.height / reference["reference_size_hw"][0]
    sx = image.width / reference["reference_size_hw"][1]
    draw = ImageDraw.Draw(image)
    for seam in reference["seams"]:
        draw.line([(float(x)*sx, float(y)*sy) for x, y in seam["points_reference"]], fill="#e83148", width=2)
    rows = [__import__("json").loads(line) for line in (inputs / "stage1/segmentation_manifest.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        pid = row["capture_id"]
        with Image.open(inputs / "stage1" / row["piece_mask_path"]) as mask:
            y, x = np.nonzero(np.asarray(mask))
        point = map_points([[float(x.mean()), float(y.mean())]], reference["pieces"][pid]["native_to_reference"])[0]
        draw.text((point[0]*sx, point[1]*sy), pid, fill="#004bd1", stroke_width=2, stroke_fill="white")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return {"path": target.name, "sha256": file_hash(target)}


def arc_mask(candidate, count):
    length = int(candidate["length_points"])
    if not 0 <= int(candidate["start_index"]) < count or not 0 < length <= count:
        raise ValueError("Invalid candidate arc")
    result = np.zeros(count, bool)
    result[(int(candidate["start_index"]) + np.arange(length)) % count] = True
    return result


def arc_ious(candidates, target, ks=(1, 3, 5, 10, 32)):
    ranked = sorted(candidates, key=lambda c: (c["rank"], c["candidate_id"]))
    values = []
    for candidate in ranked:
        mask = arc_mask(candidate, len(target))
        union = int(np.count_nonzero(mask | target))
        values.append(float(np.count_nonzero(mask & target) / union) if union else 0.)
    return {str(k): max(values[:k], default=0.) for k in ks}


def seam_coverage(candidates, target):
    """Containment and specificity distinguish missed seams from merged arcs."""
    size = int(target.sum())
    overlaps = []
    for candidate in candidates:
        mask = arc_mask(candidate, len(target))
        intersection = int(np.count_nonzero(mask & target))
        overlaps.append((intersection/size if size else 0., intersection/int(mask.sum())))
    return {"maximum_seam_fraction_in_any_candidate": max((r for r, p in overlaps), default=0.),
            "best_candidate_precision_when_seam_fraction_at_least_0_8": max((p for r, p in overlaps if r >= .8), default=None),
            "mapped": size > 0}


def stage1_audit(root):
    rows = [__import__("json").loads(line) for line in (root / "segmentation_manifest.jsonl").read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        with Image.open(root / row["piece_mask_path"]) as image:
            mask = np.asarray(image.convert("L"))
        with Image.open(root / row["piece_path"]) as image:
            image_size = [image.height, image.width]
        foreground = mask > 0
        result.append({"piece_id": row["capture_id"], "size_hw": list(mask.shape), "image_size_hw": image_size,
            "area_px": int(foreground.sum()), "components_4connected": int(ndimage.label(foreground)[1]),
            "binary": set(np.unique(mask)).issubset({0, 255}), "accepted": row["accepted"]})
    return {"status": "bypassed", "segmentation_accuracy_measured": False, "pieces": result,
            "note": "Existing generator observed masks; network segmentation was not run."}


def nearest_distance(points, curves):
    if not curves:
        return np.full(len(points), np.inf)
    return cKDTree(np.concatenate(curves)).query(points)[0]


def stage2_diagnosis(root, reference):
    manifest = read_json(root / "edge_hypotheses_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Stage 2 output is not complete")
    pieces, sides, confusion = [], {}, np.zeros((2, 3), dtype=int)
    for entry in manifest["pieces"]:
        pid = entry["piece_id"]
        record = read_json(root / entry["record_path"])
        if file_hash(root / entry["arrays_path"]) != record["arrays_sha256"]:
            raise ValueError(f"Stage 2 array checksum mismatch: {pid}")
        if record["piece_id"] != pid or record["config_hash"] != manifest["config_hash"]:
            raise ValueError(f"Stage 2 record lineage mismatch: {pid}")
        transform = np.linalg.inv(reference["pieces"][pid]["native_to_reference"])
        with np.load(root / entry["arrays_path"], allow_pickle=False) as arrays:
            points = arrays["points_xy_crop"].copy()
            probabilities = arrays["role_probabilities"].copy()
        # Roles and distances are in native pixels; ambiguous junctions are excluded.
        if record["role_names"] != ["seam", "page_border", "uncertain"]:
            raise ValueError("Unexpected Stage 2 role order")
        seams = [s for s in reference["seams"] if pid in (s["piece_a"], s["piece_b"])]
        seam_curves = [map_points(s["points_reference"], transform) for s in seams]
        border_curves = [map_points(s["points_reference"], transform) for s in reference["borders"] if s["piece_id"] == pid]
        ds, db = nearest_distance(points, seam_curves), nearest_distance(points, border_curves)
        seam_label = (ds <= 8) & (ds + 2 < db)
        border_label = (db <= 8) & (db + 2 < ds)
        predicted = np.argmax(probabilities, axis=1)
        local_confusion = np.zeros((2, 3), dtype=int)
        for i, valid in enumerate((seam_label, border_label)):
            local_confusion[i] = np.bincount(predicted[valid], minlength=3)
        confusion += local_confusion
        seam_metrics = []
        for seam, curve in zip(seams, seam_curves):
            target = cKDTree(curve).query(points)[0] <= 8
            target &= seam_label
            ious = arc_ious(record["candidate_fragments"], target)
            metric = {"seam_id": seam["seam_id"], "mapped_contour_points": int(target.sum()),
                      "best_iou_at_k": ious, "covered_iou_0_5_at_32": bool(target.any() and ious["32"] >= .5),
                      **seam_coverage(record["candidate_fragments"], target)}
            seam_metrics.append(metric)
            sides[(seam["seam_id"], pid)] = metric
        pieces.append({"piece_id": pid, "contour_points": len(points), "candidates": record["candidate_count"],
                       "mean_role_probabilities": probabilities.mean(axis=0).tolist(),
                       "excluded_boundary_points": int(len(points)-seam_label.sum()-border_label.sum()),
                       "role_confusion": local_confusion.tolist(), "seams": seam_metrics,
                       "record_path": entry["record_path"], "overlay_path": record.get("overlay_path")})
    contacts = []
    for contact in reference["contacts"]:
        pair = (contact["piece_a"], contact["piece_b"])
        pair_seams = [s for s in reference["seams"] if (s["piece_a"], s["piece_b"]) == pair]
        both = [s["seam_id"] for s in pair_seams if all(sides[(s["seam_id"], p)]["covered_iou_0_5_at_32"] for p in pair)]
        contained = [s["seam_id"] for s in pair_seams if all(sides[(s["seam_id"], p)]["maximum_seam_fraction_in_any_candidate"] >= .8 for p in pair)]
        contacts.append({"piece_a": pair[0], "piece_b": pair[1], "at_least_one_seam_covered_both_sides": bool(both), "covered_seam_ids": both,
                         "at_least_one_seam_contained_both_sides_0_8": bool(contained), "contained_seam_ids": contained})
    return {"status": "complete", "manifest_summary": {k: manifest.get(k) for k in ("piece_count", "candidate_count", "elapsed_seconds", "runtime", "software_version", "release_version", "model_sha256", "config_hash")},
            "pieces": pieces, "contacts": contacts, "role_confusion": confusion.tolist(),
            "role_confusion_rows": ["reference_seam", "reference_page_border"],
            "role_confusion_columns": ["predicted_seam", "predicted_page_border", "predicted_uncertain"],
            "metric_definition": "Diagnostic contour approximation: closest generator seam/border in native pixels, <=8px and 2px nearer than competing role; exclude ambiguous/unmapped points. Arc IoU uses circular contour samples, not polygon area. Both-side coverage is any seam with IoU>=0.5 in each piece's top32.",
            "junction_and_mask_damage_limit": "This diagnostic mapping is not the historical Stage 2 training/evaluation label pipeline."}


def pose_error(transform, contact):
    try:
        actual = map_points(contact["points_b"], transform)
        value = float(np.quantile(np.linalg.norm(actual - np.asarray(contact["points_a"]), axis=1), .95))
        return value if np.isfinite(value) else None
    except (ValueError, np.linalg.LinAlgError):
        return None


def stage3_diagnosis(root, reference, work_root=None):
    bundle = load_bundle(root, full=True)
    contacts = {(c["piece_a"], c["piece_b"]): c for c in reference["contacts"]}
    rows = {pair: {"piece_a": pair[0], "piece_b": pair[1], "archive_geometries": 0, "archive_accurate": 0,
                   "archive_best_p95_px": None, "working_geometries": 0, "working_accurate": 0,
                   "working_best_p95_px": None, "best_accurate_working_weight_rank": None} for pair in contacts}
    counts = Counter()
    working = {e["id"]: e for e in bundle.edges}
    archive_pairs = set()
    candidate_counts = Counter(c["piece_id"] for c in bundle.manifest["candidates"])
    ranks = {}
    for pair in contacts:
        ranked = sorted((e for e in bundle.edges if (e["a"], e["b"]) == pair), key=lambda e: (-e["weight"], e["id"]))
        ranks.update({e["id"]: i for i, e in enumerate(ranked, 1)})
    for group in bundle.hypotheses():
        counts["archive_geometries"] += 1
        counts["occurrences"] += len(group["occurrences"])
        counts.update("variant:"+str(o.get("variant", "unspecified")) for o in group["occurrences"])
        pair = (group["piece_a"], group["piece_b"])
        archive_pairs.add(pair)
        if pair not in rows:
            continue
        item = rows[pair]
        item["archive_geometries"] += 1
        in_working = group["hypothesis_id"] in working
        item["working_geometries"] += int(in_working)
        error = pose_error(group["transform_b_to_a"], contacts[pair])
        if error is None:
            counts["invalid_probe_geometry"] += 1
            continue
        item["archive_best_p95_px"] = min(error, item["archive_best_p95_px"]) if item["archive_best_p95_px"] is not None else error
        accurate = error <= TOLERANCE
        item["archive_accurate"] += int(accurate)
        if in_working:
            item["working_best_p95_px"] = min(error, item["working_best_p95_px"]) if item["working_best_p95_px"] is not None else error
            item["working_accurate"] += int(accurate)
            if accurate:
                rank = ranks[group["hypothesis_id"]]
                prior = item["best_accurate_working_weight_rank"]
                item["best_accurate_working_weight_rank"] = min(prior, rank) if prior else rank
    provenance = bundle.manifest.get("provenance", {})
    pool_replays = []
    while provenance.get("kind") == "observable_saved_archive_pool_replay":
        if len(pool_replays) >= 8 or not isinstance(provenance.get("source_provenance"), dict):
            raise ValueError("Invalid or excessive pool replay provenance nesting")
        pool_replays.append({"source_manifest_sha256": provenance["source_manifest_sha256"],
                             "implementation": provenance["implementation"]})
        provenance = provenance["source_provenance"]
    search = provenance.get("search")
    parent_pairs = {tuple(s["pieces"]) for s in search} if search is not None else None
    work_root = work_root or root.with_name(root.name + ".work")
    ranking_path = work_root / "rankings.json"
    retrieval_pairs, cache_status = None, "unavailable"
    if ranking_path.is_file() and provenance.get("inference_signature"):
        cache = read_json(ranking_path)
        if cache["signature"] != provenance["inference_signature"]:
            raise ValueError("Stage 3 ranking cache signature mismatch")
        if cache["payload_sha256"] != digest(cache["rows"]):
            raise ValueError("Stage 3 ranking cache checksum mismatch")
        candidates = bundle.manifest["candidates"]
        retrieval_pairs = {tuple(sorted((candidates[r["query"]]["piece_id"], candidates[r["target"]]["piece_id"]))) for r in cache["rows"]}
        cache_status = "available_signature_bound"
    for pair, row in rows.items():
        row["physical_pair_retrieved"] = pair in retrieval_pairs if retrieval_pairs is not None else None
        row["physical_pair_retained_for_pose_search"] = pair in parent_pairs if parent_pairs is not None else None
        row["boundary_candidate_counts"] = {p: candidate_counts[p] for p in pair}
    adjacency = {p: set() for p in bundle.manifest["pieces"]}
    for edge in bundle.edges:
        adjacency[edge["a"]].add(edge["b"])
        adjacency[edge["b"]].add(edge["a"])
    remaining, components = set(adjacency), []
    while remaining:
        todo, found = [min(remaining)], set()
        while todo:
            p = todo.pop()
            if p not in found:
                found.add(p)
                todo.extend(adjacency[p]-found)
        remaining -= found
        components.append(sorted(found))
    return {"status": "complete", "manifest_sha256": bundle.manifest["manifest_sha256"],
        "counts": bundle.manifest["counts"], "archive_summary": dict(counts), "working_pool_pairs": bundle.pool["pairs"],
        "retrieval": provenance.get("retrieval"), "ranking_cache_status": cache_status,
        "pool_replay_lineage": pool_replays,
        "search_audits": search, "contacts": list(rows.values()),
        "runtime_variant": provenance.get("runtime_variant", "frozen"),
        "physical_pairs_processed": len(search) if search is not None else None,
        "archive_physical_pairs": len(archive_pairs),
        "candidate_graph_components": components, "candidate_graph_isolated_piece_ids": sorted(p for p in adjacency if not adjacency[p]),
        "candidate_counts_by_piece": {p: candidate_counts[p] for p in adjacency},
        "limitations": "Per-contact archive/working availability is a post-prediction diagnostic, not proof that all accurate transforms have compatible support or form a feasible global assembly. Correlated occurrences are not independent votes. Physical-pair retrieval does not prove the correct seam candidate was retrieved."}


def classify_contact(row, final_passed=None):
    if final_passed is True:
        return "final_contact_correct"
    if row["working_accurate"]:
        return "stage4_assembly_gap"
    if row["archive_accurate"]:
        return "stage3_working_pool_loss"
    if row["physical_pair_retrieved"] is False:
        return "stage3_physical_pair_retrieval_miss"
    if row["physical_pair_retained_for_pose_search"] is False:
        return "stage3_parent_selection_miss"
    if row["physical_pair_retained_for_pose_search"] is True:
        return "stage3_pose_search_gap"
    return "stage3_no_accurate_direct_pose_unlocalized"


def selected_support_roles(problem, result, reference):
    """Explain how Stage 2 labeled the supports actually used by Stage 4."""
    from paper_reconstruction_stage4.support import occurrence_arcs
    from paper_reconstruction_stage4.active_support import record_arc
    edges = {e.id: e for e in problem.edges}
    trees = {p: cKDTree(v.contour["points_xy"]) for p,v in problem.pieces.items()}
    neighbors = {(c["piece_a"], c["piece_b"]) for c in reference["contacts"]}
    rows = []
    for join in result["layouts"][0]["selected_joins"]:
        occurrence = next(o for o in edges[join["geometry_id"]].occurrences if o["occurrence_id"] == join["occurrence_id"])
        raw = occurrence_arcs(problem, occurrence)
        active = tuple(record_arc(a) for a in join["active_support"]["arcs"]) if join.get("active_support") else raw
        summaries = []
        for name, arcs in (("raw", raw), ("active_or_raw", active)):
            for arc in arcs:
                indices = trees[arc.piece].query(arc.points)[1]
                probabilities = problem.pieces[arc.piece].contour["role_probabilities"][indices]
                summaries.append({"support": name, "piece_id": arc.piece, "length_native_px": arc.length,
                    "mean_role_probabilities_seam_border_uncertain": np.mean(probabilities, axis=0).tolist(),
                    "sample_count": len(indices)})
        rows.append({"geometry_id": join["geometry_id"], "occurrence_id": join["occurrence_id"],
                     "pieces": [join["piece_a"], join["piece_b"]],
                     "source_pose": {k: occurrence["pose"]["alignment"].get(k) for k in
                        ("family", "underconstrained", "shape_anisotropy", "geometry_score", "scale",
                         "rms_residual_query_px", "normalized_residual", "fitted_overlap_length_query_px")},
                     "true_neighbor_identity": (join["piece_a"], join["piece_b"]) in neighbors,
                     "supports": summaries})
    return {"scope": "Offline explanation of supplied Stage2 probabilities on selected native supports; not calibrated truth or a new serving penalty.", "joins": rows}


def reference_geometry_feasibility(problem, config, reference):
    """Check whether the collision/warp rules admit the saved reference geometry.

    This is an offline constraint diagnostic, never a solver seed or proposal.
    """
    from dataclasses import replace
    from paper_reconstruction_stage4.feasibility import Feasibility
    from paper_reconstruction_stage4.geometry import normalize
    from paper_reconstruction_stage4.model import State
    root = min(problem.pieces)
    maps = {p: np.asarray(reference["pieces"][p]["native_to_reference"], float) for p in problem.pieces}
    anchor = np.linalg.inv(maps[root])
    poses = {p: normalize(anchor@h) for p,h in maps.items()}
    state = State(poses, {p:root for p in poses})
    observed = Feasibility(problem, config).state(state, final=True)
    alternate = None
    if not observed["valid"] and config.edge_erosion_px != 3:
        alternate = Feasibility(problem, replace(config, edge_erosion_px=3)).state(state, final=True)
    return {"scope": "Reference-geometry constraint diagnostic only; no selected-support compatibility is asserted, no poses are sent to serving, and the primary is unchanged.",
            "current_edge_erosion_px": config.edge_erosion_px, "current_rules": observed,
            "three_pixel_erosion_counterfactual": alternate,
            "interpretation": "A violation can expose native-mask damage/registration or overly strict constraints. It does not establish that approximate construction is impossible or that a useful producer pose exists."}


def diagnose_case(case, cohort_root, run_root, progress):
    inputs = cohort_root / case["path"]
    outputs = run_root / "cases" / case["case_id"]
    paths = artifact_paths(outputs, progress)
    report = {"schema_version": "stage4.pipeline-diagnosis.v1", "case": {k: v for k, v in case.items() if k != "files"},
              "serving_progress": progress, "errors": [], "stages": {}, "outputs": {},
              "artifact_roots": {k: str(v) for k, v in paths.items()}}
    # Check prediction commitment before opening references, when a prediction exists.
    result = None
    if (paths["stage4"] / "manifest.json").is_file():
        try:
            result = read_result(paths["stage4"])
        except Exception as exc:
            report["errors"].append({"stage": "stage4", "error": str(exc)})
    reference = read_json(inputs / "evaluation_only/reference_geometry.json")
    report["reference_sha256"] = file_hash(inputs / "evaluation_only/reference_geometry.json")
    report["stages"]["stage1"] = stage1_audit(inputs / "stage1")
    for stage, sentinel, diagnostic in (("stage2", "edge_hypotheses_manifest.json", stage2_diagnosis),
                                         ("stage3", "bundle.json", stage3_diagnosis)):
        if (paths[stage] / sentinel).is_file():
            try:
                report["stages"][stage] = diagnostic(paths[stage], reference, paths["stage3_work"]) if stage == "stage3" else diagnostic(paths[stage], reference)
            except Exception as exc:
                report["stages"][stage] = {"status": "diagnostic_failed", "error": str(exc)}
                report["errors"].append({"stage": stage, "error": str(exc), "traceback": traceback.format_exc()})
        else:
            report["stages"][stage] = {"status": "not_published", "note": "Missing committed output; not scored as an accuracy failure."}
        if paths[stage].is_dir():
            report["outputs"][stage] = inventory(paths[stage])
    if result is not None and (report["stages"]["stage3"]["status"] != "complete" or
                              report["stages"]["stage3"].get("manifest_sha256") != result["input"]["manifest_sha256"]):
        report["errors"].append({"stage": "stage4", "error": "Cannot bind prediction to the verified Stage 3 bundle"})
        result = None
    if result is not None:
        reference["input_manifest_sha256"] = result["input"]["manifest_sha256"]
        evaluation = evaluate(result, reference, TOLERANCE)
        report["stages"]["stage4"] = {"status": "complete", "evaluation": evaluation,
            "solver_status": result["status"], "termination": result["termination"], "components": result["layouts"][0]["components"],
            "selected_joins": result["layouts"][0]["selected_joins"], "feasibility": result["layouts"][0]["feasibility"],
            "refinement": result["layouts"][0]["refinement"], "search": result["search"],
            "rejections": result["rejections"], "resources": result["resources"],
            "input_load": result["input"]["load"],
            "portfolio_layouts": len(result["layouts"]), "layout_evaluated": 0}
        from .quality import assembly_quality, contact_path_diagnostics, placement_overlay
        report["stages"]["stage4"]["assembly_quality"] = assembly_quality(result, reference, evaluation)
        if result["solver"] == "joint_beam_v3":
            from paper_reconstruction_stage4.config import SolverConfig
            from paper_reconstruction_stage4.inputs import load_problem
            from .joint_diagnostics import audit_candidates, support_plot
            config = SolverConfig.from_dict(result["configuration"])
            problem = load_problem(paths["stage3"], config)
            target = run_root / "report" / case["case_id"] / "support_intervals.svg"
            target.parent.mkdir(parents=True, exist_ok=True)
            report["existing_complete_candidate_audit"] = audit_candidates(result, problem, config, reference, maximum=32, seconds=30.,
                preview_path=target.parent / "diagnostic_existing_candidate.png")
            report["selected_support_role_diagnosis"] = selected_support_roles(problem, result, reference)
            report["reference_geometry_feasibility"] = reference_geometry_feasibility(problem, config, reference)
            support_plot(problem, result, target)
        path_diagnostics = contact_path_diagnostics(result["layouts"][0], reference["contacts"], pose_error)
        report["placement_overlay"] = placement_overlay(result, reference, inputs, paths["stage3"], run_root / "report" / case["case_id"] / "placement_overlay.png")
    else:
        report["stages"]["stage4"] = {"status": "diagnostic_failed" if any(e["stage"] == "stage4" for e in report["errors"]) else "not_published"}
    if paths["stage4"].is_dir():
        report["outputs"]["stage4"] = inventory(paths["stage4"])
    s3 = report["stages"]["stage3"]
    report["contact_diagnoses"] = []
    if s3["status"] == "complete":
        evaluation_rows = report["stages"]["stage4"].get("evaluation", {}).get("contacts", [])
        final = {(c["piece_a"], c["piece_b"]): c for c in evaluation_rows}
        s2 = {(c["piece_a"], c["piece_b"]): c for c in report["stages"]["stage2"].get("contacts", [])}
        for row in s3["contacts"]:
            pair = (row["piece_a"], row["piece_b"])
            last = final.get(pair)
            report["contact_diagnoses"].append({**row, "stage2_boundary_coverage": s2.get(pair),
                "selected_path_comparison": path_diagnostics[pair] if result is not None else None,
                "classification_scope": "Direct-pose availability and final precision at 4px, not a structural assembly verdict",
                "final": last, "diagnosis": classify_contact(row, last["passed"] if last else None) if result is not None else "stage4_not_published"})
    report["diagnosis_counts"] = dict(Counter(c["diagnosis"] for c in report["contact_diagnoses"]))
    report["findings"] = []
    s2 = report["stages"]["stage2"]
    if s2["status"] == "complete":
        covered = sum(c["at_least_one_seam_covered_both_sides"] for c in s2["contacts"])
        contained = sum(c["at_least_one_seam_contained_both_sides_0_8"] for c in s2["contacts"])
        report["findings"].append(f"Stage 2: {covered}/{len(s2['contacts'])} true neighbor pairs have a candidate covering at least one shared seam on both sides at IoU >= 0.5 (top32).")
        report["findings"].append(f"Stage 2: {contained}/{len(s2['contacts'])} true pairs have >=80% of at least one seam contained on both sides. High containment with low IoU indicates broad/merged arcs rather than an entirely missing seam.")
    if s3["status"] == "complete":
        archived = sum(c["archive_accurate"] > 0 for c in s3["contacts"])
        retained = sum(c["working_accurate"] > 0 for c in s3["contacts"])
        report["findings"].append(f"Stage 3: a direct pose passes the strict 4px probe test for {archived}/{len(s3['contacts'])} true pairs in the full archive and {retained}/{len(s3['contacts'])} in the working pool. This is a precision diagnostic, not an assembly-success count.")
    if result is not None:
        ev = report["stages"]["stage4"]["evaluation"]
        report["findings"].append(f"Stage 4: {ev['passing_implied_contacts']}/{ev['expected_contacts']} implied contacts pass; {len(ev['false_neighbor_joins'])} false neighbor joins; largest component linked by correct selected contacts has {ev['largest_component_from_correct_selected_contacts']}/{case['pieces']} pieces. Strict whole-page pass: {ev['strict_whole_page_pass']}.")
    if progress.get("status") != "predicted":
        report["findings"].append(f"Serving did not complete: {progress.get('failed_stage', progress.get('active_stage', 'not started'))}. Missing downstream accuracy is unavailable.")
    report["stage1_input_receipts"] = [r for r in case["files"] if r["path"].startswith("stage1/")]
    report["logs_and_work_receipts"] = []
    for folder in (paths["logs"], paths["stage3_work"], paths["stage4_work"]):
        if folder.is_dir():
            report["logs_and_work_receipts"].extend([{**r, "path": f"{folder.relative_to(outputs.resolve()).as_posix()}/{r['path']}"} for r in inventory(folder)])
    if "stage3_profile" in paths and paths["stage3_profile"].is_dir():
        from .common import receipt
        folder = paths["stage3_profile"]
        report["profile_artifacts"] = [{**receipt(p, outputs.resolve())} for p in sorted(folder.iterdir()) if p.is_file()]
        if (folder / "profile.json").is_file():
            report["stage3_profile"] = read_json(folder / "profile.json")
    report["diagnostic_code_sha256"] = digest({p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    report["reference_preview"] = reference_preview(inputs, reference, run_root / "report" / case["case_id"] / "reference_cuts.jpg")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    cohort_root, cohort = load_cohort(args.benchmark)
    run_root = args.run.resolve()
    request, progress = read_json(run_root / "request.json"), read_json(run_root / "progress.json")
    if progress.get("phase") not in {"diagnostics", "finished", "diagnostics_failed"}:
        raise ValueError("Prediction phase has not ended; run diagnostics after the serving phase")
    if request["benchmark_sha256"] != cohort["benchmark_sha256"]:
        raise ValueError("Run/benchmark identity mismatch")
    selected = [c for c in cohort["cases"] if c["case_id"] in request["case_ids"]]
    reports = []
    for index, case in enumerate(selected, 1):
        try:
            value = diagnose_case(case, cohort_root, run_root, progress["cases"].get(case["case_id"], {}))
        except Exception as exc:
            value = {"case": {k: v for k, v in case.items() if k != "files"}, "stages": {}, "outputs": {},
                     "errors": [{"stage": "diagnosis", "error": str(exc), "traceback": traceback.format_exc()}], "diagnosis_counts": {}}
        atomic_json(run_root / "report" / case["case_id"] / "diagnosis.json", value)
        reports.append(value)
        print(f"Diagnosed {index}/{len(selected)}: {case['case_id']}", flush=True)
    from .report import write_reports
    write_reports(reports, cohort_root, run_root)
    return 1 if any(r["errors"] for r in reports) else 0
