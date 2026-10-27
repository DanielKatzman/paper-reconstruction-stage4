"""Portable HTML/JSON/CSV reports with direct links to every stage artifact."""
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from collections import Counter
from .common import atomic_json, read_json

STYLE = """body{font:15px system-ui;margin:28px;color:#172a3a;background:#f4f7fa}h1,h2,h3{color:#102f46}a{color:#075f96}table{border-collapse:collapse;width:100%;background:white;margin:12px 0}th,td{text-align:left;border-bottom:1px solid #d6dfe7;padding:9px;vertical-align:top}th{background:#e5eef4;position:sticky;top:0}code,pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}pre{background:#fff;padding:14px;max-height:600px;overflow:auto}img{max-width:100%;max-height:700px;object-fit:contain;background:white}.visuals{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:white;padding:16px;border-radius:8px;margin:14px 0}.warning{border-left:5px solid #be7b19;padding:14px;background:#fff4de}.bad{color:#a62a28}.good{color:#166534}details{margin:12px 0}select{padding:7px}small{color:#4e6070}figure{margin:8px}figcaption{font-weight:600;margin:8px 0}"""


def esc(value):
    return html.escape("unavailable" if value is None else str(value), quote=True)


def link(path, here, label=None):
    relative = quote(os.path.relpath(path, here).replace("\\", "/"), safe="/")
    return f'<a href="{relative}">{esc(label or Path(path).name)}</a>'


def image(path, here, caption):
    if not path.is_file():
        return f'<div class="card">{esc(caption)}: unavailable</div>'
    relative = quote(os.path.relpath(path, here).replace("\\", "/"), safe="/")
    return f'<figure><figcaption>{esc(caption)}</figcaption><a href="{relative}"><img loading="lazy" src="{relative}" alt="{esc(caption)}"></a></figure>'


def document(title, content):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style><body>{content}</body></html>'


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def number(value):
    return 'unavailable' if value is None else f'{value:.3f}'


def brief(report):
    s = report["stages"]
    s2, s3, s4 = s.get("stage2", {}), s.get("stage3", {}), s.get("stage4", {})
    ev = s4.get("evaluation", {})
    contacts = s3.get("contacts", [])
    quality = s4.get("assembly_quality", {})
    gauge = quality.get("common_gauge", {})
    per_piece = gauge.get("per_piece", [])
    orientations = [abs(p["local_orientation_error_degrees"]) for p in per_piece if p.get("local_orientation_error_degrees") is not None]
    scales = [p["local_scale_ratio"] for p in per_piece if p.get("local_scale_ratio") is not None]
    progress = report.get("serving_progress", {})
    attempts = progress.get("attempts", [])
    boundaries = s2.get("contacts", [])
    confusion = s2.get("role_confusion", [])
    mapped = sum(sum(r) for r in confusion)
    return {"case_id": report["case"]["case_id"], "pieces": report["case"]["pieces"], "difficulty": report["case"]["difficulty"],
        "stage2_status": s2.get("status", "unavailable"), "stage3_status": s3.get("status", "unavailable"),
        "stage4_status": s4.get("status", "unavailable"), "strict_pass": ev.get("strict_whole_page_pass"),
        "implied_contact_recall": ev.get("implied_contact_recall"), "join_precision": ev.get("selected_join_precision"),
        "largest_correct_component": ev.get("largest_component_from_correct_selected_contacts"),
        "false_joins": len(ev["false_neighbor_joins"]) if "false_neighbor_joins" in ev else None,
        "selected_neighbor_identity_precision": quality.get("selected_neighbor_identity_precision"),
        "assembled_component_sizes": ",".join(str(n) for n in quality.get("component_sizes", [])) or None,
        "structural_visual_assessment": quality.get("structural_visual_assessment"),
        "all_pieces_connected": quality.get("all_pieces_connected"),
        "selected_true_neighbor_pairs": quality.get("selected_true_neighbor_pairs"),
        "selected_physical_pairs": quality.get("selected_physical_pairs"),
        "max_common_gauge_orientation_degrees": max(orientations, default=None),
        "common_gauge_scale_min": min(scales, default=None), "common_gauge_scale_max": max(scales, default=None),
        "contact_error_p50_native_a_px": quality.get("contact_p95_error_distribution_native_a_px", {}).get("p50"),
        "whole_page_placement_p95_reference_px": gauge.get("non_anchor_probe_errors_reference_px", {}).get("p95") if gauge.get("whole_page_error_available") else None,
        "latest_prediction_attempt_wall_seconds": progress.get("latest_attempt_wall_seconds"),
        "has_reused_stages": any(a.get("status") in {"reused", "reused_complete"} for a in attempts),
        "stage2_mapped_role_accuracy": (confusion[0][0]+confusion[1][1])/mapped if mapped else None,
        "stage2_both_side_arc_iou_recall": sum(c["at_least_one_seam_covered_both_sides"] for c in boundaries)/len(boundaries) if boundaries else None,
        "stage2_both_side_containment_recall": sum(c["at_least_one_seam_contained_both_sides_0_8"] for c in boundaries)/len(boundaries) if boundaries else None,
        "stage3_physical_pair_retrieval_recall": sum(c["physical_pair_retrieved"] for c in contacts)/len(contacts) if contacts and all(c["physical_pair_retrieved"] is not None for c in contacts) else None,
        "stage3_archive_contact_recall": sum(c["archive_accurate"] > 0 for c in contacts)/len(contacts) if contacts else None,
        "stage3_working_contact_recall": sum(c["working_accurate"] > 0 for c in contacts)/len(contacts) if contacts else None,
        "diagnostic_errors": len(report["errors"])}


def case_report(report, cohort_root, run_root):
    case = report["case"]
    here = run_root / "report" / case["case_id"]
    here.mkdir(parents=True, exist_ok=True)
    inputs, outputs = cohort_root / case["path"], run_root / "cases" / case["case_id"]
    summary = brief(report)
    content = link(here.parent / "index.html", here, "All puzzles")
    content += f'<h1>{esc(case["case_id"])} · {case["pieces"]} pieces · {esc(case["difficulty"])}</h1>'
    content += f'<p>Source: {esc(case["source_family"])}; page index {case["source_page_index"]}. Development training split. Stage 1 bypassed.</p>'
    content += table(["Measurement", "Value"], [(esc(k), esc(v)) for k, v in summary.items() if k not in {"case_id", "difficulty", "pieces"}])
    content += '<div class="card">'+''.join('<p>'+esc(f)+'</p>' for f in report.get("findings", []))+'</div>'
    if report["errors"]:
        content += '<div class="warning">Diagnostic errors: inspect the details before drawing conclusions.</div><pre>'+esc(json.dumps(report["errors"], indent=2))+'</pre>'
    content += '<div class="visuals">'+image(inputs / "stage1/input_pieces.jpg", here, "Input: independently captured generated pieces")
    content += image(outputs / "stage4/preview.png", here, "Stage 4: primary final construction (disconnected components packed separately)")
    content += image(here / "reference_cuts.jpg" if report.get("reference_preview") else inputs / "evaluation_only/reference_page.jpg", here, "Known source page and cut map (evaluation only)")+'</div>'
    if report.get("placement_overlay"):
        content += image(here / "placement_overlay.png", here, report["placement_overlay"]["caption"])
    quality = report["stages"].get("stage4", {}).get("assembly_quality")
    if quality:
        content += '<h2>Relative placement and precision sensitivity</h2><p>'+esc(quality["structural_visual_reason"])+'</p>'
        content += '<p>'+esc(quality["common_gauge"]["alignment"])+ ' Separate components have unavailable inter-component placement errors. Distances here use reference-page pixels; the contact table below uses native piece pixels.</p>'
        content += table(["Piece", "Anchor", "Placement p95 reference px", "Orientation error degrees", "Local scale ratio", "Unavailable reason"], [
            (esc(p["piece_id"]), esc(p.get("anchor_piece")), number(p.get("p95_reference_px")), number(p.get("local_orientation_error_degrees")), number(p.get("local_scale_ratio")), esc(p.get("reason", "")))
            for p in quality["common_gauge"]["per_piece"]])
        content += '<p>Tolerances below are precision sensitivity measurements, not additional structural success thresholds. The denominator includes unresolved contacts.</p>'
        content += table(["Native contact tolerance px", "Passing contacts", "All reference contacts"], [
            (str(r["tolerance_native_a_px"]), str(r["passing_contacts"]), str(r["total_contacts"])) for r in quality["contact_tolerance_sensitivity"]])
    content += '<h2>Contact evidence and precision diagnostics</h2><p>4px p95 tolerance on fixed native contact probes. This is separate from structural/visual assembly quality. A Stage 4 gap means a strictly passing direct proposal existed in its input; support conflicts or global constraints may prevent selection. The table localizes precision losses; it is not a proof of wrong neighbors or impossible assembly.</p>'
    rows = []
    for c in report.get("contact_diagnoses", []):
        boundary = c.get("stage2_boundary_coverage")
        rows.append((esc(c["piece_a"]+' / '+c["piece_b"]), esc(boundary["at_least_one_seam_covered_both_sides"] if boundary else "unavailable"),
            esc(c["physical_pair_retrieved"]), esc(c["physical_pair_retained_for_pose_search"]),
            number(c["archive_best_p95_px"]), number(c["working_best_p95_px"]), number((c.get("selected_path_comparison") or {}).get("selected_path_source_p95_px")),
            number(c["final"]["p95_native_a_px"] if c.get("final") else None), esc(c["diagnosis"])))
    content += table(["Contact", "Stage 2 arc IoU", "Pair retrieved", "Pose search", "Archive best px", "Pool best px", "Selected-path source px", "Final px", "4px localization"], rows)
    content += '<h2>Stage outputs and diagnostics</h2><p>'+link(here / "diagnosis.json", here, "Full machine-readable diagnosis")+'</p>'
    if report.get("existing_complete_candidate_audit"):
        content += '<h3>Existing complete alternatives</h3><p>This offline audit measures complete candidates already recorded during serving, including alternatives to partial or incorrect constructions. It creates no new predictions and does not replace the primary.</p><pre>'+esc(json.dumps(report["existing_complete_candidate_audit"], indent=2))+'</pre>'
        content += '<p>'+link(here / "support_intervals.svg", here, "Raw and active native support intervals")+'</p>'
        diagnostic_preview = report["existing_complete_candidate_audit"].get("diagnostic_preview")
        if diagnostic_preview:
            content += image(Path(diagnostic_preview["path"]), here, "DIAGNOSTIC ONLY: reference-selected existing candidate; primary prediction unchanged")
    for stage, data in report["stages"].items():
        content += f'<h3>{esc(stage)}: {esc(data.get("status"))}</h3>'
        if stage == "stage2":
            content += '<div class="visuals">'
            for piece in data.get("pieces", []):
                if piece.get("overlay_path"):
                    content += image(outputs / "stage2" / piece["overlay_path"], here, piece["piece_id"]+' boundary predictions')
            content += '</div>'
        content += '<details><summary>Metrics, counts, search decisions and limitations</summary><pre>'+esc(json.dumps(data, indent=2))+'</pre></details>'
        receipts = report.get("outputs", {}).get(stage, [])
        base = Path(report.get("artifact_roots", {}).get(stage, outputs / stage))
        if stage == "stage1":
            receipts = [{**r, "path": r["path"][len("stage1/"):]} for r in report.get("stage1_input_receipts", [])]
            base = inputs / "stage1"
        if receipts:
            content += '<details><summary>Every published output file and checksum</summary>'
            content += table(["Artifact", "Bytes", "SHA-256"], [(link(base / r["path"], here, r["path"]), str(r["bytes"]), '<code>'+esc(r["sha256"])+'</code>') for r in receipts])+'</details>'
    content += '<details><summary>Commands, durations and process failures</summary><pre>'+esc(json.dumps(report.get("serving_progress", {}), indent=2))+'</pre></details>'
    content += '<details><summary>Logs and committed work caches</summary>'+table(["Artifact", "Bytes", "SHA-256"], [(link(outputs / r["path"], here, r["path"]), str(r["bytes"]), '<code>'+esc(r["sha256"])+'</code>') for r in report.get("logs_and_work_receipts", [])])+'</details>'
    if report.get("stage3_profile"):
        content += '<h2>Stage 3 profiler</h2><p>Inclusive function times overlap. The eight-piece equivalence result is independent of the earlier four-piece qualification.</p><pre>'+esc(json.dumps(report["stage3_profile"], indent=2))+'</pre>'
    if report.get("profile_artifacts"):
        content += '<details><summary>Original profiler outputs</summary>'+table(["Artifact", "Bytes", "SHA-256"], [(link(outputs / r["path"], here, r["path"]), str(r["bytes"]), '<code>'+esc(r["sha256"])+'</code>') for r in report["profile_artifacts"]])+'</details>'
    (here / "index.html").write_text(document(case["case_id"], content), encoding="utf-8")


def write_reports(reports, cohort_root, run_root):
    here = run_root / "report"
    here.mkdir(exist_ok=True)
    rows = [brief(r) for r in reports]
    cohort_file = cohort_root / "benchmark.json"
    cohort_metadata = read_json(cohort_file) if cohort_file.is_file() else {}
    exposed = set(cohort_metadata.get("selection", {}).get("previously_reviewed_case_ids", []))
    for row in rows:
        row["previously_reviewed_before_benchmark"] = row["case_id"] in exposed
    aggregate = {"schema_version": "stage4.benchmark-summary.v1", "cases": rows, "by_size": [],
        "diagnosis_counts": dict(sum((Counter(r.get("diagnosis_counts", {})) for r in reports), Counter())),
        "limitations": "Declared development cohort with generated observed masks; Stage 1 bypassed. Not an independent holdout capacity estimate. Failed/unpublished outputs have null accuracy, not zero. Connectivity and neighbor identity need placement/visual review. The 4px tests measure seam precision separately from recognizable construction. Only the committed primary layout is evaluated."}
    for size in sorted({r["pieces"] for r in rows}):
        group = [r for r in rows if r["pieces"] == size]
        aggregate["by_size"].append({"pieces": size, "requested_cases": len(group), "predictions_published": sum(r["stage4_status"] == "complete" for r in group),
            "construction_review_counts": dict(Counter(r["structural_visual_assessment"] or "unreviewed" for r in group)),
            "new_cases_review_counts": dict(Counter(r["structural_visual_assessment"] or "unreviewed" for r in group if not r["previously_reviewed_before_benchmark"])),
            "strict_passes": sum(r["strict_pass"] is True for r in group), "diagnostic_errors": sum(r["diagnostic_errors"] for r in group),
            "mean_contact_recall_published_only": np_mean([r["implied_contact_recall"] for r in group if r["implied_contact_recall"] is not None]),
            "mean_stage2_containment_available_only": np_mean([r["stage2_both_side_containment_recall"] for r in group if r["stage2_both_side_containment_recall"] is not None]),
            "mean_stage3_pool_recall_available_only": np_mean([r["stage3_working_contact_recall"] for r in group if r["stage3_working_contact_recall"] is not None])})
    atomic_json(here / "summary.json", aggregate)
    with (here / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
    content = f'<h1>Paper reconstruction development benchmark</h1><p>{len(rows)} declared puzzles in this run; '+', '.join(f'{sum(r["pieces"] == size for r in rows)} at {size} pieces' for size in sorted({r["pieces"] for r in rows}))+'.</p>'
    content += '<div class="warning">'+esc(aggregate["limitations"])+'</div>'
    content += '<p>'+link(here / "summary.csv", here, "CSV summary")+' · '+link(here / "summary.json", here, "JSON summary")+' · '+link(run_root / "request.json", here, "Pinned run identity")+' · '+link(run_root / "progress.json", here, "Progress and process logs")+'</p>'
    content += '<h2>Construction review</h2><p>The practical target is a recognizable, correctly assembled page with small seam errors allowed. Inspect each primary preview for misplaced pieces, rotation, gaps and overlaps. A connected graph alone is insufficient. Orientation and scale below use one common gauge; no per-piece alignment.</p>'
    content += table(["Puzzle", "Components", "Correct selected neighbor IDs", "Max orientation error (degrees)", "Scale range", "Median contact p95 (native px)", "Prediction attempt (seconds)", "Visual assessment"], [
        (link(here / r["case_id"] / "index.html", here, r["case_id"]), esc(r["assembled_component_sizes"]),
         f'{esc(r["selected_true_neighbor_pairs"])}/{esc(r["selected_physical_pairs"])}', number(r["max_common_gauge_orientation_degrees"]),
         number(r["common_gauge_scale_min"])+" to "+number(r["common_gauge_scale_max"]), number(r["contact_error_p50_native_a_px"]),
         number(r["latest_prediction_attempt_wall_seconds"]), esc(r["structural_visual_assessment"] or "Review primary preview")) for r in rows])
    content += '<div class="visuals">'+''.join(image(run_root / "cases" / r["case_id"] / "stage4/preview.png", here, r["case_id"]+" — committed primary") for r in rows)+'</div>'
    content += table(["Pieces", "Requested", "Predictions", "Strict passes", "Stage 2 containment", "Stage 3 pool recall", "Final contact recall", "Diagnostic errors"], [(str(r["pieces"]), str(r["requested_cases"]), str(r["predictions_published"]), str(r["strict_passes"]), number(r["mean_stage2_containment_available_only"]), number(r["mean_stage3_pool_recall_available_only"]), number(r["mean_contact_recall_published_only"]), str(r["diagnostic_errors"])) for r in aggregate["by_size"]])
    content += '<h2>Puzzles</h2><label>Show size: <select id="size"><option value="all">All</option>'+''.join(f'<option>{s}</option>' for s in sorted({r["pieces"] for r in rows}))+'</select></label>'
    items = []
    for report, row in zip(reports, rows):
        cid = row["case_id"]
        case_report(report, cohort_root, run_root)
        items.append((f'<span data-size="{row["pieces"]}">'+link(here / cid / "index.html", here, cid)+'</span>', str(row["pieces"]), esc(row["difficulty"]),
            esc(row["stage2_status"]+' / '+row["stage3_status"]+' / '+row["stage4_status"]), esc(row["strict_pass"]),
            number(row["stage2_both_side_containment_recall"]), number(row["stage2_both_side_arc_iou_recall"]),
            number(row["implied_contact_recall"]), number(row["stage3_archive_contact_recall"]), number(row["stage3_working_contact_recall"]),
            esc(row["largest_correct_component"]), esc(row["false_joins"])))
    content += table(["Puzzle (open diagnosis)", "Pieces", "Difficulty", "Stage 2 / 3 / 4", "Strict 4px pass", "Stage 2 containment", "Stage 2 arc IoU", "Contact recall at 4px", "Archive recall at 4px", "Pool recall at 4px", "Component from correct 4px joins", "False neighbor IDs"], items)
    content += '<h2>Contact diagnosis totals</h2>'+table(["Diagnosis", "Contacts"], [(esc(k), str(v)) for k, v in aggregate["diagnosis_counts"].items()])
    content += "<script>document.querySelector('#size').addEventListener('change',e=>{document.querySelectorAll('tr').forEach(r=>{let s=r.querySelector('[data-size]');if(s)r.hidden=e.target.value!=='all'&&s.dataset.size!==e.target.value;});});</script>"
    (here / "index.html").write_text(document("Pipeline benchmark", content), encoding="utf-8")


def np_mean(values):
    return sum(values)/len(values) if values else None
