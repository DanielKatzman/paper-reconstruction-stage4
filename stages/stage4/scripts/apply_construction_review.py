"""Bind explicit visual reviews to sealed primaries, then refresh HTML/CSV reports."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import atomic_json, digest, file_hash, load_cohort, read_json
from benchmark.report import write_reports
from paper_reconstruction_stage4.outputs import read_result

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
root = args.run.resolve()
request, progress = read_json(root/"request.json"), read_json(root/"progress.json")
if progress["phase"] != "finished":
    raise ValueError("Apply reviews only after complete diagnostic publication")
cohort_root, cohort = load_cohort(request["benchmark_manifest"], serving_only=True)
if cohort["benchmark_sha256"] != request["benchmark_sha256"]:
    raise ValueError("Review cohort differs")
review_path = root/"manual_preview_review.json"
review = read_json(review_path)
allowed = {"recognizable_construction_with_seam_errors", "gross_construction_error", "incomplete_assembly", "uncertain_needs_further_review"}
reports = []
for cid in request["case_ids"]:
    path = root/"report"/cid/"diagnosis.json"
    report = read_json(path)
    entry = review["cases"].get(cid)
    if entry:
        if entry["assessment"] not in allowed:
            raise ValueError("Unknown construction review category")
        output = root/"cases"/cid/"stage4"
        result = read_result(output)
        if (result["run_id"] != entry["run_id"] or result["layouts"][0]["layout_sha256"] != entry["primary_layout_sha256"] or
                file_hash(output/"preview.png") != entry["preview_sha256"]):
            raise ValueError("Visual review belongs to another primary: "+cid)
        quality = report["stages"]["stage4"]["assembly_quality"]
        quality["structural_visual_assessment"] = entry["assessment"]
        quality["structural_visual_reason"] = entry["notes"]
        report["manual_construction_review"] = {**entry, "review_file_sha256": file_hash(review_path),
                                                "applied_by_script_sha256": file_hash(__file__)}
        atomic_json(path, report)
    reports.append(report)
write_reports(reports, cohort_root, root)
print("Construction reviews applied:", len(review["cases"]))
