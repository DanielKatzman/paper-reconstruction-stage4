"""Offline physical-pose coverage and reference-guided witness diagnosis."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmark.common import atomic_json, digest, file_hash, load_cohort, read_json, verify_files
from benchmark.diagnostics import pose_error
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.evidence import archive_edge
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.outputs import read_result
from evaluation.pose_coverage import physical_probes, full_pose_error, oracle_constructions

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--case-id", action="append", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--top", type=int, default=3)
p.add_argument("--oracle-seconds", type=float, default=20.)
args = p.parse_args()
root, output = args.run.resolve(), args.output.resolve()
request, progress = read_json(root/"request.json"), read_json(root/"progress.json")
if progress["phase"] not in {"diagnostics", "finished", "diagnostics_failed"}:
    raise ValueError("Prediction phase has not stopped")
if request["request_sha256"] != digest({k:v for k,v in request.items() if k != "request_sha256"}):
    raise ValueError("Source request checksum mismatch")
if set(args.case_id)-set(request["case_ids"]) or not 1 <= args.top <= 8 or not 0 < args.oracle_seconds <= 60:
    raise ValueError("Invalid source case or diagnostic limits")
cohort_root, cohort = load_cohort(request["benchmark_manifest"], verify=False)
if cohort["benchmark_sha256"] != request["benchmark_sha256"]:
    raise ValueError("Source cohort differs")
for protected in (cohort_root, root/"cases"):
    if output.is_relative_to(protected) or protected.is_relative_to(output):
        raise ValueError("Diagnostic output overlaps an input/prediction")
output.mkdir(parents=True,exist_ok=False)
for cid in args.case_id:
    if progress["cases"][cid]["status"] != "predicted":
        raise ValueError("Source case was not published")
    case = next(c for c in cohort["cases"] if c["case_id"] == cid)
    verify_files(cohort_root/case["path"],case["files"])
    reference_path = cohort_root/case["path"]/"evaluation_only/reference_geometry.json"
    reference = read_json(reference_path)
    primary_root = root/"cases"/cid/"stage4"
    primary = read_result(primary_root)
    config = SolverConfig.from_dict(primary["configuration"])
    problem = load_problem(root/"cases"/cid/"stage3",config)
    if primary["input"]["manifest_sha256"] != problem.manifest["manifest_sha256"]:
        raise ValueError("Primary evidence differs")
    reference["input_manifest_sha256"] = problem.manifest["manifest_sha256"]
    maps = {p:np.asarray(v["native_to_reference"],float) for p,v in reference["pieces"].items()}
    shapes = {p:v.shape for p,v in problem.pieces.items()}
    probes = physical_probes(problem)
    working_ids = {e.id for e in problem.edges}
    contacts = {(c["piece_a"],c["piece_b"]):c for c in reference["contacts"]}
    choices = {name:{pair:[] for pair in contacts} for name in ("working","archive")}
    all_edges, invalid = [], 0
    data = load_bundle(problem.root,full=False)
    for group in data.hypotheses():
        edge = archive_edge(group)
        all_edges.append(edge)
        pair = edge.a,edge.b
        if pair not in contacts:
            continue
        try:
            quality = full_pose_error(edge,probes,maps,shapes)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
            invalid += 1
            continue
        row = {"geometry_id":edge.id,"source_weight":edge.weight,"in_working_pool":edge.id in working_ids,
               "contact_p95_native_a_px":pose_error(edge.transform,contacts[pair]),**quality}
        for name in (("working","archive") if edge.id in working_ids else ("archive",)):
            values = choices[name][pair]
            values.append((edge,row))
            values.sort(key=lambda v:(v[1]["worst_fraction_native_crop_diagonal"],v[0].id))
            del values[args.top:]
    folder = output/cid
    folder.mkdir()
    report = {"schema_version":"stage4.offline-pose-coverage.v1","evaluation_truth_opened":True,
        "case_id":cid,"source_run_id":primary["run_id"],"source_result_sha256":file_hash(primary_root/"result.json"),
        "input_manifest_sha256":problem.manifest["manifest_sha256"],"reference_sha256":file_hash(reference_path),
        "code_sha256":{str(path):file_hash(path) for path in (Path(__file__),Path(__file__).parents[1]/"evaluation/pose_coverage.py")},
        "probe_semantics":"Up to 129 fixed foreground-mask pixels per native piece, bidirectional p95 normalized by each target crop diagonal. This supplements seam-only error; no success cutoff is chosen.",
        "invalid_reference_probe_mappings":invalid,"families":{},"serving_primary_unchanged":True}
    for name in ("working","archive"):
        family_problem = replace(problem,edges=problem.edges if name=="working" else all_edges,structural=None,initial_states=[])
        candidates = [(e,row["worst_fraction_native_crop_diagonal"]) for rows in choices[name].values() for e,row in rows]
        report["families"][name] = {"evidence_edges":len(family_problem.edges),
            "top_physical_pose_candidates":[{"pieces":list(pair),"candidates":[row for _,row in rows]} for pair,rows in choices[name].items()],
            "oracle":oracle_constructions(family_problem,config,candidates,primary,reference,folder/(name+"_oracle.png"),seconds=args.oracle_seconds)}
        atomic_json(folder/"coverage.json",report)
        print(cid,name,"oracle complete:",report["families"][name]["oracle"]["complete_final_physical_and_support_valid"],flush=True)
    if file_hash(primary_root/"result.json") != report["source_result_sha256"]:
        raise ValueError("Serving prediction changed")
