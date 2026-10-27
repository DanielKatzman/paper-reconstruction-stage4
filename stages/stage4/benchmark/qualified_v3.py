"""Authenticate the reviewed v3 qualification without importing producer code."""
import json
import os
from pathlib import Path
import subprocess
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.outputs import read_result
from .common import DESKTOP, WORKSPACE, digest, file_hash, load_cohort, read_json, verify_files

IMPLEMENTATION = "900b093fbfb26bf82c48978132c7ba9af92ba95deca07e138b019ba78a29a548"
RECIPE = "6ee0427f8fd85cee76de78a185e19fd895713f309fb23c96012fce24f7b9d058"
RECEIPT_SHA = "80bb8640ba0127f4bb3b781c94eed852163942162500079f760237681f4a6405"
POOL_SHA = "75d1238af3b2ba59663d5e56a500e34bde10157be39148596bd6406934b88fc0"
CONSUMER = "f1018f904cfb387185d39fef9591deb856062dc3cadd9042882e3606d413c08f"
UPSTREAM = DESKTOP / "paper-reconstruction-stage3"
DEFAULT_RECEIPT = UPSTREAM / "docs/STAGE4_VISUAL_V3_QUALIFIED_RECEIPT_2026-10-23.json"
EXCLUDED_METADATA = {"identity", "manifest_sha256", "provenance"}
ARMS = {"coarse":("joint_coarse_fast_development.json", "c81be8a49a45589e68e064c025351e5ca41ebb85290adab67f97700038b08f41"),
        "robust":("joint_robust_development.json", "9dfd83ea26f02a9e64c4cb8104c92676a5931dc6bf59568a708bb2d1964b51f8")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_json(path, expected=None):
    path = Path(path)
    if expected:
        require(file_hash(path) == expected, "Receipt checksum changed: " + str(path))
    return read_json(path)


def checked_request(path):
    value = checked_json(path)
    require(value["request_sha256"] == digest({k:v for k,v in value.items() if k != "request_sha256"}), "Request checksum mismatch")
    for name, sha in value["controller_files"].items():
        checked_json_file = Path(name)
        require(file_hash(checked_json_file) == sha, "Bound controller changed: " + name)
    return value


def inspect_runtime(stage3=UPSTREAM, runtime=None):
    stage3 = Path(stage3).resolve()
    runtime = Path(runtime or stage3/"generated/stage3_candidates/visual-reuse-v3_900b093fbfb26bf8_6ee0427f").resolve()
    require(runtime.is_dir(), "Qualified v3 runtime is missing")
    command = [str(stage3/".venv/Scripts/python.exe"), "-B", str(stage3/"scripts/stage3_visual_runtime.py"), "--inspect", str(runtime)]
    done = subprocess.run(command, cwd=stage3, capture_output=True, text=True,
                          env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"}, check=False)
    require(done.returncode == 0, "Read-only v3 inspection failed: " + done.stderr.strip())
    value = json.loads(done.stdout)
    require(value.get("status") == "candidate_verified" and Path(value["runtime_root"]).resolve() == runtime,
            "Unexpected v3 inspector output")
    m = value["manifest"]
    require(m.get("variant") == "visual-reuse-v3" and m.get("candidate_implementation_sha256") == IMPLEMENTATION and
            m.get("manifest_sha256") == RECIPE and m.get("runtime_policy_changed") is False and m.get("frozen_models_changed") is False,
            "V3 runtime is not the reviewed implementation/recipe")
    return {"status":"candidate_verified", "path":str(runtime), "manifest":m,
            "qualification":"Code inspection; completed-case qualification is bound separately", "command":command}


def semantic_evidence(bundle):
    """Every non-execution manifest field plus the exact ordered pool payload."""
    return {"manifest":{k:v for k,v in bundle.manifest.items() if k not in EXCLUDED_METADATA}, "pool":bundle.pool}


def compare_evidence(left, right):
    a, b = load_bundle(left, full=False), load_bundle(right, full=False)
    first, second = semantic_evidence(a), semantic_evidence(b)
    require(first == second, "Consumer evidence differs beyond execution identity/provenance")
    return {"schema_version":"stage4.semantic-evidence-equivalence.v1", "equal":True,
        "left":str(a.root), "right":str(b.root), "left_manifest_sha256":a.manifest["manifest_sha256"],
        "right_manifest_sha256":b.manifest["manifest_sha256"], "semantic_sha256":digest(first),
        "excluded_manifest_fields":sorted(EXCLUDED_METADATA), "counts":a.manifest["counts"],
        "validation":"SDK schema/inventory/pool/assets/archive byte receipts; qualified archive object comparison authenticated separately"}


def validate_profile(profile, folder, case, manifest, policy):
    """Qualification context, deliberately distinct from fresh-pipeline validation."""
    req = profile["request"]
    require(all(profile.get(k) == v for k,v in {
        "status":"profile_complete", "scope":"full", "sources_verified_unchanged":True,
        "source_validation_status":"passed", "ready_bundle_published":True,
        "evaluation_truth_opened":False, "training_performed":False, "runtime_policy_changed":False}.items()),
        "Incomplete/unverified imported profile")
    require(profile.get("feature_image_cache") is not None and profile.get("visual_reuse_counters") is not None and
            profile.get("visual_preparation_pairs") is not None,"Imported profile lacks qualified reuse diagnostics")
    require(req["candidate_runtime"] == manifest and req["device"] == "cuda" and req["cprofile"] is False and
            req["fresh_inference_caches"] is True and req["puzzle_id"] == case["case_id"] and req["scope"] == "full" and
            req["model_policy_sha256"] == policy, "Imported profile runtime/device/instrumentation differs")
    require(Path(req["output"]).resolve() == Path(folder).resolve(), "Imported profile output binding differs")
    for key in ("baseline","stage1","stage2"):
        require(Path(req[key]).resolve() == Path(case[key]).resolve(), "Imported profile source binding differs: " + key)
    c = profile["comparison"]
    require(c.get("status") == "passed" and c.get("equal") is True and c.get("archive_equal") is True and
            c["archive"]["equal"] is True and c["archive"]["different_groups"] == 0 and
            c.get("working_pool_payload_equal") is True and c.get("assets_equal") is True and
            set(c["excluded_manifest_metadata"]) == EXCLUDED_METADATA, "Imported profile did not pass exact evidence comparison")


def select_cases(cases, wanted):
    if not wanted:
        return cases
    selected = []
    for name in wanted:
        matches = [c for c in cases if c["case_id"] == name or (name.isdigit() and int(c["case_id"].split("_")[1]) == int(name))]
        require(len(matches) == 1, "Unknown/ambiguous qualified case: " + name)
        require(matches[0] not in selected, "Duplicate qualified case: " + name)
        selected.append(matches[0])
    return selected


def authenticate(receipt=DEFAULT_RECEIPT, wanted=None):
    receipt = Path(receipt).resolve()
    h = checked_json(receipt, RECEIPT_SHA)
    runtime = inspect_runtime(runtime=h["candidate_runtime"])
    require(runtime["manifest"] == h["candidate_manifest"], "Qualification candidate differs")
    require(file_hash(Path(h["integration"]["pool_script"])) == POOL_SHA == h["integration"]["pool_script_sha256"], "Pool implementation changed")
    model_path = Path(h["models"]["root"])/"alpha.json"
    models = checked_json(model_path,h["models"]["alpha_json_file_sha256"])
    require(models["manifest_sha256"] == h["models"]["policy_manifest_sha256"], "Model/policy differs")
    for ranker in models["rankers"]:
        require(file_hash(model_path.parent/ranker["checkpoint"]["path"]) == ranker["checkpoint"]["sha256"], "Frozen checkpoint changed")
    roots = [(Path(h["qualification"]["root"]),h["qualification"]["files"],None),
             (Path(h["twelve_piece_qualification"]["root"]),{"qualification.json":h["twelve_piece_qualification"]["qualification_json_sha256"]},h["twelve_piece_qualification"]["case_id"])]
    candidates = []
    for root, files, only in roots:
        for name,sha in files.items():
            require(file_hash(root/name) == sha, "Qualification file changed: " + name)
        q, req, progress = checked_json(root/"qualification.json"), checked_request(root/"request.json"), checked_json(root/"progress.json")
        require(q["request_sha256"] == req["request_sha256"] and q["status"] == "qualified_selected_cases" and
                progress["phase"] == "complete" and req["plan"]["candidate_manifest"] == runtime["manifest"], "Qualification request/progress differs")
        for case in req["plan"]["cases"]:
            if only and case["case_id"] != only:
                continue
            row = next(r for r in q["cases"] if r["case_id"] == case["case_id"])
            status = progress["cases"][case["case_id"]]
            candidates.append({"case_id":case["case_id"], "case":case,"row":row,"status":status,
                               "qualification_root":str(root),"qualification_request_sha256":req["request_sha256"],
                               "benchmark_file":req["plan"]["benchmark_file"], "benchmark_sha256":req["plan"]["benchmark_sha256"]})
    selected = select_cases(candidates,wanted)
    cohort_root, cohort = load_cohort(selected[0]["benchmark_file"],serving_only=True)
    result = []
    for item in selected:
        case, row, status = item["case"],item["row"],item["status"]
        cid = case["case_id"]
        require(item["benchmark_sha256"] == cohort["benchmark_sha256"], "Imported cohort differs")
        cohort_case = next(c for c in cohort["cases"] if c["case_id"] == cid)
        verify_files(cohort_root/cohort_case["path"], [f for f in cohort_case["files"] if f["path"].startswith("stage1/")])
        require(Path(case["stage1"]).resolve() == (cohort_root/cohort_case["path"]/"stage1").resolve(), "Qualified observations differ from cohort")
        folder = Path(status["profile"]).resolve()
        require(folder.is_relative_to(Path(item["qualification_root"])/"cases"/cid) and folder == Path(row["profile"]).resolve(), "Profile escaped qualified case")
        require(status["status"] == row["status"] == "complete" and file_hash(folder/"profile.json") == status["profile_sha256"] == row["profile_sha256"], "Qualified profile changed")
        profile = checked_json(folder/"profile.json")
        validate_profile(profile,folder,case,runtime["manifest"],models["manifest_sha256"])
        native = load_bundle(folder/"evidence",full=False)
        require(native.manifest["puzzle_id"] == cid and native.manifest["counts"]["pieces"] == cohort_case["pieces"] and
                native.manifest["provenance"]["inference_signature"] == profile["request"]["expected_inference_signature"], "Profile/evidence binding differs")
        for piece in native.manifest["pieces"].values():
            require(piece["piece_id"] in native.entries, "Physical inventory mismatch")
        old_profile = checked_json(case["baseline_profile"],case["baseline_profile_sha256"])
        require(old_profile["request"]["candidate_runtime"]["candidate_implementation_sha256"] == runtime["manifest"]["comparison_base_implementation_sha256"], "Native comparison is not v1")
        equality = compare_evidence(native.root,case["baseline"])
        native_root = Path(case["baseline"]).parent.parent
        old_pool = native_root/"stage3"
        old_pool_data = load_bundle(old_pool,full=False)
        require(old_pool_data.manifest["provenance"]["source_manifest_sha256"] == equality["right_manifest_sha256"] and
                old_pool_data.manifest["provenance"]["implementation"]["stage3_pool_replay_tools.py"] == POOL_SHA,
                "Historical replay does not bind the native source/policy")
        controls = {}
        for arm in ARMS:
            name = ("validation8x10_coarse_v4" if arm == "coarse" else "validation8x10_cached_v4") if cohort_case["pieces"] == 8 else ("validation12_simple_coarse_v4" if arm == "coarse" else "validation12_simple_refined_v4")
            path = WORKSPACE/"outputs"/name/"cases"/cid/"stage4"
            control = read_result(path)
            require(control["input"]["manifest_sha256"] == old_pool_data.manifest["manifest_sha256"] and
                    control["implementation"]["sha256"] == CONSUMER, "Historical consumer binding differs")
            require(file_hash(path.parents[2]/"stage4_config.json") == ARMS[arm][1], "Control configuration changed")
            controls[arm] = {"path":str(path),"result_sha256":file_hash(path/"result.json"),"run_id":control["run_id"]}
        result.append({"case_id":cid,"pieces":cohort_case["pieces"],"native_bundle":str(native.root),
            "native_manifest_sha256":native.manifest["manifest_sha256"],"native_equivalence":equality,
            "qualification_root":item["qualification_root"],"qualification_request_sha256":item["qualification_request_sha256"],
            "source_profile":str(folder/"profile.json"),"source_profile_sha256":row["profile_sha256"],
            "stage1":case["stage1"],"stage2":case["stage2"],"historical_pool":str(old_pool),"controls":controls,
            "upstream_device":"cuda","stage2_reused":True,"stage3_reused":True})
    return {"receipt":str(receipt),"receipt_sha256":RECEIPT_SHA,"candidate":runtime,
            "benchmark_manifest":str(cohort_root/"benchmark.json"),"benchmark_sha256":cohort["benchmark_sha256"],
            "pool_script_sha256":POOL_SHA,"models_sha256":models["manifest_sha256"],"cases":result,
            "archive_comparison":"Authenticated completed exact-object comparisons plus current SDK byte receipts; object exhaustion not repeated at preflight"}
