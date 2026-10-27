"""Opt-in import of completed v3 evidence, unchanged pool and frozen consumers."""
import argparse
import os
from pathlib import Path
import sys
from .candidate import THREADS
from .common import WORKSPACE, atomic_json, digest, file_hash, load_cohort, read_json
from .qualified_v3 import ARMS, CONSUMER, DEFAULT_RECEIPT, UPSTREAM, authenticate, compare_evidence, require
from .runner import execute
from .validation import safe_output
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.outputs import read_result


def source_inventory(context):
    values = {context["receipt"]:context["receipt_sha256"]}
    for case in context["cases"]:
        values[case["source_profile"]] = case["source_profile_sha256"]
        for root in (case["native_bundle"],case["historical_pool"]):
            values[str(Path(root)/"bundle.json")] = file_hash(Path(root)/"bundle.json")
    return values


def verify_sources(request):
    for name, sha in request["source_files"].items():
        require(file_hash(Path(name)) == sha, "Imported source changed: " + name)


def verify_snapshot(path):
    path = Path(path).resolve(); receipt = read_json(path/"receipt.json")
    for name,sha in receipt["files"].items():
        target = (path/name).resolve()
        require(target.is_relative_to(path) and file_hash(target) == sha, "Frozen consumer snapshot changed")
    require(receipt["implementation"]["sha256"] == CONSUMER,"Integration requires the unchanged cached-v4 consumer")
    for name,sha in ARMS.values():
        require(file_hash(path/"configs"/name) == sha,"Frozen integration configuration changed")
    return receipt


def request_for(context, output, snapshot, arms):
    receipt = verify_snapshot(snapshot)
    controllers = [Path(__file__),WORKSPACE/"benchmark/qualified_v3.py",WORKSPACE/"scripts/run_qualified_v3_integration.py",
                   WORKSPACE/"scripts/diagnose_qualified_v3.py",WORKSPACE/"scripts/replay_observable_pool.py",WORKSPACE/"scripts/run_source_snapshot.py"]
    value = {"schema_version":"stage4.qualified-v3-integration-request.v1", "import":context,
        "benchmark_manifest":context["benchmark_manifest"],"benchmark_sha256":context["benchmark_sha256"],
        "case_ids":[c["case_id"] for c in context["cases"]],"arms":arms,"output":str(output),
        "consumer_snapshot":str(snapshot),"consumer_implementation":receipt["implementation"],
        "configurations":{a:{"path":str(snapshot/"configs"/ARMS[a][0]),"sha256":ARMS[a][1]} for a in arms},
        "source_files":source_inventory(context),"controllers":{str(p):file_hash(p) for p in controllers},
        "stage1":"bypassed: generator observations", "stage2":"reused", "stage3":"reused CUDA qualification",
        "fresh_pipeline":False,"cpu_only_qualification":False,"cold_consumers":True,
        "timing_scope":"New observable pool replay and consumer solves; Stage 1/2/3 inference not repeated"}
    value["request_sha256"] = digest(value)
    return value


def validate_replay(case, bundle, pool_sha):
    data = load_bundle(bundle,full=False)
    provenance = data.manifest["provenance"]
    require(provenance["source_manifest_sha256"] == case["native_manifest_sha256"] and
            provenance["implementation"]["stage3_pool_replay_tools.py"] == pool_sha,
            "Imported replay source/policy differs")
    return compare_evidence(bundle,case["historical_pool"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,default=DEFAULT_RECEIPT)
    p.add_argument("--case-id",action="append")
    p.add_argument("--arms",choices=tuple(ARMS),nargs="+",default=["coarse"])
    p.add_argument("--snapshot",type=Path,default=WORKSPACE/"controls/stage4_joint_v4_cached_v4")
    p.add_argument("--output",type=Path)
    p.add_argument("--check",action="store_true")
    p.add_argument("--plan-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--diagnose-only",action="store_true")
    args = p.parse_args(argv)
    require(len(args.arms) == len(set(args.arms)),"Duplicate consumer arm")
    context = authenticate(args.receipt,args.case_id)
    snapshot = args.snapshot.resolve()
    verify_snapshot(snapshot)
    if args.check:
        print(__import__("json").dumps({"status":"v3_import_preflight_passed","case_ids":[c["case_id"] for c in context["cases"]],
            "runtime":context["candidate"]["path"],"serving_executed":False,"output_created":False,"cpu_only_qualified":False},indent=2))
        return 0
    require(args.output is not None,"--output is required unless --check")
    output = args.output.resolve()
    safe_output(output,[UPSTREAM,WORKSPACE/"src",WORKSPACE/"scripts",WORKSPACE/"benchmark",WORKSPACE/"controls",
                        Path(context["benchmark_manifest"]).parent,*[Path(c["historical_pool"]).parents[2] for c in context["cases"]],
                        *[Path(v["path"]).parents[2] for c in context["cases"] for v in c["controls"].values()]])
    request = request_for(context,output,snapshot,args.arms)
    if args.plan_only:
        print(__import__("json").dumps(request,indent=2));return 0
    require(not output.exists() or args.resume or args.diagnose_only,"Existing integration: use matching --resume or a new output")
    output.mkdir(parents=True,exist_ok=True)
    lock = output/"run.lock"
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),"w") as stream:
        stream.write(str(os.getpid()))
    progress_path = output/"progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"phase":"predictions","cases":{}}
    try:
        if (output/"request.json").exists():
            saved = read_json(output/"request.json")
            require(saved == request,"Integration inputs/code/configuration changed; choose a new output")
        else:
            require(not args.diagnose_only,"No integration to diagnose")
            atomic_json(output/"request.json",request)
        if not args.diagnose_only:
            progress["phase"] = "predictions";atomic_json(progress_path,progress)
            for index, case in enumerate(context["cases"],1):
                cid = case["case_id"]; target = output/"cases"/cid
                state = progress["cases"].setdefault(cid,{"arms":{},"attempts":[]})
                state["status"] = "running";atomic_json(progress_path,progress)
                verify_sources(request)
                bundle = target/"stage3"
                if not (bundle/"bundle.json").exists():
                    print(f"[{index}/{len(context['cases'])}] {cid}: unchanged observable pool",flush=True)
                    cmd = [sys.executable,"-B",WORKSPACE/"scripts/replay_observable_pool.py","--stage3-root",UPSTREAM,
                           "--source",case["native_bundle"],"--output",bundle,"--expected-script-sha256",context["pool_script_sha256"]]
                    attempt = execute(cmd,target/"logs/pool.log",env_overrides=THREADS,echo_progress=True)
                    state["attempts"].append({"stage":"pool",**attempt});atomic_json(progress_path,progress)
                    require(attempt["returncode"] == 0,"Observable pool replay failed; inspect case pool.log")
                equality = validate_replay(case,bundle,context["pool_script_sha256"])
                atomic_json(target/"semantic_equivalence.json",equality)
                for arm in args.arms:
                    result_path = target/arm/"stage4"
                    if not (result_path/"manifest.json").exists():
                        print(f"[{index}/{len(context['cases'])}] {cid}: frozen {arm} consumer",flush=True)
                        cmd = [sys.executable,"-B",WORKSPACE/"scripts/run_source_snapshot.py",snapshot,"solve","--bundle",bundle,
                               "--config",request["configurations"][arm]["path"],"--output",result_path,"--resume"]
                        attempt = execute(cmd,target/"logs"/(arm+".log"),env_overrides=THREADS,echo_progress=True)
                        state["attempts"].append({"stage":arm,**attempt});atomic_json(progress_path,progress)
                        require(attempt["returncode"] == 0,"Frozen consumer failed; inspect case arm log")
                    result = read_result(result_path);control = read_result(case["controls"][arm]["path"])
                    require(result["input"]["manifest_sha256"] == equality["left_manifest_sha256"] and
                            result["implementation"] == request["consumer_implementation"] and
                            result["configuration"] == control["configuration"],"Frozen imported consumer binding differs")
                    state["arms"][arm] = {"status":"predicted","path":str(result_path),"run_id":result["run_id"],
                        "result_sha256":file_hash(result_path/"result.json"),"input_manifest_sha256":equality["left_manifest_sha256"]}
                    atomic_json(progress_path,progress)
                state["status"] = "predicted";atomic_json(progress_path,progress)
            verify_sources(request)
        require(all(progress["cases"].get(c["case_id"],{}).get("status") == "predicted" for c in context["cases"]),"Serving barrier not reached")
        progress["phase"] = "diagnostics";atomic_json(progress_path,progress)
        progress["diagnostics"] = execute([sys.executable,"-B",WORKSPACE/"scripts/diagnose_qualified_v3.py","--run",output],
            output/"logs/diagnostics.log",env_overrides=THREADS,echo_progress=True)
        progress["phase"] = "finished" if progress["diagnostics"]["returncode"] == 0 else "diagnostics_failed"
        atomic_json(progress_path,progress)
        print("Integration report:",output/"report/index.html",flush=True)
        return int(progress["phase"] != "finished")
    except KeyboardInterrupt:
        progress["phase"] = "interrupted";atomic_json(progress_path,progress);return 130
    except (ValueError,OSError,KeyError) as error:
        progress.update(phase="failed",error=str(error));atomic_json(progress_path,progress)
        raise
    finally:
        lock.unlink(missing_ok=True)


def diagnose(argv=None):
    p = argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    args = p.parse_args(argv); root = args.run.resolve()
    request, progress = read_json(root/"request.json"),read_json(root/"progress.json")
    require(progress["phase"] in {"diagnostics","finished","diagnostics_failed"},"Serving prediction barrier not reached")
    require(request["request_sha256"] == digest({k:v for k,v in request.items() if k != "request_sha256"}),"Integration request checksum differs")
    verify_sources(request)
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from .report import document,esc,image,link,number,table
    cohort_root, cohort = load_cohort(request["benchmark_manifest"])
    require(cohort["benchmark_sha256"] == request["benchmark_sha256"],"Diagnostic cohort differs")
    rows, cells, details = [],[],[]
    for case in request["import"]["cases"]:
        cid = case["case_id"]; folder = root/"cases"/cid
        require(progress["cases"].get(cid,{}).get("status") == "predicted","Missing sealed imported case")
        equality = validate_replay(case,folder/"stage3",request["import"]["pool_script_sha256"])
        require(read_json(folder/"semantic_equivalence.json") == equality,"Equivalence receipt changed")
        cohort_case = next(c for c in cohort["cases"] if c["case_id"] == cid)
        reference = read_json(cohort_root/cohort_case["path"]/"evaluation_only/reference_geometry.json")
        for arm in request["arms"]:
            control_root = Path(case["controls"][arm]["path"]); output = folder/arm/"stage4"
            require(file_hash(control_root/"result.json") == case["controls"][arm]["result_sha256"],"Control result changed")
            old,new = read_result(control_root),read_result(output)
            require(old["input"]["manifest_sha256"] == equality["right_manifest_sha256"] and
                    new["input"]["manifest_sha256"] == equality["left_manifest_sha256"],"Cross-manifest comparison lacks bound equivalence")
            require(file_hash(output/"result.json") == progress["cases"][cid]["arms"][arm]["result_sha256"],"Published prediction changed")
            same = all(old["layouts"][0][k] == new["layouts"][0][k] for k in ("pieces","selected_joins","components","objective_score"))
            preview_same = file_hash(output/"preview.png") == file_hash(control_root/"preview.png")
            qualities = []
            for result in (old,new):
                ref = {**reference,"input_manifest_sha256":result["input"]["manifest_sha256"]}
                qualities.append(assembly_quality(result,ref,evaluate(result,ref)))
            inherited = None;review = control_root.parents[2]/"manual_preview_review.json"
            if same and preview_same and review.exists():
                item = read_json(review)["cases"].get(cid)
                if item and item["run_id"] == old["run_id"] and item["preview_sha256"] == file_hash(control_root/"preview.png"):
                    inherited = {"assessment":item["assessment"],"source":str(review),"reason":"Identical primary geometry and preview bytes to the reviewed control"}
            row = {"case_id":cid,"pieces":case["pieces"],"arm":arm,"semantic_equivalence":equality,
                "primary_geometry_identical":same,"preview_bytes_identical":preview_same,"quality_control":qualities[0],"quality_imported":qualities[1],
                "inherited_construction_review":inherited,"consumer_seconds":[v["resources"]["elapsed_before_publication_seconds"] for v in (old,new)],
                "proposals":[v["search"]["proposals"] for v in (old,new)],"termination":[v["termination"] for v in (old,new)],
                "evaluation_truth_opened":True,"source_profile":case["source_profile"],"cold_primary":True}
            rows.append(row)
            cells.append((esc(cid),esc(arm),esc(same),esc(qualities[1]["component_sizes"]),number(qualities[1]["selected_neighbor_identity_precision"]),
                          esc(inherited["assessment"] if inherited else "Review required"),number(row["consumer_seconds"][1])))
            details.append('<h2>'+esc(cid+' / '+arm)+'</h2><p>'+link(Path(case["source_profile"]),root/"report","Original qualified profile")+' · '+
                link(Path(case["stage2"]),root/"report","Reused Stage 2")+' · '+link(folder/"semantic_equivalence.json",root/"report","Semantic input comparison")+'</p>')
            details.append('<div class="visuals">'+image(control_root/"preview.png",root/"report","Historical primary")+image(output/"preview.png",root/"report","Imported v3 cold primary")+'</div>')
    report = root/"report";report.mkdir(exist_ok=True)
    atomic_json(report/"summary.json",{"schema_version":"stage4.qualified-v3-comparison.v1","request_sha256":request["request_sha256"],"cases":rows,
        "upstream_reused":True,"fresh_pipeline":False,"full_cpu_qualified":False})
    content = '<h1>Qualified v3 integration</h1><p>Original Stage 2/3 evidence reused. New unchanged observable pool replay and frozen cold consumers. This is not a fresh pipeline or CPU deployment measurement. Construction quality is separate from four-pixel precision.</p>'
    content += table(["Case","Arm","Same primary geometry","Components","True-neighbor fraction","Construction review","Consumer seconds"],cells)+''.join(details)
    (report/"index.html").write_text(document("Qualified v3 integration",content),encoding="utf-8")
    atomic_json(root/"integration_receipt.json",{"schema_version":"stage4.qualified-v3-integration.v1","request_sha256":request["request_sha256"],
        "source_receipt_sha256":request["import"]["receipt_sha256"],"cases":request["case_ids"],"arms":request["arms"],
        "semantic_equivalence_passed":True,"all_primary_geometry_identical":all(r["primary_geometry_identical"] for r in rows),
        "report_sha256":file_hash(report/"summary.json"),"cold_consumers":True,"upstream_inference_repeated":False,"full_cpu_qualified":False})
    return 0
