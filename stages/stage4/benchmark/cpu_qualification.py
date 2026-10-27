"""User-run CPU Stage 3/4 slice: CPU-v1/v3 exactness and CUDA comparison."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from paper_reconstruction_contracts import load_bundle
from paper_reconstruction_stage4.outputs import read_result
from .candidate import THREADS,CANDIDATE_SHA,RECIPE_SHA
from .common import WORKSPACE,atomic_json,digest,file_hash,load_cohort,read_json
from .qualified_v3 import UPSTREAM,authenticate,require,semantic_evidence
from .runner import execute
from .validation import safe_output
from .v3_integration import verify_snapshot

CPU_ENV={**THREADS,"CUDA_VISIBLE_DEVICES":""}


def cpu_profile(folder,case,variant,manifest,baseline=None):
    p=read_json(Path(folder)/"profile.json");r=p["request"]
    require(all(p.get(k)==v for k,v in {"status":"profile_complete","scope":"full","sources_verified_unchanged":True,
        "source_validation_status":"passed","ready_bundle_published":True,"training_performed":False,"evaluation_truth_opened":False,
        "runtime_policy_changed":False}.items()),"CPU profile is incomplete/unverified")
    require(r["device"]=="cpu" and r["scope"]=="full" and r["puzzle_id"]==case["case_id"] and r["cprofile"] is False and
        r["candidate_runtime"]==manifest and Path(r["output"]).resolve()==Path(folder).resolve(),"CPU profile device/runtime/output binding differs")
    require(p["inference_result"]["device"]=="cpu","Profile inference actually used a non-CPU backend")
    for key in ("stage1","stage2"):
        require(Path(r[key]).resolve()==Path(case[key]).resolve(),"CPU source observation binding differs")
    if variant=="v1":
        require(r["baseline"] is None and p["comparison"]=={"status":"not_requested","equal":None},"CPU v1 must be a fresh CPU baseline")
    else:
        require(baseline is not None and Path(r["baseline"]).resolve()==Path(baseline).resolve() and
                p["comparison"].get("equal") is True and p["comparison"].get("status")=="passed" and
                p["comparison"].get("archive_equal") is True and p["comparison"]["archive"]["different_groups"]==0,"CPU v3 did not qualify against CPU v1")
    data=load_bundle(Path(folder)/"evidence",full=False)
    require(data.manifest["provenance"]["inference_signature"]==r["expected_inference_signature"] and data.manifest["puzzle_id"]==case["case_id"],"CPU native evidence/profile differs")
    return p


def machine():
    processor=os.environ.get("PROCESSOR_IDENTIFIER",platform.processor());ram=None
    if os.name=="nt":
        import ctypes
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor=winreg.QueryValueEx(key,"ProcessorNameString")[0]
        except OSError:pass
        class MemoryStatus(ctypes.Structure):
            _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong),*[(k,ctypes.c_ulonglong) for k in
                ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","extended")]]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):ram=int(status.total_phys)
    return {"platform":platform.platform(),"processor":processor,"total_ram_bytes":ram,
            "logical_cores":os.cpu_count(),"physical_cores":None,"physical_core_note":"Not inferred from logical core count",
            "parent_python":platform.python_version(),"thread_environment":CPU_ENV,"cache_state":"Each profiler invocation uses fresh inference caches; OS disk cache is not flushed"}


def inspect_v1():
    command=[str(UPSTREAM/".venv/Scripts/python.exe"),"-B",str(WORKSPACE/"scripts/inspect_stage3_candidate.py"),str(UPSTREAM)]
    done=subprocess.run(command,cwd=WORKSPACE,capture_output=True,text=True,check=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
    v=json.loads(done.stdout)
    require(v["manifest"]["candidate_implementation_sha256"]==CANDIDATE_SHA and v["manifest"]["manifest_sha256"]==RECIPE_SHA,"CPU comparison v1 runtime changed")
    return v


def compare_rankings(cpu_path,cuda_path):
    payloads=[read_json(Path(p)) for p in (cpu_path,cuda_path)]
    for p in payloads:require(p["payload_sha256"]==digest(p["rows"]),"Ranking cache payload checksum differs")
    maps=[{(r["query"],r["target"]):r["scores"] for r in p["rows"]} for p in payloads]
    common=set(maps[0])&set(maps[1]);models=sorted({k for rows in maps for scores in rows.values() for k in scores})
    reports={}
    for model in models:
        paired=[(a,b) for a,b in common if model in maps[0][a,b] and model in maps[1][a,b]]
        differences=[abs(maps[0][key][model]-maps[1][key][model]) for key in paired]
        queries=sorted({q for q,t in paired});overlaps=[]
        for query in queries:
            lists=[]
            for rows in maps:
                values=sorted(((scores[model],target) for (q,target),scores in rows.items() if q==query and model in scores),key=lambda x:(-x[0],x[1]))
                lists.append({target for score,target in values[:5]})
            overlaps.append(len(lists[0]&lists[1])/max(1,min(len(lists[0]),len(lists[1]))))
        reports[model]={"maximum_probability_difference":max(differences,default=None),"mean_probability_difference":float(np.mean(differences)) if differences else None,
                        "mean_top5_target_overlap":float(np.mean(overlaps)) if overlaps else None,"matched_rows":len(paired)}
    return {"models":reports,"cpu_rows":len(maps[0]),"cuda_rows":len(maps[1]),"common_pair_rows":len(common),
            "bitwise_equality_assumed":False,"score_type":"saved ranker probabilities, not logits"}


def profile_command(case,variant,runtime,output,baseline=None,device="cpu"):
    entry="profile_stage3.py" if variant=="v1" else "profile_stage3_visual.py"
    command=[UPSTREAM/".venv/Scripts/python.exe","-B",UPSTREAM/"scripts"/entry,"--candidate-runtime",runtime,
        "--stage1",case["stage1"],"--stage2",case["stage2"],"--models",UPSTREAM/"models","--device",device,
        "--scope","full","--no-cprofile","--puzzle-id",case["case_id"],"--output",output]
    if baseline is not None:command.extend(["--baseline",baseline])
    return list(map(str,command))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id",action="append");p.add_argument("--output",type=Path)
    p.add_argument("--check",action="store_true");p.add_argument("--plan-only",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--observation-label",default="first-observation",help="Describe this timing observation; does not flush OS caches")
    p.add_argument("--consumer-snapshot",type=Path,default=WORKSPACE/"controls/stage4_joint_v4_cached_v4")
    a=p.parse_args(argv);context=authenticate(wanted=a.case_id or ["174","917","341"]);v1=inspect_v1()
    snapshot=a.consumer_snapshot.resolve();frozen=verify_snapshot(snapshot)
    # Bind the current observations/models to the saved CUDA source through a
    # read-only context check, then bind both fresh CPU signatures. Device names
    # here select a signature only: --check performs no ranker/CUDA execution.
    preflights={}
    for case in context["cases"]:
        checks={}
        for name,variant,runtime,baseline,device in (
            ("cuda_context","v3",context["candidate"]["path"],case["native_equivalence"]["right"],"cuda"),
            ("v1","v1",v1["path"],None,"cpu"),("v3","v3",context["candidate"]["path"],None,"cpu")):
            command=profile_command(case,variant,runtime,WORKSPACE/"outputs/cpu_preflight_placeholder",baseline,device)
            command.append("--check")
            done=subprocess.run(command,cwd=WORKSPACE,capture_output=True,text=True,env={**os.environ,**CPU_ENV},check=False)
            require(done.returncode==0,"CPU/source read-only preflight failed: "+done.stderr)
            checks[name]=json.loads(done.stdout)
        preflights[case["case_id"]]=checks
    if a.check:
        print("CPU input/code preflight passed; inference and complete CPU pipeline qualification have not run.");return 0
    require(a.output is not None,"--output is required")
    output=a.output.resolve();safe_output(output,[UPSTREAM,WORKSPACE/"src",WORKSPACE/"scripts",WORKSPACE/"benchmark",WORKSPACE/"controls",
        Path(context["benchmark_manifest"]).parent,*[Path(c["historical_pool"]).parents[2] for c in context["cases"]]])
    cuda_rankings={}
    for case in context["cases"]:
        path=Path(case["source_profile"]).parent/"evidence.work/rankings.json"
        if path.is_file():cuda_rankings[case["case_id"]]={"path":str(path),"sha256":file_hash(path)}
    request={"schema_version":"stage4.cpu-slice-request.v1","output":str(output),"context":context,"v1_runtime":v1,"consumer_snapshot":str(snapshot),
        "preflights":preflights,"stage3_environment":v1["environment"],
        "saved_cuda_rankings":cuda_rankings,"observation_label":a.observation_label,
        "consumer_implementation":frozen["implementation"],"case_ids":[c["case_id"] for c in context["cases"]],"machine":machine(),
        "stage1_bypassed":True,"stage2_reused":True,"device":"cpu","cuda_visible_devices":"","fresh_pipeline":False,
        "timing":"Fresh CPU-v1 and CPU-v3 inference, pool and frozen consumer; compared with saved CUDA observations. Not controlled alternating or warm OS-cache trials.",
        "controllers":{str(p):file_hash(p) for p in (Path(__file__),WORKSPACE/"scripts/run_v3_cpu_qualification.py",WORKSPACE/"scripts/diagnose_v3_cpu.py",WORKSPACE/"benchmark/qualified_v3.py")}}
    request["request_sha256"]=digest(request)
    if a.plan_only:print(json.dumps(request,indent=2));return 0
    require(not output.exists() or a.resume,"Existing CPU run; use matching --resume or a new output")
    output.mkdir(parents=True,exist_ok=True);lock=output/"run.lock"
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),"w") as stream:stream.write(str(os.getpid()))
    state=read_json(output/"progress.json") if (output/"progress.json").exists() else {"phase":"predictions","cases":{}}
    try:
        if (output/"request.json").exists():require(read_json(output/"request.json")==request,"CPU run hardware/source/controller context changed")
        else:atomic_json(output/"request.json",request)
        state["phase"]="predictions";atomic_json(output/"progress.json",state)
        for case in context["cases"]:
            cid=case["case_id"];target=output/"cases"/cid;record=state["cases"].setdefault(cid,{})
            for variant,runtime in (("v1",v1),("v3",context["candidate"])):
                baseline=Path(record["v1"]["profile"])/"evidence" if variant=="v3" else None
                saved=record.get(variant,{})
                if saved.get("status")=="qualified":
                    folder=Path(saved["profile"])
                    require(file_hash(folder/"profile.json")==saved["profile_sha256"],"Completed CPU profile changed")
                    p=cpu_profile(folder,case,variant,runtime["manifest"],baseline)
                    require(p["request"]["expected_inference_signature"]==preflights[cid][variant]["expected_inference_signature"],"CPU inputs differ from preflight")
                    require(load_bundle(folder/"evidence",full=False).manifest["manifest_sha256"]==saved["native_manifest_sha256"],"Completed CPU native bundle changed")
                    continue
                attempts=target/variant;attempts.mkdir(parents=True,exist_ok=True)
                count=len(list(attempts.glob("attempt_*")))+1;folder=attempts/("attempt_"+str(count).zfill(3))
                command=profile_command(case,variant,runtime["path"],folder,baseline)
                print(cid,variant,": fresh CPU inference",flush=True)
                attempt=execute(command,target/"logs"/(variant+"_"+str(count)+".log"),env_overrides=CPU_ENV,echo_progress=True)
                record[variant]={"status":"failed","profile":str(folder),"attempt":attempt};atomic_json(output/"progress.json",state)
                require(attempt["returncode"]==0,"CPU inference/qualification failed; inspect profiler log")
                p=cpu_profile(folder,case,variant,runtime["manifest"],baseline)
                require(p["request"]["expected_inference_signature"]==preflights[cid][variant]["expected_inference_signature"],"CPU inputs differ from preflight")
                record[variant].update(status="qualified",profile_sha256=file_hash(folder/"profile.json"),
                    native_manifest_sha256=load_bundle(folder/"evidence",full=False).manifest["manifest_sha256"])
                rankings=folder/"evidence.work/rankings.json"
                if rankings.is_file():record[variant]["rankings_sha256"]=file_hash(rankings)
                atomic_json(output/"progress.json",state)
            bundle=target/"stage3"
            if not (bundle/"bundle.json").exists():
                cmd=[sys.executable,"-B",WORKSPACE/"scripts/replay_observable_pool.py","--stage3-root",UPSTREAM,
                     "--source",Path(record["v3"]["profile"])/"evidence","--output",bundle,"--expected-script-sha256",context["pool_script_sha256"]]
                attempt=execute(cmd,target/"logs/pool.log",env_overrides=CPU_ENV,echo_progress=True)
                record["pool_attempt"]=attempt;atomic_json(output/"progress.json",state)
                require(attempt["returncode"]==0,"CPU pool replay failed")
            consumer=target/"stage4"
            if not (consumer/"manifest.json").exists():
                cmd=[sys.executable,"-B",WORKSPACE/"scripts/run_source_snapshot.py",snapshot,"solve","--bundle",bundle,
                     "--config",snapshot/"configs/joint_coarse_fast_development.json","--output",consumer,"--resume"]
                attempt=execute(cmd,target/"logs/consumer.log",env_overrides=CPU_ENV,echo_progress=True)
                record["consumer_attempt"]=attempt;atomic_json(output/"progress.json",state)
                require(attempt["returncode"]==0,"CPU consumer failed")
            r=read_result(consumer);data=load_bundle(bundle,full=False)
            require(data.manifest["provenance"]["source_manifest_sha256"]==record["v3"]["native_manifest_sha256"] and
                data.manifest["provenance"]["implementation"]["stage3_pool_replay_tools.py"]==context["pool_script_sha256"],"CPU replay source/policy differs")
            require(r["input"]["manifest_sha256"]==data.manifest["manifest_sha256"] and r["implementation"]==frozen["implementation"] and
                r["configuration"]==read_result(case["controls"]["coarse"]["path"])["configuration"],"CPU consumer/evidence/configuration differs")
            record.update(status="predicted",result_sha256=file_hash(consumer/"result.json"));atomic_json(output/"progress.json",state)
        state["phase"]="diagnostics";atomic_json(output/"progress.json",state)
        state["diagnostics"]=execute([sys.executable,"-B",WORKSPACE/"scripts/diagnose_v3_cpu.py","--run",output],output/"diagnostics.log",env_overrides=CPU_ENV,echo_progress=True)
        state["phase"]="finished" if state["diagnostics"]["returncode"]==0 else "diagnostics_failed"
        atomic_json(output/"progress.json",state);print("CPU slice report:",output/"report/index.html",flush=True)
        return int(state["phase"]!="finished")
    except KeyboardInterrupt:
        state["phase"]="interrupted";atomic_json(output/"progress.json",state);return 130
    except (ValueError,OSError,KeyError) as error:
        state.update(phase="failed",error=str(error));atomic_json(output/"progress.json",state);raise
    finally:lock.unlink(missing_ok=True)


def diagnose(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args(argv);root=a.run.resolve()
    req,state=read_json(root/"request.json"),read_json(root/"progress.json")
    require(state["phase"] in {"diagnostics","finished","diagnostics_failed"},"CPU serving barrier not reached")
    require(req["request_sha256"]==digest({k:v for k,v in req.items() if k!="request_sha256"}),"CPU request checksum differs")
    require(all(state["cases"].get(cid,{}).get("status")=="predicted" for cid in req["case_ids"]),"CPU predictions incomplete")
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from .report import document,esc,table,number,image
    cohort_root,cohort=load_cohort(req["context"]["benchmark_manifest"])
    rows,cells,visuals=[],[],[];report=root/"report";report.mkdir(exist_ok=True)
    for case in req["context"]["cases"]:
        cid=case["case_id"];record=state["cases"][cid];folder=root/"cases"/cid
        profiles=[]
        for variant in ("v1","v3"):
            require(file_hash(Path(record[variant]["profile"])/"profile.json")==record[variant]["profile_sha256"],"CPU profile changed after qualification")
            manifest=req["v1_runtime"]["manifest"] if variant=="v1" else req["context"]["candidate"]["manifest"]
            baseline=Path(record["v1"]["profile"])/"evidence" if variant=="v3" else None
            profile=cpu_profile(record[variant]["profile"],case,variant,manifest,baseline)
            require(load_bundle(Path(record[variant]["profile"])/"evidence",full=False).manifest["manifest_sha256"]==record[variant]["native_manifest_sha256"],"CPU native result changed")
            profiles.append(profile)
        cpu=load_bundle(Path(record["v3"]["profile"])/"evidence",full=False);cuda=load_bundle(case["native_bundle"],full=False)
        require(profiles[1]["comparison"]["equal"] is True,"CPU v1/v3 exact qualification failed")
        native_equal=semantic_evidence(cpu)==semantic_evidence(cuda)
        cpu_pool=load_bundle(folder/"stage3",full=False);saved_pool=load_bundle(case["historical_pool"],full=False)
        r=read_result(folder/"stage4");old=read_result(case["controls"]["coarse"]["path"])
        require(file_hash(folder/"stage4/result.json")==record["result_sha256"],"CPU prediction changed")
        c=next(c for c in cohort["cases"] if c["case_id"]==cid);ref=read_json(cohort_root/c["path"]/"evaluation_only/reference_geometry.json")
        qualities=[]
        for result in (old,r):
            reference={**ref,"input_manifest_sha256":result["input"]["manifest_sha256"]}
            qualities.append(assembly_quality(result,reference,evaluate(result,reference)))
        cpu_ids={e["id"] for e in cpu_pool.edges};cuda_ids={e["id"] for e in saved_pool.edges}
        row={"case_id":cid,"cpu_v1_v3_exact_qualification":True,"native_cpu_cuda_semantically_equal":native_equal,
            "cpu_inference_seconds":[p["timing"]["inference_wall_seconds"] for p in profiles],
            "cuda_v3_saved_seconds":read_json(case["source_profile"])["timing"]["inference_wall_seconds"],
            "cpu_pool_subprocess_seconds":record.get("pool_attempt",{}).get("elapsed_seconds"),
            "cpu_consumer_subprocess_seconds":record.get("consumer_attempt",{}).get("elapsed_seconds"),
            "cpu_consumer_before_publication_seconds":r["resources"]["elapsed_before_publication_seconds"],
            "working_geometry_id_overlap":len(cpu_ids&cuda_ids),"working_geometry_id_union":len(cpu_ids|cuda_ids),
            "identity_overlap_is_not_tolerance_pose_recall":True,"pool_weight_quantiles_cpu":np.quantile([e["weight"] for e in cpu_pool.edges],[0,.5,1]).tolist() if cpu_pool.edges else [],
            "pool_weight_quantiles_cuda":np.quantile([e["weight"] for e in saved_pool.edges],[0,.5,1]).tolist() if saved_pool.edges else [],
            "quality_saved_cuda":qualities[0],"quality_cpu":qualities[1],"cpu_memory":profiles[1]["memory_at_report"],"visual_review":None}
        if cid in req["saved_cuda_rankings"]:
            saved=req["saved_cuda_rankings"][cid];cpu_rankings=Path(record["v3"]["profile"])/"evidence.work/rankings.json"
            require(file_hash(Path(saved["path"]))==saved["sha256"] and file_hash(cpu_rankings)==record["v3"]["rankings_sha256"],"Backend ranking evidence changed")
            row["ranker_backend_comparison"]=compare_rankings(cpu_rankings,saved["path"])
        rows.append(row);cells.append((esc(cid),esc(native_equal),*[number(t) for t in row["cpu_inference_seconds"]],esc(qualities[1]["component_sizes"]),number(qualities[1]["selected_neighbor_identity_precision"])))
        visuals.append('<h2>'+esc(cid)+'</h2><div class="visuals">'+image(Path(case["controls"]["coarse"]["path"])/"preview.png",report,"Saved CUDA-evidence cold construction")+image(folder/"stage4/preview.png",report,"Fresh CPU-evidence cold construction")+'</div>')
    atomic_json(report/"summary.json",{"schema_version":"stage4.cpu-slice-comparison.v1","request_sha256":req["request_sha256"],"machine":req["machine"],"cases":rows,
        "stage1_bypassed":True,"stage2_reused":True,"complete_cpu_product_qualification":False,"cpu_gpu_bitwise_equality_assumed":False})
    content='<h1>CPU Stage 3/4 slice qualification</h1><p>Fresh CPU-v1 and CPU-v3; saved CUDA observations compared separately. Stage 1 bypassed, Stage 2 reused. This does not qualify the complete CPU product or real capture path. Inspect primary constructions, memory and numeric differences.</p>'
    content+=table(["Case","CPU/CUDA native equality","CPU v1 seconds","CPU v3 seconds","CPU components","True-neighbor fraction"],cells)+''.join(visuals)
    (report/"index.html").write_text(document("CPU slice qualification",content),encoding="utf-8")
    atomic_json(root/"qualification_receipt.json",{"request_sha256":req["request_sha256"],"summary_sha256":file_hash(report/"summary.json"),
        "cpu_v1_v3_execution_equal":True,"cpu_stage3_stage4_slice_run":True,"complete_cpu_product_qualified":False})
    return 0
