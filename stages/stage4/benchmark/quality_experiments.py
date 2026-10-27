"""Cold objective/search ablations on successfully imported v3 evidence."""
import argparse
import os
from pathlib import Path
import sys
from .candidate import THREADS
from .common import WORKSPACE,atomic_json,digest,file_hash,load_cohort,read_json
from .qualified_v3 import require,select_cases,RECEIPT_SHA,CONSUMER
from .runner import execute
from .snapshots import verify_snapshot
from .v3_integration import verify_sources,validate_replay
from .validation import safe_output
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.outputs import read_result

VARIANTS={"perpendicular":"joint_pitch_geometry_development.json","scale":"joint_scale_coherence_development.json",
          "search":"joint_component_beam_development.json","combined":"joint_scale_component_beam_development.json"}


def validate_variant(variant,config,baseline):
    expected={"perpendicular":("observable-joint.v5-perpendicular","diverse-constructions.v1"),
              "scale":("observable-joint.v5-scale","diverse-constructions.v1"),
              "search":("observable-joint.v4","component-beam.v1"),
              "combined":("observable-joint.v5-scale","component-beam.v1")}
    require((config["objective_version"],config.get("joint_search_policy","diverse-constructions.v1"))==expected[variant],"Variant objective/search binding differs")
    different={k for k in set(config)|set(baseline) if config.get(k)!=baseline.get(k)}
    allowed={"objective_version","scale_coherence_weight","scale_coherence_sigma","scale_coherence_min_confidence"} if variant!="search" else {"joint_search_policy"}
    if variant=="combined":allowed.add("joint_search_policy")
    require(different<=allowed,"Quality ablation also changed budgets or unrelated policy")


def checked_integration(root):
    root=Path(root).resolve();request=read_json(root/"request.json");state=read_json(root/"progress.json");receipt=read_json(root/"integration_receipt.json")
    require(request["schema_version"]=="stage4.qualified-v3-integration-request.v1" and state["phase"]=="finished","A completed v3 integration is required")
    require(request["request_sha256"]==digest({k:v for k,v in request.items() if k!="request_sha256"})==receipt["request_sha256"],"Integration identity changed")
    require(receipt["semantic_equivalence_passed"] is True and receipt["source_receipt_sha256"]==RECEIPT_SHA and
            file_hash(root/"report/summary.json")==receipt["report_sha256"],"Integration has not passed its checked evidence gate")
    require("coarse" in request["arms"],"Quality ablations require the matching coarse control")
    verify_sources(request)
    for case in request["import"]["cases"]:
        cid=case["case_id"];folder=root/"cases"/cid
        equality=validate_replay(case,folder/"stage3",request["import"]["pool_script_sha256"])
        require(read_json(folder/"semantic_equivalence.json")==equality,"Imported equivalence changed")
        baseline=read_result(folder/"coarse/stage4")
        require(baseline["implementation"]["sha256"]==CONSUMER and baseline["input"]["manifest_sha256"]==equality["left_manifest_sha256"] and
                file_hash(folder/"coarse/stage4/result.json")==state["cases"][cid]["arms"]["coarse"]["result_sha256"],"Imported coarse control changed")
    return request


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--integration",type=Path,required=True);p.add_argument("--case-id",action="append")
    p.add_argument("--variants",choices=tuple(VARIANTS),nargs="+",default=["perpendicular","scale","search"])
    p.add_argument("--snapshot",type=Path,default=WORKSPACE/"controls/stage4_quality_v5_combined_development")
    p.add_argument("--output",type=Path);p.add_argument("--check",action="store_true");p.add_argument("--plan-only",action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--deep-diagnostics",action="store_true")
    a=p.parse_args(argv);source=a.integration.resolve();original=checked_integration(source)
    require(len(a.variants)==len(set(a.variants)),"Duplicate quality variant")
    cases=select_cases(original["import"]["cases"],a.case_id)
    snapshot=a.snapshot.resolve();frozen=verify_snapshot(snapshot)
    configs={v:snapshot/"configs"/VARIANTS[v] for v in a.variants}
    for variant,path in configs.items():
        config=SolverConfig.from_dict(read_json(path)).to_dict()
        for case in cases:
            baseline=read_result(source/"cases"/case["case_id"]/"coarse/stage4")["configuration"]
            validate_variant(variant,config,baseline)
    if a.check:
        print("Quality preflight passed:",[c["case_id"] for c in cases],a.variants,"(no serving or output)");return 0
    require(a.output is not None,"--output is required")
    output=a.output.resolve();safe_output(output,[source,snapshot,WORKSPACE/"src",WORKSPACE/"scripts",WORKSPACE/"benchmark",WORKSPACE/"controls",Path(original["benchmark_manifest"]).parent])
    request={"schema_version":"stage4.v3-quality-request.v1","output":str(output),"integration":str(source),"integration_request_sha256":original["request_sha256"],
        "integration_receipt_sha256":file_hash(source/"integration_receipt.json"),"case_ids":[c["case_id"] for c in cases],"variants":a.variants,
        "snapshot":str(snapshot),"implementation":frozen["implementation"],"configurations":{k:{"path":str(v),"sha256":file_hash(v)} for k,v in configs.items()},
        "deep_diagnostics":a.deep_diagnostics,"cold_primaries":True,"upstream_reused":True,"fresh_pipeline":False,
        "controllers":{str(p):file_hash(p) for p in (Path(__file__),WORKSPACE/"scripts/run_v3_quality_experiments.py",WORKSPACE/"scripts/diagnose_v3_quality.py",WORKSPACE/"scripts/run_source_snapshot.py")}}
    request["request_sha256"]=digest(request)
    if a.plan_only:
        print(__import__("json").dumps(request,indent=2));return 0
    require(not output.exists() or a.resume,"Existing quality run: use matching --resume or a new output")
    output.mkdir(parents=True,exist_ok=True);lock=output/"run.lock"
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),"w") as stream:stream.write(str(os.getpid()))
    progress=read_json(output/"progress.json") if (output/"progress.json").exists() else {"phase":"predictions","cases":{}}
    try:
        if (output/"request.json").exists():require(read_json(output/"request.json")==request,"Quality run context changed")
        else:atomic_json(output/"request.json",request)
        progress["phase"]="predictions";atomic_json(output/"progress.json",progress)
        for case in cases:
            cid=case["case_id"];state=progress["cases"].setdefault(cid,{})
            baseline=read_result(source/"cases"/cid/"coarse/stage4")
            for variant,path in configs.items():
                target=output/"cases"/cid/variant/"stage4"
                if not (target/"manifest.json").exists():
                    print(cid,variant,": cold consumer",flush=True)
                    cmd=[sys.executable,"-B",WORKSPACE/"scripts/run_source_snapshot.py",snapshot,"solve","--bundle",source/"cases"/cid/"stage3",
                         "--config",path,"--output",target,"--resume"]
                    attempt=execute(cmd,target.parent/"consumer.log",env_overrides=THREADS,echo_progress=True)
                    state[variant]=attempt;atomic_json(output/"progress.json",progress)
                    require(attempt["returncode"]==0,"Quality consumer failed")
                result=read_result(target)
                require(result["input"]["manifest_sha256"]==baseline["input"]["manifest_sha256"] and result["implementation"]==frozen["implementation"] and
                        result["configuration"]==SolverConfig.from_dict(read_json(path)).to_dict(),"Quality consumer binding differs")
                state.setdefault(variant,{}).update(status="predicted",result_sha256=file_hash(target/"result.json"))
                atomic_json(output/"progress.json",progress)
        progress["phase"]="diagnostics";atomic_json(output/"progress.json",progress)
        progress["diagnostics"]=execute([sys.executable,"-B",WORKSPACE/"scripts/diagnose_v3_quality.py","--run",output],output/"diagnostics.log",env_overrides=THREADS,echo_progress=True)
        progress["phase"]="finished" if progress["diagnostics"]["returncode"]==0 else "diagnostics_failed"
        atomic_json(output/"progress.json",progress);print("Quality report:",output/"report/index.html",flush=True)
        return int(progress["phase"]!="finished")
    except KeyboardInterrupt:
        progress["phase"]="interrupted";atomic_json(output/"progress.json",progress);return 130
    except (ValueError,OSError,KeyError) as error:
        progress.update(phase="failed",error=str(error));atomic_json(output/"progress.json",progress);raise
    finally:lock.unlink(missing_ok=True)


def diagnose(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args(argv);root=a.run.resolve()
    req,state=read_json(root/"request.json"),read_json(root/"progress.json")
    require(state["phase"] in {"diagnostics","finished","diagnostics_failed"},"Quality serving barrier not reached")
    require(req["request_sha256"]==digest({k:v for k,v in req.items() if k!="request_sha256"}),"Quality request checksum differs")
    source=Path(req["integration"]);original=checked_integration(source)
    require(original["request_sha256"]==req["integration_request_sha256"] and file_hash(source/"integration_receipt.json")==req["integration_receipt_sha256"],"Integration source changed")
    for cid in req["case_ids"]:
        require(all(state["cases"].get(cid,{}).get(v,{}).get("status")=="predicted" for v in req["variants"]),"Quality serving barrier incomplete")
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from .report import document,esc,image,number,table,link
    cohort_root,cohort=load_cohort(original["benchmark_manifest"])
    rows,cells,visuals=[],[],[];report=root/"report";report.mkdir(exist_ok=True)
    for cid in req["case_ids"]:
        case=next(c for c in cohort["cases"] if c["case_id"]==cid)
        reference=read_json(cohort_root/case["path"]/"evaluation_only/reference_geometry.json")
        baseline=read_result(source/"cases"/cid/"coarse/stage4")
        for variant in req["variants"]:
            target=root/"cases"/cid/variant/"stage4";result=read_result(target)
            require(file_hash(target/"result.json")==state["cases"][cid][variant]["result_sha256"],"Quality prediction changed")
            require(result["input"]["manifest_sha256"]==baseline["input"]["manifest_sha256"],"Quality input differs")
            ref={**reference,"input_manifest_sha256":result["input"]["manifest_sha256"]}
            qualities=[assembly_quality(r,ref,evaluate(r,ref)) for r in (baseline,result)]
            row={"case_id":cid,"variant":variant,"quality_control":qualities[0],"quality_experiment":qualities[1],
                 "score":[r["layouts"][0]["objective_score"] for r in (baseline,result)],"objective":[r["configuration"]["objective_version"] for r in (baseline,result)],
                 "consumer_seconds":[r["resources"]["elapsed_before_publication_seconds"] for r in (baseline,result)],
                 "score_terms":result["search"]["observable_structure"]["layout_score_terms"][0],"search":result["search"],
                 "rejections":result["rejections"],"visual_review":None,"cold_primary":True,"evaluation_truth_opened":True}
            if req["deep_diagnostics"]:
                from paper_reconstruction_stage4.inputs import load_problem
                from .diagnostics import selected_support_roles
                from .joint_diagnostics import support_plot,audit_candidates
                config=SolverConfig.from_dict(result["configuration"]);problem=load_problem(source/"cases"/cid/"stage3",config)
                row["selected_support_roles"]=selected_support_roles(problem,result,ref)
                row["existing_complete_candidates"]=audit_candidates(result,problem,config,ref,maximum=24,seconds=20.)
                support_plot(problem,result,report/(cid+"_"+variant+"_supports.svg"))
            rows.append(row);q=qualities[1];g=q["common_gauge"]
            whole=g["non_anchor_probe_errors_reference_px"]["p95"] if g["whole_page_error_available"] else None
            cells.append((esc(cid),esc(variant),esc(q["component_sizes"]),number(q["selected_neighbor_identity_precision"]),number(whole),number(row["consumer_seconds"][1])))
            visuals.append('<h2>'+esc(cid+" / "+variant)+'</h2><div class="visuals">'+image(source/"cases"/cid/"coarse/stage4/preview.png",report,"Frozen v4 cold primary")+image(target/"preview.png",report,"Experimental cold primary")+'</div>')
    atomic_json(report/"summary.json",{"schema_version":"stage4.v3-quality-comparison.v1","request_sha256":req["request_sha256"],"cases":rows,
        "construction_requires_visual_review":True,"objective_values_not_comparable_between_versions":True})
    content='<h1>Separate cold quality ablations</h1><p>Same imported evidence. Coarse controls. No reference-guided seed or upstream inference. Objective versions have different score scales; judge primary construction, inventory and physical/support validity. Four-pixel accuracy is separate.</p>'
    content+=table(["Case","Variant","Components","True-neighbor fraction","Whole-page p95 reference px","Consumer seconds"],cells)+''.join(visuals)
    (report/"index.html").write_text(document("Quality ablations",content),encoding="utf-8")
    atomic_json(root/"experiment_receipt.json",{"request_sha256":req["request_sha256"],"summary_sha256":file_hash(report/"summary.json"),"status":"development_comparison_not_promoted","cold_primaries":True})
    return 0
