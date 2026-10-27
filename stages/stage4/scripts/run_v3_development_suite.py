"""Run checked v3 integration, then cold quality comparisons automatically."""
from pathlib import Path
import argparse
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import WORKSPACE,atomic_json,file_hash,read_json
from benchmark.qualified_v3 import require
from benchmark.snapshots import verify_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    p.add_argument("--case-id",action="append")
    p.add_argument("--arms",nargs="+",choices=("coarse","robust"),default=["coarse"])
    p.add_argument("--quality-variants",nargs="+",choices=("perpendicular","scale","search","combined"),default=["combined"])
    p.add_argument("--snapshot",type=Path,default=WORKSPACE/"controls/stage4_quality_v5_combined_development")
    p.add_argument("--resume",action="store_true");p.add_argument("--check",action="store_true")
    p.add_argument("--deep-diagnostics",action="store_true")
    a=p.parse_args();require("coarse" in a.arms,"Suite needs the coarse integration control")
    verify_snapshot(a.snapshot)
    paths=[WORKSPACE/"benchmark/qualified_v3.py",WORKSPACE/"benchmark/v3_integration.py",WORKSPACE/"benchmark/quality_experiments.py",
           WORKSPACE/"scripts/run_qualified_v3_integration.py",WORKSPACE/"scripts/run_v3_quality_experiments.py",Path(__file__),a.snapshot.resolve()/"receipt.json"]
    pins={str(v):file_hash(v) for v in paths}
    integration=[sys.executable,"-B",str(WORKSPACE/"scripts/run_qualified_v3_integration.py"),"--arms",*a.arms]
    for cid in a.case_id or []:integration.extend(["--case-id",cid])
    if a.check:
        code=subprocess.run([*integration,"--check"],cwd=WORKSPACE).returncode
        if code==0:print("Suite preflight passed. Quality source is gated after successful integration. No inference or output created.")
        return code
    require(a.output is not None,"--output is required")
    output=a.output.resolve()
    integration.extend(["--output",str(output/"integration")])
    quality=[sys.executable,"-B",str(WORKSPACE/"scripts/run_v3_quality_experiments.py"),"--integration",str(output/"integration"),
             "--snapshot",str(a.snapshot.resolve()),"--variants",*a.quality_variants,"--output",str(output/"quality")]
    if a.resume:integration.append("--resume");quality.append("--resume")
    if a.deep_diagnostics:quality.append("--deep-diagnostics")
    for name,command in (("integration",integration),("quality",quality)):
        require(all(file_hash(Path(path))==sha for path,sha in pins.items()),"Suite code/snapshot changed between phases")
        print("Starting",name,flush=True)
        code=subprocess.run(command,cwd=WORKSPACE).returncode
        if code:return code
    atomic_json(output/"suite_receipt.json",{"schema_version":"stage4.v3-development-suite.v1","controllers":pins,
        "integration":read_json(output/"integration/integration_receipt.json"),
        "quality":read_json(output/"quality/experiment_receipt.json"),"default_promoted":False,
        "upstream_inference_repeated":False,"cpu_only_qualified":False})
    print("Integration:",output/"integration/report/index.html")
    print("Quality:",output/"quality/report/index.html")
    return 0


if __name__=="__main__":raise SystemExit(main())
