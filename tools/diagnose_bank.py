"""Explicit post-seal, reference-assisted saved-bank construction witness."""
import argparse
from pathlib import Path
import subprocess
import sys
from common import ROOT,environment,python,read,sha,inventory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--seconds',type=float,default=60.)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.worker:
        return subprocess.call([python('stage4'),'-B',str(Path(__file__).resolve()),'--run',str(a.run.resolve()),
            '--case',a.case,'--seconds',str(a.seconds),'--worker'],cwd=ROOT,env=environment('stage4'))
    run=a.run.resolve();barrier=read(run/'predictions_sealed.json');request=read(run/'request.json')
    for cid,files in barrier['predictions'].items():
        if inventory(run/'cases'/cid/'stage4')!=files:raise ValueError('Prediction barrier differs: '+cid)
    case=next(c for c in request['cases'] if c['case_id']==a.case)
    from paper_reconstruction_stage4.outputs import read_result
    from benchmark.constructibility_v20 import probe
    result=read_result(run/'cases'/a.case/'stage4')
    reference=ROOT/case['evaluation_reference']
    reference_receipt=next(x for x in read(ROOT/'provenance/assets.json')['files'] if x['path']==case['evaluation_reference'])
    if sha(reference)!=reference_receipt['sha256']:raise ValueError('Reference receipt differs')
    bound={**case,'bundle':str(ROOT/case['bundle']),'bundle_sha256':result['input']['manifest_sha256'],
        'capture_profile':str(ROOT/case['capture_profile']),'evaluation_reference_sha256':sha(reference)}
    value=probe(bound,result,read(reference),{'profile':str(ROOT/'stages/stage4/configs/shared_scale_complete_rectangle_v1.json'),
        'constructibility_probe_tolerances_px':[8.,16.,32.,48.], 'constructibility_probe_seconds_per_tolerance':a.seconds},
        run/'bank_diagnostics'/a.case,sha(run/'predictions_sealed.json'))
    for cid,files in barrier['predictions'].items():
        if inventory(run/'cases'/cid/'stage4')!=files:raise ValueError('Primary predictions changed')
    print('Positive construction witness:',value['approximately_correct_native_assembly_demonstrated'])
    print('A negative/expired result is inconclusive; this is never a blind success.')
    return 0


if __name__=='__main__':raise SystemExit(main())
