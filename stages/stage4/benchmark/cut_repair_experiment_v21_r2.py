"""Cold Stage 4 replay of the fourteen sealed capacity inputs, no upstream inference."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys
from .common import WORKSPACE,atomic_json,read_json,digest,file_hash,inventory
from .snapshots import freeze,verify_snapshot
from .runner import execute
from .recovery_experiment_v19_r2 import run_attempt,THREADS

SOURCE=WORKSPACE/'outputs/v20_capacity_stage4'
SOURCE_REQUEST='84034b288850ba814e7426a2fc4b328ce40db37e949d44aed49e7e64748d6787'
SOURCE_BARRIER='e65384d65a453dec73073b6c825b6cc806fc2b07c90365d4b171f20d699c96a2'
SNAPSHOT=WORKSPACE/'controls/stage4_v21_cut_repair_development_r2'
CONFIG=WORKSPACE/'configs/joint_v21_cut_repair_development.json'
PROFILE=WORKSPACE/'configs/shared_scale_complete_rectangle_v1.json'
NEW_CONTROLLERS=['benchmark/cut_repair_experiment_v21_r2.py','benchmark/cut_repair_diagnosis_v21_r2.py',
    'scripts/run_cut_repair_v21_r2.py','scripts/diagnose_cut_repair_v21_r2.py']


def require(condition,message):
    if not condition:raise ValueError(message)


def checked(value,key):
    require(value[key]==digest({k:v for k,v in value.items() if k!=key}),'Changed '+key)
    return value


def controllers():
    from .scale_extension_v20 import controllers as previous
    return {**previous(),**{str(WORKSPACE/p):file_hash(WORKSPACE/p) for p in NEW_CONTROLLERS}}


def output_path(path):
    path=Path(path).resolve();parent=(WORKSPACE/'outputs').resolve()
    require(path!=parent and path.is_relative_to(parent),'Choose a new folder inside Stage 4 outputs')
    return path


def sealed_source(root):
    """Reuse semantic-validation commitments and verify their exact bytes.

    Do not decompress/validate every archive repeatedly during startup. The
    cold serving worker still validates the actual input it consumes. Baseline
    diagnosis JSON and references are not parsed here.
    """
    root=output_path(root);request=checked(read_json(root/'request.json'),'request_sha256')
    seal=checked(read_json(root/'predictions_sealed.json'),'barrier_sha256')
    require(request['request_sha256']==SOURCE_REQUEST and seal['barrier_sha256']==SOURCE_BARRIER and
        seal['request_sha256']==SOURCE_REQUEST and seal['references_opened'] is False,'Use the sealed V20 capacity run')
    baseline=Path(request['baseline']['root']);base=checked(read_json(baseline/'request.json'),'request_sha256')
    base_seal=checked(read_json(baseline/'predictions_sealed.json'),'barrier_sha256')
    require(base['request_sha256']==request['baseline']['request_sha256'] and
        base_seal['barrier_sha256']==request['baseline']['barrier_sha256'] and
        base_seal['request_sha256']==base['request_sha256'] and base_seal['references_opened'] is False,'Wrong eight-case baseline')
    six_root=Path(request['upstream']);six=checked(read_json(six_root/'upstream_sealed.json'),'barrier_sha256')
    eight_root=Path(base['source']['root']);eight=checked(read_json(eight_root/'predictions_sealed.json'),'barrier_sha256')
    require(six['barrier_sha256']==request['upstream_barrier_sha256'] and
        eight['barrier_sha256']==base['source']['barrier_sha256'],'Changed upstream commitments')
    receipt=read_json(root/'experiment_receipt.json')
    require(receipt['request_sha256']==SOURCE_REQUEST and receipt['barrier_sha256']==SOURCE_BARRIER,'Unfinished capacity report')
    summary=next(r for r in receipt['files'] if r['path']=='report/summary.json')
    require(file_hash(root/summary['path'])==summary['sha256'],'Changed baseline comparison report')
    bound={str(root/summary['path']):summary['sha256']}
    for parent in (root,baseline,six_root,eight_root):
        for name in ('request.json','predictions_sealed.json','upstream_sealed.json'):
            path=parent/name
            if path.exists():bound[str(path)]=file_hash(path)
    cases=[]
    for original,upstream_seal,old_root,arm in [
        *[(c,eight,baseline,'adaptive') for c in base['cases']],
        *[(c,six,root,None) for c in request['cases']]]:
        case=dict(original);bundle=Path(case['bundle']).resolve();cid=case['case_id']
        print('[inputs] Verify sealed bytes: '+cid+' ('+str(case['pieces'])+' pieces)',flush=True)
        expected={str(Path(p).resolve()):h for p,h in upstream_seal['files'].items() if Path(p).resolve().is_relative_to(bundle)}
        actual={str((bundle/r['path']).resolve()):r['sha256'] for r in inventory(bundle)}
        require(expected and actual==expected,'Stage 3 serving files differ from their seal: '+cid)
        bound.update(expected)
        capture=Path(case['capture_profile']);require(file_hash(capture)==case['capture_profile_sha256'],'Changed capture: '+cid)
        bound[str(capture)]=case['capture_profile_sha256']
        old=old_root/'cases'/cid
        if arm:old=old/arm
        old=old/'stage4'
        require(str(old/'result.json') in (base_seal if arm else seal)['files'],'Unsealed comparison result')
        bound[str(old/'result.json')]=(base_seal if arm else seal)['files'][str(old/'result.json')]
        require(file_hash(old/'result.json')==bound[str(old/'result.json')],'Changed comparison result: '+cid)
        case['baseline']=str(old)
        # Only serving metadata and protected-reference commitments enter the
        # request. The child solve command receives no baseline or reference.
        cases.append({k:case[k] for k in ('case_id','pieces','bundle','bundle_sha256','capture_profile',
            'capture_profile_sha256','constraint_binding','evaluation_reference','evaluation_reference_sha256','baseline')})
    require(Counter(c['pieces'] for c in cases)=={4:2,8:2,12:2,16:2,20:2,24:2,30:2},'Capacity cohort differs')
    return cases,bound


def create_request(output,source,case_ids=None):
    print('[preflight] Authenticate capacity commitments and serving input bytes',flush=True)
    cases,bound=sealed_source(source)
    selected=[c for c in cases if not case_ids or c['case_id'] in case_ids]
    require(selected and (not case_ids or len(set(case_ids))==len(case_ids)==len(selected)),'Choose unique known case IDs')
    from paper_reconstruction_stage4.config import SolverConfig
    config=SolverConfig.from_dict(read_json(CONFIG)).to_dict()
    old=read_json(Path(source)/'request.json')['configuration']
    require({k:v for k,v in config.items() if k!='native_cut_repair'}==old and config['native_cut_repair'] is True,
        'V21 changes only the explicit recovery/scheduling opt-in')
    print('[preflight] Freeze versioned Stage 4 source and configuration',flush=True)
    receipt=freeze(SNAPSHOT,[CONFIG,PROFILE])
    frozen_config=SNAPSHOT/'configs'/CONFIG.name;profile=SNAPSHOT/'configs'/PROFILE.name
    bound.update({str(p):file_hash(p) for p in (CONFIG,PROFILE,frozen_config,profile)})
    value={'schema_version':'stage4.cut-repair-replay.v21','output':str(output),'source':str(Path(source).resolve()),
        'source_request_sha256':SOURCE_REQUEST,'source_barrier_sha256':SOURCE_BARRIER,
        'snapshot':str(SNAPSHOT),'implementation':receipt['implementation'],'configuration':config,
        'config':str(frozen_config),'profile':str(profile),'bound_files':bound,'controllers':controllers(),
        'cases':sorted(selected,key=lambda c:(c['pieces'],c['case_id'])),
        'stage2_stage3_inference_run':False,'saved_layout_seeds':False,
        'evaluation_truth_opened_during_predictions':False,
        'input_validation_policy':'sealed-bytes-once-per-controller-invocation; semantic-validation-in-serving-worker',
        'constructibility_probe_seconds_per_tolerance':40.,'constructibility_probe_tolerances_px':[8.,16.]}
    value['request_sha256']=digest(value);return value


def authenticate(request):
    checked(request,'request_sha256')
    require(verify_snapshot(request['snapshot'])['implementation']==request['implementation'],'Frozen source differs')
    entries={**request['controllers'],**request['bound_files']}
    print('[preflight] Verify '+str(len(entries))+' bound files; reuse prior semantic validation',flush=True)
    for index,(p,h) in enumerate(entries.items(),1):
        require(file_hash(p)==h,'Bound file changed: '+p)
        if index%250==0:print('[preflight] Verified '+str(index)+'/'+str(len(entries))+' files',flush=True)


def command_for(output,request,case):
    return [sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',request['snapshot'],'solve',
        '--bundle',case['bundle'],'--config',request['config'],'--constraint-profile',request['profile'],
        '--capture-profile',case['capture_profile'],'--output',output/'cases'/case['case_id']/'stage4','--resume']


def verify_predictions(output,request,progress):
    from paper_reconstruction_stage4.outputs import read_result
    from paper_reconstruction_stage4.assessment import verify_package
    files={};rows=[]
    for case in request['cases']:
        cid=case['case_id'];folder=output/'cases'/cid;entry=progress['cases'][cid]
        require(entry['status'] in ('predicted','failed'),'Unfinished serving attempt: '+cid)
        if entry['status']=='predicted':
            value=read_result(folder/'stage4');assessment=verify_package(folder/'stage4.assessment')
            require(value['implementation']==request['implementation'] and value['configuration']==request['configuration'] and
                value['input']['manifest_sha256']==case['bundle_sha256'] and value['evaluation_truth_opened'] is False and
                value['search']['constraint_profile']==case['constraint_binding'],'Serving result binding differs: '+cid)
            binding=assessment['binding']
            require(binding['result_sha256']==file_hash(folder/'stage4/result.json') and
                binding['assessment_implementation_sha256']==request['implementation']['sha256'] and
                binding['configuration_sha256']==digest(request['configuration']) and
                binding['constraint_profile']==case['constraint_binding'],'Assessment binding differs: '+cid)
        files.update({str(folder/r['path']):r['sha256'] for r in inventory(folder)})
        rows.append({'case_id':cid,'pieces':case['pieces'],'status':entry['status']})
    return {'cases':rows,'files':files,'progress_cases_sha256':digest(progress['cases'])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id',action='append');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args(argv)
    output=output_path(args.output);source=output_path(args.source)
    require(not output.is_relative_to(source) and not source.is_relative_to(output),'Use separate output and source trees')
    print('[startup] V21 cold Stage 4 replay; Stage 2/3 inputs will be reused',flush=True)
    if (output/'request.json').exists():
        require(args.resume,'Use --resume for this unchanged request');request=read_json(output/'request.json')
        require(request['source']==str(source) and request['output']==str(output),'Resume source/output differs')
        if args.case_id:require(set(args.case_id)=={c['case_id'] for c in request['cases']} and len(set(args.case_id))==len(args.case_id),'Resume selection differs')
        authenticate(request)
    else:
        require(not output.exists() or not any(output.iterdir()),'Unrelated nonempty output')
        request=create_request(output,source,args.case_id)
    output.mkdir(parents=True,exist_ok=True);lock=output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:f.write(str(os.getpid()))
    try:
        atomic_json(output/'request.json',request)
        progress=read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'preflight','cases':{}}
        if args.preflight_only:
            progress['phase']='preflight_complete';atomic_json(output/'progress.json',progress)
            print('Preflight complete; no solver or upstream inference ran:',output/'request.json',flush=True);return 0
        if not (output/'predictions_sealed.json').exists():
            progress['phase']='predictions';atomic_json(output/'progress.json',progress)
            for i,case in enumerate(request['cases'],1):
                cid=case['case_id'];folder=output/'cases'/cid;entry=progress['cases'].setdefault(cid,{'status':'pending','attempts':[]})
                if entry['status']=='predicted':
                    require(file_hash(folder/'stage4/result.json')==entry['result_sha256'],'Changed committed result');continue
                entry['status']='working';atomic_json(output/'progress.json',progress)
                print(f'[{i}/{len(request["cases"])}] {cid} ({case["pieces"]} pieces): cold V21 Stage 4',flush=True)
                attempt=run_attempt(command_for(output,request,case),folder,entry,lambda:atomic_json(output/'progress.json',progress))
                entry['status']='predicted' if attempt['returncode']==0 else 'failed'
                if entry['status']=='predicted':entry['result_sha256']=file_hash(folder/'stage4/result.json')
                atomic_json(output/'progress.json',progress)
            print('[seal] Verify and seal all serving attempts before reference diagnosis',flush=True)
            seal={'schema_version':'stage4.cut-repair-prediction-seal.v21','request_sha256':request['request_sha256'],
                **verify_predictions(output,request,progress),'references_opened':False}
            seal['barrier_sha256']=digest(seal);atomic_json(output/'predictions_sealed.json',seal)
        progress['phase']='diagnostics';atomic_json(output/'progress.json',progress)
        attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_cut_repair_v21_r2.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        progress['diagnostics']=attempt;atomic_json(output/'progress.json',progress)
        require(attempt['returncode']==0,'Diagnostics failed; predictions preserved; inspect diagnostics.log')
        progress['phase']='finished';atomic_json(output/'progress.json',progress)
        print('V21 comparison report:',output/'report/index.html',flush=True)
        return int(any(e['status']=='failed' for e in progress['cases'].values()))
    except KeyboardInterrupt:
        print('Interrupted; committed results/timings preserved. Re-run the same command with --resume.',flush=True)
        return 130
    finally:
        lock.unlink(missing_ok=True)
