"""Consumer-only V19 ablations on the eight sealed V14 scaling bundles."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys
from .common import WORKSPACE,atomic_json,digest,file_hash,inventory,read_json,verify_files
from .runner import execute
from .snapshots import freeze,verify_snapshot
from .scale_pipeline_v14 import authenticate as authenticate_source,checked,require
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.constraints import bind_profile
from paper_reconstruction_stage4.outputs import read_result
from paper_reconstruction_stage4.assessment import verify_package

SOURCE=WORKSPACE/'outputs/v14_full_pipeline_scale_2each'
SOURCE_REQUEST='662815e6205b0e28b107fb12afe2c311d40b4149bab4dd60d57e122ebdbe0510'
SOURCE_BARRIER='8a53f67d6986fc990c64c73f015c7f6cdfc74595be3ee7b1e788ab5fc6d2a2f5'
SNAPSHOT=WORKSPACE/'controls/stage4_v19_straight_deferred_development'
ARMS={name:WORKSPACE/'configs/joint_v19_straight_deferred_development.json' for name in ('adaptive',)}
PROFILE=WORKSPACE/'configs/shared_scale_complete_rectangle_v1.json'
THREADS={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
DEPENDENCIES=['benchmark/recovery_experiment_v19.py','benchmark/recovery_diagnosis_v19.py','benchmark/recovery_gallery_v18_r2.py',
    'benchmark/common.py','benchmark/runner.py','benchmark/snapshots.py','benchmark/qualified_v3.py',
    'benchmark/scale_pipeline_v14.py','benchmark/scale_pipeline_diagnosis_v14.py',
    'benchmark/diagnostics.py','benchmark/quality.py','benchmark/report.py',
    'benchmark/native_repair_experiment_v14.py','evaluation/evaluate.py',
    'benchmark/recovery_report_v18_r2_fix.py','scripts/repair_recovery_v18_r2_report.py',
    'scripts/run_recovery_v19.py','scripts/diagnose_recovery_v19.py','scripts/run_source_snapshot.py']



COMPARISON=WORKSPACE/'outputs/v15_recovery_ablation'


def comparison_v17_binding():
    root=WORKSPACE/'outputs/v17_native_graph_focus'
    request=checked(read_json(root/'request.json'),'request_sha256')
    barrier=checked(read_json(root/'predictions_sealed.json'),'barrier_sha256')
    require(request['request_sha256']=='a3908b9de78f973fd4024780c10715cdfd2360278f00a13263561076b3edccaf' and
        barrier['barrier_sha256']=='e96d79698a5f1e1e1bb40bb3e023d1bbc106937b9950569e5f8f8df6fe32ce06' and
        barrier['request_sha256']==request['request_sha256'] and barrier['references_opened'] is False,'Wrong V17 comparison')
    receipt=read_json(root/'experiment_receipt.json')
    require(receipt['request_sha256']==request['request_sha256'] and receipt['barrier_sha256']==barrier['barrier_sha256'],'V17 receipt differs')
    verify_files(root,receipt['files'])
    for path,sha in barrier['files'].items():require(file_hash(path)==sha,'V17 prediction changed: '+path)
    return {'root':str(root),'request_sha256':request['request_sha256'],'barrier_sha256':barrier['barrier_sha256'],
        'report_receipt_sha256':file_hash(root/'experiment_receipt.json')}


def comparison_binding():
    request=checked(read_json(COMPARISON/'request.json'),'request_sha256')
    barrier=checked(read_json(COMPARISON/'predictions_sealed.json'),'barrier_sha256')
    require(request['request_sha256']=='db9504abe87d2b2a439506597f2cbdedb95b2d1b766fa082959c552e8fdbee54' and
        barrier['barrier_sha256']=='da870984db98c02ea2f384d5e38d94f1e9fa8ee1ed84736437d37395c524b6b8' and
        barrier['request_sha256']==request['request_sha256'] and barrier['references_opened'] is False,'Wrong V15 comparison')
    receipt=read_json(COMPARISON/'experiment_receipt.json')
    require(receipt['request_sha256']==request['request_sha256'] and receipt['barrier_sha256']==barrier['barrier_sha256'],'V15 receipt differs')
    verify_files(COMPARISON,receipt['files'])
    for path,sha in barrier['files'].items():require(file_hash(path)==sha,'V15 prediction changed: '+path)
    return {'root':str(COMPARISON),'request_sha256':request['request_sha256'],
        'barrier_sha256':barrier['barrier_sha256'],'report_receipt_sha256':file_hash(COMPARISON/'experiment_receipt.json')}



def comparison_v16_binding():
    root=WORKSPACE/'outputs/v16_recovery_portfolio'
    request=checked(read_json(root/'request.json'),'request_sha256')
    barrier=checked(read_json(root/'predictions_sealed.json'),'barrier_sha256')
    require(request['request_sha256']=='d1eb07b21dc94755e120b3da907defe2af6fe234b268f77d6fbb48dece6c612e' and
        barrier['barrier_sha256']=='c881dfaa7e32948e37704be6a60140680832efd2f46fc6753a91027746d98ad9' and
        barrier['request_sha256']==request['request_sha256'] and barrier['references_opened'] is False,'Wrong V16 comparison')
    receipt=read_json(root/'experiment_receipt.json')
    require(receipt['request_sha256']==request['request_sha256'] and receipt['barrier_sha256']==barrier['barrier_sha256'],'V16 receipt differs')
    verify_files(root,receipt['files'])
    for path,sha in barrier['files'].items():require(file_hash(path)==sha,'V16 prediction changed: '+path)
    return {'root':str(root),'request_sha256':request['request_sha256'],
        'barrier_sha256':barrier['barrier_sha256'],'report_receipt_sha256':file_hash(root/'experiment_receipt.json')}



def comparison_v18_binding():
    from scripts.repair_recovery_v18_r2_report import bootstrap
    root=WORKSPACE/'outputs/v18_geometry_full_r2'
    request,_=bootstrap(root)
    checked(request,'request_sha256')
    barrier=checked(read_json(root/'predictions_sealed.json'),'barrier_sha256')
    require(request['request_sha256']=='a9d4ad7cdbdc2e47096556d9156aa2fe1accde89581ea3fb14195f175b459068' and
        barrier['barrier_sha256']=='b71ac2dded466e127426378ae0ebd05767be41f3c25f9fcef26af1846244ff82','Wrong V18 comparison')
    receipt=read_json(root/'report_repair_v1/receipt.json')
    checked(receipt,'receipt_sha256');verify_files(root,receipt['files'])
    return {'root':str(root),'request_sha256':request['request_sha256'],'barrier_sha256':barrier['barrier_sha256'],
        'report_repair_receipt_sha256':file_hash(root/'report_repair_v1/receipt.json')}


def source_binding(source):
    """No evaluation JSON is parsed. Bind completed source evidence by bytes."""
    source=Path(source).resolve();request=checked(read_json(source/'request.json'),'request_sha256')
    barrier=checked(read_json(source/'predictions_sealed.json'),'barrier_sha256')
    require(request['request_sha256']==SOURCE_REQUEST and barrier['barrier_sha256']==SOURCE_BARRIER and
            barrier['request_sha256']==SOURCE_REQUEST and barrier['references_opened'] is False,'Use the reviewed eight-case scaling source')
    authenticate_source(request)
    require(Counter(c['pieces'] for c in request['cases'])=={4:2,8:2,12:2,16:2},'Source cohort differs')
    for path,sha in barrier['files'].items():require(file_hash(path)==sha,'Sealed source changed: '+path)
    report=read_json(source/'experiment_receipt.json')
    require(report['request_sha256']==SOURCE_REQUEST and report['barrier_sha256']==SOURCE_BARRIER,'Source report binding differs')
    verify_files(source,report['files'])
    metadata=[source/'request.json',source/'predictions_sealed.json',source/'experiment_receipt.json',source/'review/retained_candidates.json']
    return request,{'root':str(source),'request_sha256':SOURCE_REQUEST,'barrier_sha256':SOURCE_BARRIER,
        'files':{str(p):file_hash(p) for p in metadata}}


def create_request(output,source,arms,case_ids=None):
    upstream,binding=source_binding(source)
    selected=[c for c in upstream['cases'] if not case_ids or c['case_id'] in case_ids]
    require(selected and (not case_ids or len(set(case_ids))==len(case_ids)==len(selected)),'Select unique source case IDs')
    require(arms and len(set(arms))==len(arms) and set(arms)<=set(ARMS),'Select unique V19 arms')
    configs={arm:SolverConfig.from_dict(read_json(ARMS[arm])).to_dict() for arm in arms}
    base=upstream['configuration']
    for value in configs.values():
        require({k:v for k,v in value.items() if k not in {'native_growth_policy','native_repair_policy','repair_seconds','repair_source_limit','repair_merge_checks','joint_wall_seconds','joint_finalize_seconds'}}==
                {k:v for k,v in base.items() if k not in {'native_growth_policy','native_repair_policy','repair_seconds','repair_source_limit','repair_merge_checks','joint_wall_seconds','joint_finalize_seconds'}},'V19 changes only its explicit repair policy and declared wall/finalization scheduling envelope')
    frozen=freeze(SNAPSHOT,[*ARMS.values(),PROFILE])
    profile=SNAPSHOT/'configs'/PROFILE.name;cases=[]
    for case in selected:
        cid=case['case_id'];bundle=Path(source)/'cases'/cid/'stage3'
        inputs=Path(upstream['cohort_root'])/case['path'];capture=inputs/'stage1/capture_profile.json'
        config=SolverConfig.from_dict(configs[arms[0]])
        problem=load_problem(bundle,config);bind_profile(problem,config,profile,capture)
        require(len(problem.pieces)==case['pieces'],'Physical inventory differs')
        baseline=Path(source)/'cases'/cid/'stage4'
        # The serving worker receives only the bundle, capture and configuration.
        # Baseline poses and references are not supplied as seeds.
        cases.append({'case_id':cid,'pieces':case['pieces'],'bundle':str(bundle),
            'bundle_sha256':problem.manifest['manifest_sha256'],'capture_profile':str(capture),
            'capture_profile_sha256':file_hash(capture),'constraint_binding':problem.constraint_profile,
            'baseline':str(baseline),'baseline_result_sha256':file_hash(baseline/'result.json'),
            'evaluation_reference':str(inputs/'evaluation_only/reference_geometry.json'),
            'evaluation_reference_sha256':next(f['sha256'] for f in case['files'] if f['path']=='evaluation_only/reference_geometry.json'),
            'reference_image':str(inputs/'evaluation_only/reference_page.png'),
            'reference_image_sha256':next(f['sha256'] for f in case['files'] if f['path']=='evaluation_only/reference_page.png')})
        print('Authenticate '+cid+': sealed Stage 3 input and capture binding',flush=True)
    value={'schema_version':'stage4.v19-recovery-ablation-request.v1','output':str(output),
        'comparison_v18':comparison_v18_binding(),'comparison_v17':comparison_v17_binding(),'comparison_v16':comparison_v16_binding(),'comparison_v15':comparison_binding(),'source':binding,'source_implementation':upstream['implementation'],'snapshot':str(SNAPSHOT),
        'implementation':frozen['implementation'],'arms':list(arms),'cases':cases,
        'configurations':configs,'configs':{a:str(SNAPSHOT/'configs'/ARMS[a].name) for a in arms},
        'profile':str(profile),'bound_files':{str(p):file_hash(p) for p in [*ARMS.values(),PROFILE,profile,
            *(SNAPSHOT/'configs'/p.name for p in ARMS.values())]},
        'controllers':{str(WORKSPACE/p):file_hash(WORKSPACE/p) for p in DEPENDENCIES},
        'stage2_stage3_inference_run':False,'saved_layout_seeds':False,'evaluation_truth_opened_during_predictions':False,
        'timing_scope':'Single-thread Stage 4 subprocess wall time, summed over completed and interrupted local attempts, including loading, publication and assessment. Stage 1/2/3 evidence reused; resume uses this request only.',
        'source_selection_changed':True,'default_promoted':False}
    value['request_sha256']=digest(value)
    return value


def authenticate(request):
    require(request['comparison_v18']==comparison_v18_binding(),'V18 comparison evidence changed')
    require(request['comparison_v17']==comparison_v17_binding(),'V17 comparison evidence changed')
    require(request['comparison_v16']==comparison_v16_binding(),'V16 comparison evidence changed')
    require(request['comparison_v15']==comparison_binding(),'V15 comparison evidence changed')
    checked(request,'request_sha256')
    require(verify_snapshot(request['snapshot'])['implementation']==request['implementation'],'V19 snapshot changed')
    for path,sha in {**request['controllers'],**request['bound_files'],**request['source']['files']}.items():
        require(file_hash(path)==sha,'Bound experiment file changed: '+path)
    _,binding=source_binding(request['source']['root'])
    require(binding==request['source'],'Source binding changed')
    for case in request['cases']:
        require(file_hash(case['capture_profile'])==case['capture_profile_sha256'] and
                file_hash(Path(case['baseline'])/'result.json')==case['baseline_result_sha256'],'Capture or baseline changed')


def command_for(output,request,case,arm):
    return [sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',request['snapshot'],'solve',
        '--bundle',case['bundle'],'--config',request['configs'][arm],
        '--constraint-profile',request['profile'],'--capture-profile',case['capture_profile'],
        '--output',output/'cases'/case['case_id']/arm/'stage4','--resume']


def run_attempt(command,folder,entry,save_progress):
    """Keep immutable per-attempt timing even when the child is interrupted."""
    history=[read_json(p) for p in sorted((folder/'attempts').glob('attempt_*.json'))]
    require(len(history)>=len(entry['attempts']) and history[:len(entry['attempts'])]==entry['attempts'],
        'Attempt history differs from committed progress')
    if len(history)>len(entry['attempts']):
        entry['attempts']=history;save_progress()
    existing=(folder/'stage4/manifest.json').exists()
    def record(attempt):
        attempt={**attempt,'existing_result_before_invocation':existing}
        entry['attempts'].append(attempt)
        path=folder/'attempts'/f'attempt_{len(entry["attempts"]):03d}.json'
        require(not path.exists(),'Attempt history must not be overwritten')
        atomic_json(path,attempt)
        save_progress()
        return attempt
    try:
        attempt=execute(command,folder/'stage4.log',env_overrides=THREADS,echo_progress=True)
    except KeyboardInterrupt:
        entry['status']='interrupted'
        path=folder/'stage4.attempt.json'
        if path.exists() and read_json(path).get('status')=='interrupted':record(read_json(path))
        else:save_progress()
        raise
    return record(attempt)


def verify_predictions(output,request,progress):
    authenticate(request);files={};records=[]
    for case in request['cases']:
        cid=case['case_id']
        for arm in request['arms']:
            entry=progress['cases'][cid][arm];folder=output/'cases'/cid/arm;target=folder/'stage4'
            require(entry['status'] in ('predicted','failed'),'Prediction attempt is unfinished')
            if entry['status']=='predicted':
                result=read_result(target);manifest=verify_package(folder/'stage4.assessment')
                require(result['implementation']==request['implementation'] and
                    result['configuration']==request['configurations'][arm] and
                    result['input']['manifest_sha256']==case['bundle_sha256'] and
                    result['evaluation_truth_opened'] is False and
                    result['search']['constraint_profile']==case['constraint_binding'],'V19 prediction identity differs')
                require(manifest['binding']['result_sha256']==file_hash(target/'result.json') and
                    manifest['binding']['assessment_implementation_sha256']==request['implementation']['sha256'] and
                    manifest['binding']['constraint_profile']==case['constraint_binding'] and
                    manifest['binding']['configuration_sha256']==digest(request['configurations'][arm]),'V19 assessment binding differs')
            for item in inventory(folder):files[str(folder/item['path'])]=item['sha256']
            records.append({'case_id':cid,'arm':arm,'status':entry['status']})
    return {'cases':records,'files':files,'serving_progress_cases_sha256':digest(progress['cases'])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=['adaptive'])
    parser.add_argument('--case-id',action='append')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(argv);output=args.output.resolve()
    require(output!=(WORKSPACE/'outputs').resolve() and output.is_relative_to((WORKSPACE/'outputs').resolve()),'Output must be inside Stage 4 outputs')
    require(not output.is_relative_to(args.source.resolve()) and not args.source.resolve().is_relative_to(output),'New output must be separate from the source')
    if (output/'request.json').exists():
        require(args.resume,'Use --resume for the unchanged experiment')
        request=read_json(output/'request.json');authenticate(request)
        require(request['output']==str(output) and request['source']['root']==str(args.source.resolve()) and
            request['arms']==args.arms and (not args.case_id or len(set(args.case_id))==len(args.case_id) and
                set(args.case_id)=={c['case_id'] for c in request['cases']}),'Resume selection differs')
    else:
        require(not output.exists() or not any(output.iterdir()),'Unrelated nonempty output')
        request=create_request(output,args.source.resolve(),args.arms,args.case_id)
    output.mkdir(parents=True,exist_ok=True);lock=output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as stream:stream.write(str(os.getpid()))
    try:
        atomic_json(output/'request.json',request)
        progress=read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'preflight','cases':{}}
        if args.preflight_only:
            progress['phase']='preflight_complete';atomic_json(output/'progress.json',progress)
            print('V19 preflight complete; no solver or upstream inference ran:',output/'request.json',flush=True);return 0
        if not (output/'predictions_sealed.json').exists():
            progress['phase']='predictions';atomic_json(output/'progress.json',progress)
            total=len(request['cases'])*len(request['arms']);index=0
            for case in request['cases']:
                cid=case['case_id'];entries=progress['cases'].setdefault(cid,{})
                for arm in request['arms']:
                    index+=1;folder=output/'cases'/cid/arm
                    entry=entries.setdefault(arm,{'status':'pending','attempts':[]})
                    if entry['status']=='predicted':
                        require(file_hash(folder/'stage4/result.json')==entry['result_sha256'],'Committed prediction changed');continue
                    entry['status']='working';atomic_json(output/'progress.json',progress)
                    print(f'[{index}/{total}] {cid} ({case["pieces"]} pieces): V19 {arm}; Stage 4 only',flush=True)
                    attempt=run_attempt(command_for(output,request,case,arm),folder,entry,
                        lambda:atomic_json(output/'progress.json',progress))
                    if attempt['returncode']==0:
                        entry.update(status='predicted',result_sha256=file_hash(folder/'stage4/result.json'))
                    else:entry['status']='failed'
                    atomic_json(output/'progress.json',progress)
            sealed=verify_predictions(output,request,progress)
            barrier={'schema_version':'stage4.v19-recovery-prediction-seal.v1',
                'request_sha256':request['request_sha256'],**sealed,'references_opened':False}
            barrier['barrier_sha256']=digest(barrier);atomic_json(output/'predictions_sealed.json',barrier)
        progress['phase']='diagnostics';atomic_json(output/'progress.json',progress)
        attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_recovery_v19.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        progress['diagnostics']=attempt;atomic_json(output/'progress.json',progress)
        require(attempt['returncode']==0,'V19 report failed; predictions remain sealed; inspect diagnostics.log')
        progress['phase']='finished';atomic_json(output/'progress.json',progress)
        print('V19 Stage 4 comparison:',output/'report/index.html',flush=True);return 0
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
