"""Cold speed comparison against the completed conservative construction run."""
import argparse
import os
from pathlib import Path
import sys
from .common import WORKSPACE,atomic_json,digest,file_hash,load_cohort,read_json
from .candidate import THREADS
from .qualified_v3 import require,select_cases
from .quality_experiments import checked_integration
from .runner import execute
from .snapshots import verify_snapshot
from .validation import safe_output
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.outputs import read_result

CONFIG = 'joint_speed_reuse_development.json'
CONTROLLERS = ('benchmark/speed_experiments.py','benchmark/snapshot_diagnosis.py',
    'benchmark/quality_experiments.py','benchmark/qualified_v3.py','benchmark/common.py',
    'benchmark/runner.py','benchmark/candidate.py','benchmark/report.py','benchmark/quality.py',
    'benchmark/diagnostics.py','benchmark/validation.py','evaluation/evaluate.py','evaluation/pose_coverage.py',
    'scripts/run_v6_speed_experiments.py','scripts/diagnose_v6_speed.py','scripts/run_source_snapshot.py')


def validate_speed_config(value, baseline):
    expected = {**baseline,'joint_execution_policy':'exact-reuse.v1'}
    require(value == expected,'Speed experiment must change only the execution policy, not score/search/budgets')
    require(value['objective_version']=='observable-joint.v6-attachment' and
        value['joint_search_policy']=='cold-portfolio.v1' and value['joint_portfolio_component_policy']=='component-beam.v1' and
        value['attachment_uncertainty_cost']==0.,'Speed baseline must be the conservative construction policy')


def checked_baseline(root):
    request,state,receipt = [read_json(root/name) for name in ('request.json','progress.json','experiment_receipt.json')]
    require(request['schema_version']=='stage4.v6-retention-request.v1' and request['variants']==['conservative'] and
        state['phase']=='finished' and not (root/'run.lock').exists(),'Conservative baseline is not complete')
    require(request['request_sha256']==receipt['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'}),
        'Conservative baseline request changed')
    require(file_hash(root/'report/summary.json')==receipt['summary_sha256'],'Conservative baseline report changed')
    verify_snapshot(request['snapshot'])
    for path,sha in request['controllers'].items():
        require(file_hash(Path(path))==sha,'Conservative baseline controller changed')
    integration=Path(request['integration'])
    original=checked_integration(integration)
    require(file_hash(integration/'integration_receipt.json')==request['integration_receipt_sha256'],'Baseline integration changed')
    for cid in request['case_ids']:
        result=read_result(root/'cases'/cid/'conservative/stage4')
        require(result['implementation']==request['implementation'] and
            file_hash(root/'cases'/cid/'conservative/stage4/result.json')==state['cases'][cid]['conservative']['result_sha256'],
            'Conservative baseline prediction changed')
    return request,state,original


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=WORKSPACE/'outputs/v6_conservative_full')
    parser.add_argument('--snapshot',type=Path,default=WORKSPACE/'controls/stage4_speed_reuse_v1')
    parser.add_argument('--case-id',action='append')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    baseline,snapshot=args.baseline.resolve(),args.snapshot.resolve()
    print('Checking saved inputs, conservative baseline and speed snapshot (no upstream inference)...',flush=True)
    old,old_state,original=checked_baseline(baseline)
    cases=select_cases([c for c in original['import']['cases'] if c['case_id'] in old['case_ids']],args.case_id)
    frozen=verify_snapshot(snapshot);config=snapshot/'configs'/CONFIG
    value=SolverConfig.from_dict(read_json(config)).to_dict()
    for case in cases:
        cid=case['case_id'];previous=read_result(baseline/'cases'/cid/'conservative/stage4')
        validate_speed_config(value,previous['configuration'])
    if args.check:
        print('Speed preflight passed:',len(cases),'cases; score/search/physical limits unchanged; no inference or output')
        return 0
    require(args.output is not None,'--output is required')
    output=args.output.resolve();integration=Path(old['integration'])
    safe_output(output,[baseline,integration,snapshot,WORKSPACE/'src',WORKSPACE/'scripts',WORKSPACE/'benchmark',
        WORKSPACE/'research',WORKSPACE/'controls',Path(original['benchmark_manifest']).parent])
    request={'schema_version':'stage4.v6-speed-request.v1','baseline':str(baseline),
        'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),'baseline_request_sha256':old['request_sha256'],
        'integration':str(integration),'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
        'benchmark_manifest':original['benchmark_manifest'],'snapshot':str(snapshot),'implementation':frozen['implementation'],
        'configuration':{'path':str(config),'sha256':file_hash(config)},'output':str(output),
        'case_ids':[c['case_id'] for c in cases],'controllers':{str(WORKSPACE/p):file_hash(WORKSPACE/p) for p in CONTROLLERS},
        'cold_primaries':True,'upstream_reused':True,'fresh_pipeline':False,'accuracy_policy_changed':False}
    request['request_sha256']=digest(request)
    require(not output.exists() or args.resume,'Existing speed run: use matching --resume or a fresh output')
    output.mkdir(parents=True,exist_ok=True);lock=output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as stream:stream.write(str(os.getpid()))
    state=read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'predictions','cases':{}}
    try:
        if (output/'request.json').exists():require(read_json(output/'request.json')==request,'Speed run context changed')
        else:atomic_json(output/'request.json',request)
        state['phase']='predictions';atomic_json(output/'progress.json',state)
        for index,case in enumerate(cases,1):
            cid=case['case_id'];target=output/'cases'/cid/'reuse/stage4'
            entry=state['cases'].setdefault(cid,{})
            if not (target/'manifest.json').exists():
                print(f'[{index}/{len(cases)}] {cid}: cold Stage 4 exact reuse',flush=True)
                attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',snapshot,'solve',
                    '--bundle',integration/'cases'/cid/'stage3','--config',config,'--output',target,'--resume'],
                    target.parent/'consumer.log',env_overrides=THREADS,echo_progress=True)
                entry['consumer']=attempt;atomic_json(output/'progress.json',state)
                require(attempt['returncode']==0,'Speed consumer failed')
            result=read_result(target);previous=read_result(baseline/'cases'/cid/'conservative/stage4')
            require(result['implementation']==frozen['implementation'] and result['configuration']==value and
                result['input']['manifest_sha256']==previous['input']['manifest_sha256'],'Speed prediction binding differs')
            entry.update(status='predicted',result_sha256=file_hash(target/'result.json'))
            atomic_json(output/'progress.json',state)
        state['phase']='diagnostics';atomic_json(output/'progress.json',state)
        state['diagnostics']=execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_v6_speed.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        state['phase']='finished' if state['diagnostics']['returncode']==0 else 'diagnostics_failed'
        atomic_json(output/'progress.json',state)
        print('Speed report:',output/'report/index.html',flush=True)
        return int(state['phase']!='finished')
    except KeyboardInterrupt:
        state['phase']='interrupted';atomic_json(output/'progress.json',state);return 130
    except (ValueError,OSError,KeyError) as error:
        state.update(phase='failed',error=str(error));atomic_json(output/'progress.json',state);raise
    finally:lock.unlink(missing_ok=True)


def geometry_identity(layout):
    return digest({k:layout[k] for k in ('pieces','components','selected_joins')})


def diagnose(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args(argv).run.resolve()
    request,state=read_json(root/'request.json'),read_json(root/'progress.json')
    require(state['phase'] in {'diagnostics','finished','diagnostics_failed'},'Speed serving barrier not reached')
    require(request['schema_version']=='stage4.v6-speed-request.v1' and
        request['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'}),'Speed request changed')
    require(all(state['cases'].get(cid,{}).get('status')=='predicted' for cid in request['case_ids']),'Speed prediction barrier incomplete')
    for path,sha in request['controllers'].items():require(file_hash(Path(path))==sha,'Speed controller changed')
    baseline,integration=Path(request['baseline']),Path(request['integration'])
    require(file_hash(baseline/'experiment_receipt.json')==request['baseline_receipt_sha256'] and
        file_hash(integration/'integration_receipt.json')==request['integration_receipt_sha256'],'Speed source receipts changed')
    old,old_state,original=checked_baseline(baseline)
    require(old['request_sha256']==request['baseline_request_sha256'],'Speed baseline request differs')
    frozen=verify_snapshot(request['snapshot'])
    require(file_hash(Path(request['configuration']['path']))==request['configuration']['sha256'],'Speed configuration changed')
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity()==frozen['implementation']==request['implementation'],'Speed diagnosis requires frozen prediction source')
    # References/evaluator imports occur only after every selected cold primary
    # is sealed. Nothing below is an input to serving search or arm selection.
    from evaluation.evaluate import evaluate
    from evaluation.pose_coverage import state_from_layout
    from .quality import assembly_quality
    from .diagnostics import selected_support_roles
    from .report import document,esc,image,table,number,link
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
    from paper_reconstruction_stage4.joint_score import JointScorer
    cohort_root,cohort=load_cohort(request['benchmark_manifest'])
    report=root/'report';report.mkdir(exist_ok=True);rows=[];cells=[];visuals=[]
    for cid in request['case_ids']:
        prior=baseline/'cases'/cid/'conservative/stage4';target=root/'cases'/cid/'reuse/stage4'
        old_result,new=read_result(prior),read_result(target)
        require(file_hash(target/'result.json')==state['cases'][cid]['result_sha256'],'Speed prediction changed')
        require(new['implementation']==request['implementation'],'Speed prediction source differs')
        config=SolverConfig.from_dict(new['configuration']);validate_speed_config(config.to_dict(),old_result['configuration'])
        require(config.to_dict()==SolverConfig.from_dict(read_json(request['configuration']['path'])).to_dict(),'Speed config binding differs')
        require(new['input']['manifest_sha256']==old_result['input']['manifest_sha256'],'Speed inputs differ')
        case=next(c for c in cohort['cases'] if c['case_id']==cid)
        reference={**read_json(cohort_root/case['path']/'evaluation_only/reference_geometry.json'),
            'input_manifest_sha256':new['input']['manifest_sha256']}
        quality=[assembly_quality(r,reference,evaluate(r,reference)) for r in (old_result,new)]
        problem=load_problem(integration/'cases'/cid/'stage3',config);support=ActiveSupportIndex(problem,config)
        problem.structural=JointScorer(problem,config,support)
        scores=[problem.structural.evaluate(state_from_layout(r['layouts'][0]))['score'] for r in (old_result,new)]
        require(all(abs(value-r['layouts'][0]['objective_score'])<1e-8 for value,r in zip(scores,(old_result,new))),
            'Execution-only change altered a sealed objective score')
        require(new['search']['proposals']<=config.max_proposals,'Speed proposal budget exceeded')
        row={'case_id':cid,'quality_baseline':quality[0],'quality_reuse':quality[1],
            'consumer_seconds':[r['resources']['elapsed_before_publication_seconds'] for r in (old_result,new)],
            'primary_geometry_identical':geometry_identity(old_result['layouts'][0])==geometry_identity(new['layouts'][0]),
            'preview_bytes_identical':file_hash(prior/'preview.png')==file_hash(target/'preview.png'),
            'same_objective_scores':scores,'support_verification':verify_result_support(problem,new,config),
            'selected_support_roles':selected_support_roles(problem,new,reference),
            'execution_reuse':new['search'].get('execution_reuse'), 'search':new['search'],
            'termination':new['termination'],'proposals':new['search']['proposals'],
            'result_sha256':[file_hash(p/'result.json') for p in (prior,target)],
            'preview_sha256':[file_hash(p/'preview.png') for p in (prior,target)],
            'source_evidence':str(integration/'cases'/cid/'stage3'),'visual_review':None}
        rows.append(row);g=quality[-1]['common_gauge']
        error=g['non_anchor_probe_errors_reference_px']['p95'] if g['whole_page_error_available'] else None
        cells.append((esc(cid),esc(quality[0]['component_sizes']),esc(quality[1]['component_sizes']),
            esc(row['preview_bytes_identical']),number(error),number(row['consumer_seconds'][0]),
            number(row['consumer_seconds'][1]),esc(new['termination'])))
        visuals.append('<h2>'+esc(cid)+'</h2><p>'+link(target.parent/'consumer.log',report,'Consumer log')+' | '+
            link(integration/'cases'/cid/'stage3',report,'Reused Stage 3 evidence')+'</p><div class="visuals">'+
            image(prior/'preview.png',report,'Conservative baseline')+image(target/'preview.png',report,'Exact-reuse candidate')+'</div>')
    atomic_json(report/'summary.json',{'schema_version':'stage4.v6-speed-comparison.v1','request_sha256':request['request_sha256'],
        'cases':rows,'diagnostic_implementation':implementation_identity(),'accuracy_policy_changed':False,
        'construction_requires_visual_review':True,'default_promoted':False,
        'timing_scope':'Recorded consumer runs on saved upstream evidence; not alternating controlled speed trials'})
    content='<h1>Stage 4 speed comparison</h1><p>Same conservative objective, search limits and physical checks. Exact repeated proposal work is reused; scores are evaluated under each requesting policy. Saved Stage 2/3 outputs are reused. Reference diagnostics run only after all cold predictions finish. Time-limited search may explore different candidates when execution gets faster; review construction quality as well as latency.</p>'
    content+=table(['Case','Baseline components','Reuse components','Exact preview','Whole-page p95 ref px','Baseline s','Reuse s','Termination'],cells)+''.join(visuals)
    (report/'index.html').write_text(document('Stage 4 speed comparison',content),encoding='utf-8')
    atomic_json(root/'experiment_receipt.json',{'schema_version':'stage4.v6-speed-receipt.v1',
        'request_sha256':request['request_sha256'],'summary_sha256':file_hash(report/'summary.json'),
        'accuracy_policy_changed':False,'cold_primaries':True,'default_promoted':False,'status':'development_not_promoted'})
    return 0
