"""Cold v9 retention experiments against both accepted historical consumers."""
import argparse
import os
from pathlib import Path
import sys
from .common import WORKSPACE,atomic_json,digest,file_hash,load_cohort,read_json
from .candidate import THREADS
from .qualified_v3 import require,select_cases
from .support_aware_experiments import checked_baseline,CONTROLLERS as V8_CONTROLLERS
from .speed_experiments import geometry_identity
from .runner import execute
from .snapshots import verify_snapshot
from .validation import safe_output
from paper_reconstruction_stage4.config import SolverConfig,RETENTION_DEFAULTS,RETAINED_POLICY
from paper_reconstruction_stage4.outputs import read_result

VARIANTS={name:'joint_v9_'+name+'_development.json' for name in ('retained','checked','capped')}
CONTROLLERS=tuple(dict.fromkeys(('benchmark/policy_retention_experiments.py','scripts/run_v9_policy_retention.py',
    'scripts/diagnose_v9_policy_retention.py','benchmark/snapshots.py',*V8_CONTROLLERS)))


def expected_variant(name,baseline):
    require(name in VARIANTS,'Unknown v9 variant')
    require(baseline['objective_version']=='observable-joint.v8-continuation' and
        baseline['joint_search_policy']=='cold-portfolio.v1' and
        baseline['joint_portfolio_component_policy']=='component-beam.v3-lookahead' and
        baseline['joint_execution_policy']=='exact-reuse.v1','V9 requires the combined v8 reference')
    expected={**baseline,**RETENTION_DEFAULTS,'joint_search_policy':RETAINED_POLICY,
        'max_proposals':40000,'max_seconds':140.,'joint_wall_seconds':160.,'joint_finalize_seconds':15.}
    if name=='checked':expected['joint_finalist_policy']='checked-incumbents.v1'
    if name=='capped':expected.update(max_proposals=20000,max_seconds=85.,joint_wall_seconds=100.,
        joint_finalize_seconds=10.,joint_retention_route_proposals=10000)
    return expected


def validate_variant(name,value,baseline):
    require(value==expected_variant(name,baseline),
        'V9 variant differs from its explicit retention/budget changes; physical limits and common objective must stay fixed')


def checked_v8(root,baseline):
    request,state,receipt=[read_json(root/p) for p in ('request.json','progress.json','experiment_receipt.json')]
    require(request['schema_version']=='stage4.v8-support-aware-request.v1' and state['phase']=='finished' and
        not (root/'run.lock').exists(),'V8 guard reference is incomplete')
    require(request['request_sha256']==receipt['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'}),
        'V8 guard request changed')
    require(Path(request['baseline']).resolve()==baseline.resolve(),'V8 guard uses a different speed baseline')
    old,_,_=checked_baseline(baseline)
    require(request['baseline_request_sha256']==old['request_sha256'] and
        request['baseline_receipt_sha256']==file_hash(baseline/'experiment_receipt.json') and
        request['integration_receipt_sha256']==file_hash(Path(request['integration'])/'integration_receipt.json'),
        'V8 reference lineage changed')
    require(file_hash(root/'report/summary.json')==receipt['summary_sha256'],'V8 guard summary changed')
    frozen=verify_snapshot(request['snapshot'])
    require(frozen['implementation']==request['implementation'],'V8 snapshot implementation differs from reference')
    for path,sha in request['controllers'].items():require(file_hash(Path(path))==sha,'V8 reference controller changed')
    config=read_json(request['configurations']['combined']['path'])
    require(file_hash(Path(request['configurations']['combined']['path']))==request['configurations']['combined']['sha256'],
        'V8 combined configuration changed')
    for cid in request['case_ids']:
        target=root/'cases'/cid/'combined/stage4';result=read_result(target)
        previous=read_result(baseline/'cases'/cid/'reuse/stage4')
        require(result['implementation']==request['implementation'] and
            result['configuration']==SolverConfig.from_dict(config).to_dict() and
            result['input']['manifest_sha256']==previous['input']['manifest_sha256'] and
            file_hash(target/'result.json')==state['cases'][cid]['combined']['result_sha256'],
            'V8 combined reference prediction changed')
    return request,SolverConfig.from_dict(config).to_dict()


def serving_barrier(request,state):
    require(state['phase'] in {'diagnostics','finished','diagnostics_failed'},'V9 serving barrier not reached')
    require(request['schema_version']=='stage4.v9-policy-retention-request.v1' and
        request['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'}),'V9 request changed')
    require(all(state['cases'].get(cid,{}).get(name,{}).get('status')=='predicted'
        for cid in request['case_ids'] for name in request['variants']),'V9 prediction barrier incomplete')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=WORKSPACE/'outputs/v6_speed_reuse_full')
    parser.add_argument('--v8-reference',type=Path,default=WORKSPACE/'outputs/v8_support_aware_guards')
    parser.add_argument('--snapshot',type=Path,default=WORKSPACE/'controls/stage4_policy_retention_v9_development')
    parser.add_argument('--case-id',action='append');parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=['retained','checked'])
    parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);baseline,v8,snapshot=args.baseline.resolve(),args.v8_reference.resolve(),args.snapshot.resolve()
    require(len(args.variants)==len(set(args.variants)),'Duplicate v9 variant')
    require(not(args.all_cases and args.case_id),'Choose --all-cases or --case-id')
    print('Checking both historical consumers and unchanged upstream evidence (no upstream inference)...',flush=True)
    old,_,original=checked_baseline(baseline);v8_request,v8_config=checked_v8(v8,baseline)
    integration=Path(old['integration'])
    cases=select_cases([c for c in original['import']['cases'] if c['case_id'] in old['case_ids']],
        None if args.all_cases else args.case_id or v8_request['case_ids'])
    frozen=verify_snapshot(snapshot);configs={name:snapshot/'configs'/VARIANTS[name] for name in args.variants}
    for name,path in configs.items():validate_variant(name,SolverConfig.from_dict(read_json(path)).to_dict(),v8_config)
    if args.check:
        print('V9 preflight passed:',len(cases),'cases;',args.variants,'; no inference or output');return 0
    require(args.output is not None,'--output is required')
    output=args.output.resolve();safe_output(output,[baseline,v8,integration,snapshot,WORKSPACE/'src',WORKSPACE/'scripts',
        WORKSPACE/'benchmark',WORKSPACE/'research',WORKSPACE/'controls',Path(original['benchmark_manifest']).parent])
    request={'schema_version':'stage4.v9-policy-retention-request.v1','baseline':str(baseline),
        'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),'baseline_request_sha256':old['request_sha256'],
        'v8_reference':str(v8),'v8_request_sha256':v8_request['request_sha256'],
        'v8_receipt_sha256':file_hash(v8/'experiment_receipt.json'),
        'v8_reference_case_ids':v8_request['case_ids'],
        'integration':str(integration),'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
        'benchmark_manifest':original['benchmark_manifest'],'snapshot':str(snapshot),'implementation':frozen['implementation'],
        'configurations':{name:{'path':str(path),'sha256':file_hash(path)} for name,path in configs.items()},
        'variants':args.variants,'case_ids':[c['case_id'] for c in cases],'output':str(output),
        'controllers':{str(WORKSPACE/p):file_hash(WORKSPACE/p) for p in CONTROLLERS},
        'cold_primaries':True,'upstream_reused':True,'fresh_pipeline':False,'saved_layout_seeds':False,
        'budget_scope':'retained/checked: 40000 proposals, 140s search, 160s wall; capped: 20000, 85s, 100s; cooperative deadlines'}
    request['request_sha256']=digest(request)
    require(not output.exists() or args.resume,'Existing v9 run: use matching --resume or a fresh output')
    output.mkdir(parents=True,exist_ok=True);lock=output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as stream:stream.write(str(os.getpid()))
    state=read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'predictions','cases':{}}
    try:
        if (output/'request.json').exists():require(read_json(output/'request.json')==request,'V9 run context changed')
        else:atomic_json(output/'request.json',request)
        state['phase']='predictions';atomic_json(output/'progress.json',state)
        for index,case in enumerate(cases,1):
            cid=case['case_id'];entry=state['cases'].setdefault(cid,{})
            prior=read_result(baseline/'cases'/cid/'reuse/stage4')
            for variant,path in configs.items():
                target=output/'cases'/cid/variant/'stage4'
                if not(target/'manifest.json').exists():
                    print(f'[{index}/{len(cases)}] {cid}/{variant}: cold retained consumer',flush=True)
                    entry[variant]=execute([sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',snapshot,'solve',
                        '--bundle',integration/'cases'/cid/'stage3','--config',path,'--output',target,'--resume'],
                        target.parent/'consumer.log',env_overrides=THREADS,echo_progress=True)
                    atomic_json(output/'progress.json',state);require(entry[variant]['returncode']==0,'V9 consumer failed')
                result=read_result(target)
                require(result['implementation']==frozen['implementation'] and
                    result['configuration']==SolverConfig.from_dict(read_json(path)).to_dict() and
                    result['input']['manifest_sha256']==prior['input']['manifest_sha256'],'V9 prediction binding differs')
                if entry.get(variant,{}).get('result_sha256'):
                    require(entry[variant]['result_sha256']==file_hash(target/'result.json'),'Sealed v9 prediction changed on resume')
                entry.setdefault(variant,{}).update(status='predicted',result_sha256=file_hash(target/'result.json'))
                atomic_json(output/'progress.json',state)
        state['phase']='diagnostics';atomic_json(output/'progress.json',state)
        state['diagnostics']=execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_v9_policy_retention.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        state['phase']='finished' if state['diagnostics']['returncode']==0 else 'diagnostics_failed'
        atomic_json(output/'progress.json',state);print('V9 retention report:',output/'report/index.html',flush=True)
        return int(state['phase']!='finished')
    except KeyboardInterrupt:
        state['phase']='interrupted';atomic_json(output/'progress.json',state);return 130
    except (ValueError,OSError,KeyError) as error:
        state.update(phase='failed',error=str(error));atomic_json(output/'progress.json',state);raise
    finally:lock.unlink(missing_ok=True)


def diagnose(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    root=parser.parse_args(argv).run.resolve();request,state=read_json(root/'request.json'),read_json(root/'progress.json')
    serving_barrier(request,state)
    for path,sha in request['controllers'].items():require(file_hash(Path(path))==sha,'V9 controller changed')
    baseline,v8,integration=map(Path,(request['baseline'],request['v8_reference'],request['integration']))
    require(file_hash(baseline/'experiment_receipt.json')==request['baseline_receipt_sha256'] and
        file_hash(v8/'experiment_receipt.json')==request['v8_receipt_sha256'] and
        file_hash(integration/'integration_receipt.json')==request['integration_receipt_sha256'],'V9 reference receipts changed')
    old,_,_=checked_baseline(baseline);older,common=checked_v8(v8,baseline)
    require(old['request_sha256']==request['baseline_request_sha256'] and older['request_sha256']==request['v8_request_sha256'] and
        older['case_ids']==request['v8_reference_case_ids'],'V9 historical reference request differs')
    frozen=verify_snapshot(request['snapshot'])
    for config in request['configurations'].values():require(file_hash(Path(config['path']))==config['sha256'],'V9 configuration changed')
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity()==request['implementation']==frozen['implementation'],'V9 diagnosis must use frozen source')
    # Evaluation/reference geometry becomes available only after the serving barrier.
    from evaluation.evaluate import evaluate
    from evaluation.pose_coverage import state_from_layout
    from .quality import assembly_quality
    from .diagnostics import selected_support_roles
    from .report import document,esc,image,table,number,link
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
    from paper_reconstruction_stage4.joint_score import JointScorer
    cohort_root,cohort=load_cohort(request['benchmark_manifest']);report=root/'report';report.mkdir(exist_ok=True)
    rows=[];cells=[];visuals=[]
    for cid in request['case_ids']:
        paths={'faster_v6':baseline/'cases'/cid/'reuse/stage4'}
        if cid in older['case_ids']:paths['combined_v8']=v8/'cases'/cid/'combined/stage4'
        historical={name:read_result(path) for name,path in paths.items()}
        previous=historical['faster_v6'];case=next(c for c in cohort['cases'] if c['case_id']==cid)
        for variant in request['variants']:
            target=root/'cases'/cid/variant/'stage4';result=read_result(target)
            require(file_hash(target/'result.json')==state['cases'][cid][variant]['result_sha256'],'V9 prediction changed')
            config=SolverConfig.from_dict(result['configuration']);validate_variant(variant,config.to_dict(),common)
            require(result['implementation']==request['implementation'] and
                config.to_dict()==SolverConfig.from_dict(read_json(request['configurations'][variant]['path'])).to_dict() and
                result['input']['manifest_sha256']==previous['input']['manifest_sha256'],'V9 comparison binding differs')
            reference={**read_json(cohort_root/case['path']/'evaluation_only/reference_geometry.json'),
                'input_manifest_sha256':result['input']['manifest_sha256']}
            problem=load_problem(integration/'cases'/cid/'stage3',config);support=ActiveSupportIndex(problem,config)
            problem.structural=JointScorer(problem,config,support)
            all_results={**historical,'candidate':result}
            quality={name:assembly_quality(r,reference,evaluate(r,reference)) for name,r in all_results.items()}
            terms={name:problem.structural.evaluate(state_from_layout(r['layouts'][0])) for name,r in all_results.items()}
            require(abs(terms['candidate']['score']-result['layouts'][0]['objective_score'])<1e-8,'V9 primary rescore differs')
            require(result['search']['proposals']<=config.max_proposals,'V9 shared proposal budget exceeded')
            row={'case_id':cid,'variant':variant,'quality':quality,'common_objective_terms':terms,
                'consumer_seconds':{name:r['resources']['elapsed_before_publication_seconds'] for name,r in all_results.items()},
                'geometry_identical_to':{name:geometry_identity(r['layouts'][0])==geometry_identity(result['layouts'][0]) for name,r in historical.items()},
                'preview_identical_to':{name:file_hash(path/'preview.png')==file_hash(target/'preview.png') for name,path in paths.items()},
                'support_verification':verify_result_support(problem,result,config),
                'selected_support_roles':selected_support_roles(problem,result,reference),
                'search':result['search'],'termination':result['termination'],'source_evidence':str(integration/'cases'/cid/'stage3'),
                'result_sha256':{name:file_hash(path/'result.json') for name,path in {**paths,'candidate':target}.items()},
                'preview_sha256':{name:file_hash(path/'preview.png') for name,path in {**paths,'candidate':target}.items()},
                'visual_review':None,'v8_reference_available':cid in older['case_ids']}
            rows.append(row);q=quality['candidate'];g=q['common_gauge']
            error=g['non_anchor_probe_errors_reference_px']['p95'] if g['whole_page_error_available'] else None
            cells.append((esc(cid),esc(variant),esc(q['component_sizes']),number(q['selected_neighbor_identity_precision']),
                number(error),number(row['consumer_seconds']['candidate'])))
            visuals.append('<h2>'+esc(cid+' / '+variant)+'</h2><p>'+link(target.parent/'consumer.log',report,'Consumer log')+
                '</p><div class="visuals">'+''.join(image(path/'preview.png',report,name) for name,path in paths.items())+
                image(target/'preview.png',report,'v9 '+variant)+'</div>')
    atomic_json(report/'summary.json',{'schema_version':'stage4.v9-policy-retention-comparison.v1','request_sha256':request['request_sha256'],
        'cases':rows,'diagnostic_implementation':implementation_identity(),'construction_requires_visual_review':True,
        'timing_scope':'Consumer only; upstream evidence reused; development preset has a larger explicit budget',
        'default_promoted':False})
    content='<h1>V9 retained cold policies</h1><p>Three independent cold search routes: older constructions, the successful v6 component policy, and v8 lookahead. Feasible incumbents survive beam pruning and all candidates share the v8 objective. Checked adds bounded native feasibility before complete candidates enter the frontier. Retained/checked use up to 40000 proposals and 140s search; capped uses 20000 and 85s. These are cooperative limits. Physical thresholds are unchanged. Correct overall construction with modest gaps is the goal; connectivity and strict four-pixel labels alone do not establish success.</p>'
    content+='<p>'+link(integration/'report/index.html',report,'Upstream integration / per-stage outputs')+' | '+link(report/'summary.json',report,'Detailed route, budget, incumbent and quality diagnostics')+'</p>'
    content+=table(['Case','Variant','Components','True-neighbor fraction','Whole-page p95 ref px','Candidate s'],cells)+''.join(visuals)
    (report/'index.html').write_text(document('V9 cold policy retention',content),encoding='utf-8')
    atomic_json(root/'experiment_receipt.json',{'schema_version':'stage4.v9-policy-retention-receipt.v1',
        'request_sha256':request['request_sha256'],'summary_sha256':file_hash(report/'summary.json'),
        'cold_primaries':True,'default_promoted':False,'status':'development_not_promoted'})
    return 0
