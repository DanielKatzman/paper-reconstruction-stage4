"""Separate continuation-score ablations against the frozen faster consumer."""
import argparse
import os
from pathlib import Path
import sys
from .common import WORKSPACE,atomic_json,digest,file_hash,load_cohort,read_json
from .candidate import THREADS
from .qualified_v3 import require,select_cases
from .speed_experiments import checked_baseline as checked_conservative,geometry_identity,CONTROLLERS as SPEED_CONTROLLERS
from .runner import execute
from .snapshots import verify_snapshot
from .validation import safe_output
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.outputs import read_result

VARIANTS={'continuation':('joint_continuation_development.json',1.25),
    'mild':('joint_continuation_mild_development.json',.5),'off':('joint_continuation_off_development.json',0.)}
CONTROLLERS=tuple(dict.fromkeys(('benchmark/accuracy_experiments.py','scripts/run_v7_accuracy_experiments.py',
    'scripts/diagnose_v7_accuracy.py',*SPEED_CONTROLLERS)))


def validate_accuracy_config(name,value,baseline):
    expected={**baseline,'objective_version':'observable-joint.v7-continuation',
        'continuation_weight':VARIANTS[name][1],'continuation_min_coverage':.60,
        'continuation_samples':65,'continuation_image_side':768}
    require(baseline['objective_version']=='observable-joint.v6-attachment' and
        baseline.get('joint_execution_policy')=='exact-reuse.v1','Accuracy baseline must be the frozen faster consumer')
    require(value==expected,'Accuracy experiment changed execution, search, physical limits or unrelated score policy')


def checked_baseline(root):
    request,state,receipt=[read_json(root/name) for name in ('request.json','progress.json','experiment_receipt.json')]
    require(request['schema_version']=='stage4.v6-speed-request.v1' and state['phase']=='finished' and
        not (root/'run.lock').exists(),'Speed baseline is incomplete')
    require(request['request_sha256']==receipt['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'}),
        'Speed baseline request changed')
    require(file_hash(root/'report/summary.json')==receipt['summary_sha256'],'Speed baseline report changed')
    verify_snapshot(request['snapshot'])
    for path,sha in request['controllers'].items():require(file_hash(Path(path))==sha,'Speed baseline controller changed')
    previous,_,original=checked_conservative(Path(request['baseline']))
    require(previous['request_sha256']==request['baseline_request_sha256'] and
        file_hash(Path(request['baseline'])/'experiment_receipt.json')==request['baseline_receipt_sha256'],
        'Conservative source of speed baseline changed')
    require(file_hash(Path(request['integration'])/'integration_receipt.json')==request['integration_receipt_sha256'],
        'Accuracy input integration changed')
    for cid in request['case_ids']:
        target=root/'cases'/cid/'reuse/stage4';result=read_result(target)
        require(file_hash(target/'result.json')==state['cases'][cid]['result_sha256'] and
            result['implementation']==request['implementation'],'Speed baseline prediction changed')
    return request,state,original


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=WORKSPACE/'outputs/v6_speed_reuse_full')
    parser.add_argument('--snapshot',type=Path,default=WORKSPACE/'controls/stage4_continuation_v7_development')
    parser.add_argument('--case-id',action='append');parser.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=['continuation'])
    parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);baseline,snapshot=args.baseline.resolve(),args.snapshot.resolve()
    require(len(args.variants)==len(set(args.variants)),'Duplicate accuracy variant')
    print('Checking saved evidence, speed baseline and accuracy snapshot (no upstream inference)...',flush=True)
    old,_,original=checked_baseline(baseline);integration=Path(old['integration'])
    cases=select_cases([c for c in original['import']['cases'] if c['case_id'] in old['case_ids']],args.case_id)
    frozen=verify_snapshot(snapshot);configs={name:snapshot/'configs'/VARIANTS[name][0] for name in args.variants}
    for case in cases:
        prior=read_result(baseline/'cases'/case['case_id']/'reuse/stage4')
        for name,path in configs.items():validate_accuracy_config(name,SolverConfig.from_dict(read_json(path)).to_dict(),prior['configuration'])
    if args.check:
        print('Accuracy preflight passed:',len(cases),'cases;',args.variants,'; no inference or output');return 0
    require(args.output is not None,'--output is required')
    output=args.output.resolve();safe_output(output,[baseline,integration,snapshot,WORKSPACE/'src',WORKSPACE/'scripts',
        WORKSPACE/'benchmark',WORKSPACE/'research',WORKSPACE/'controls',Path(original['benchmark_manifest']).parent])
    request={'schema_version':'stage4.v7-accuracy-request.v1','baseline':str(baseline),
        'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),'baseline_request_sha256':old['request_sha256'],
        'integration':str(integration),'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
        'benchmark_manifest':original['benchmark_manifest'],'snapshot':str(snapshot),'implementation':frozen['implementation'],
        'configurations':{name:{'path':str(path),'sha256':file_hash(path)} for name,path in configs.items()},
        'variants':args.variants,'case_ids':[c['case_id'] for c in cases],'output':str(output),
        'controllers':{str(WORKSPACE/p):file_hash(WORKSPACE/p) for p in CONTROLLERS},
        'cold_primaries':True,'upstream_reused':True,'fresh_pipeline':False,'execution_policy_changed':False}
    request['request_sha256']=digest(request)
    require(not output.exists() or args.resume,'Existing accuracy run: use matching --resume or a fresh output')
    output.mkdir(parents=True,exist_ok=True);lock=output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as stream:stream.write(str(os.getpid()))
    state=read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'predictions','cases':{}}
    try:
        if (output/'request.json').exists():require(read_json(output/'request.json')==request,'Accuracy run context changed')
        else:atomic_json(output/'request.json',request)
        state['phase']='predictions';atomic_json(output/'progress.json',state)
        for index,case in enumerate(cases,1):
            cid=case['case_id'];entry=state['cases'].setdefault(cid,{})
            prior=read_result(baseline/'cases'/cid/'reuse/stage4')
            for variant,path in configs.items():
                target=output/'cases'/cid/variant/'stage4'
                if not (target/'manifest.json').exists():
                    print(f'[{index}/{len(cases)}] {cid}/{variant}: cold accuracy consumer',flush=True)
                    entry[variant]=execute([sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',snapshot,'solve',
                        '--bundle',integration/'cases'/cid/'stage3','--config',path,'--output',target,'--resume'],
                        target.parent/'consumer.log',env_overrides=THREADS,echo_progress=True)
                    atomic_json(output/'progress.json',state);require(entry[variant]['returncode']==0,'Accuracy consumer failed')
                result=read_result(target)
                require(result['implementation']==frozen['implementation'] and
                    result['configuration']==SolverConfig.from_dict(read_json(path)).to_dict() and
                    result['input']['manifest_sha256']==prior['input']['manifest_sha256'],'Accuracy prediction binding differs')
                entry.setdefault(variant,{}).update(status='predicted',result_sha256=file_hash(target/'result.json'))
                atomic_json(output/'progress.json',state)
        state['phase']='diagnostics';atomic_json(output/'progress.json',state)
        state['diagnostics']=execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_v7_accuracy.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        state['phase']='finished' if state['diagnostics']['returncode']==0 else 'diagnostics_failed'
        atomic_json(output/'progress.json',state);print('Accuracy report:',output/'report/index.html',flush=True)
        return int(state['phase']!='finished')
    except KeyboardInterrupt:
        state['phase']='interrupted';atomic_json(output/'progress.json',state);return 130
    except (ValueError,OSError,KeyError) as error:
        state.update(phase='failed',error=str(error));atomic_json(output/'progress.json',state);raise
    finally:lock.unlink(missing_ok=True)


def diagnose(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    root=parser.parse_args(argv).run.resolve();request,state=read_json(root/'request.json'),read_json(root/'progress.json')
    require(state['phase'] in {'diagnostics','finished','diagnostics_failed'},'Accuracy serving barrier not reached')
    require(request['schema_version']=='stage4.v7-accuracy-request.v1' and
        request['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'}),'Accuracy request changed')
    require(all(state['cases'].get(cid,{}).get(name,{}).get('status')=='predicted'
        for cid in request['case_ids'] for name in request['variants']),'Accuracy prediction barrier incomplete')
    for path,sha in request['controllers'].items():require(file_hash(Path(path))==sha,'Accuracy controller changed')
    baseline,integration=Path(request['baseline']),Path(request['integration'])
    require(file_hash(baseline/'experiment_receipt.json')==request['baseline_receipt_sha256'] and
        file_hash(integration/'integration_receipt.json')==request['integration_receipt_sha256'],'Accuracy source receipts changed')
    old,_,_=checked_baseline(baseline);require(old['request_sha256']==request['baseline_request_sha256'],'Accuracy baseline request differs')
    frozen=verify_snapshot(request['snapshot'])
    for name,config in request['configurations'].items():require(file_hash(Path(config['path']))==config['sha256'],'Accuracy configuration changed')
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity()==request['implementation']==frozen['implementation'],'Accuracy diagnosis must use frozen prediction source')
    # Reference access starts only after every selected prediction is sealed.
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
        prior=baseline/'cases'/cid/'reuse/stage4';previous=read_result(prior)
        case=next(c for c in cohort['cases'] if c['case_id']==cid)
        for variant in request['variants']:
            target=root/'cases'/cid/variant/'stage4';result=read_result(target)
            require(file_hash(target/'result.json')==state['cases'][cid][variant]['result_sha256'],'Accuracy prediction changed')
            config=SolverConfig.from_dict(result['configuration']);validate_accuracy_config(variant,config.to_dict(),previous['configuration'])
            require(result['implementation']==request['implementation'] and
                config.to_dict()==SolverConfig.from_dict(read_json(request['configurations'][variant]['path'])).to_dict() and
                result['input']['manifest_sha256']==previous['input']['manifest_sha256'],'Accuracy comparison binding differs')
            reference={**read_json(cohort_root/case['path']/'evaluation_only/reference_geometry.json'),
                'input_manifest_sha256':result['input']['manifest_sha256']}
            quality=[assembly_quality(r,reference,evaluate(r,reference)) for r in (previous,result)]
            problem=load_problem(integration/'cases'/cid/'stage3',config);support=ActiveSupportIndex(problem,config)
            problem.structural=JointScorer(problem,config,support)
            terms=[problem.structural.evaluate(state_from_layout(r['layouts'][0])) for r in (previous,result)]
            require(abs(terms[1]['score']-result['layouts'][0]['objective_score'])<1e-8,'Accuracy primary rescore differs')
            require(abs(terms[0]['score']+terms[0]['continuation_evidence']['cost']-previous['layouts'][0]['objective_score'])<1e-8,
                'Accuracy baseline changed beyond the declared continuation term')
            row={'case_id':cid,'variant':variant,'quality_baseline':quality[0],'quality_accuracy':quality[1],
                'same_objective_scores':[t['score'] for t in terms],'score_terms':terms,
                'consumer_seconds':[r['resources']['elapsed_before_publication_seconds'] for r in (previous,result)],
                'primary_geometry_identical':geometry_identity(previous['layouts'][0])==geometry_identity(result['layouts'][0]),
                'preview_bytes_identical':file_hash(prior/'preview.png')==file_hash(target/'preview.png'),
                'support_verification':verify_result_support(problem,result,config),
                'selected_support_roles':selected_support_roles(problem,result,reference),
                'search':result['search'],'termination':result['termination'],'proposals':result['search']['proposals'],
                'result_sha256':[file_hash(p/'result.json') for p in (prior,target)],
                'preview_sha256':[file_hash(p/'preview.png') for p in (prior,target)],'visual_review':None}
            row['source_evidence']=str(integration/'cases'/cid/'stage3')
            require(row['proposals']<=config.max_proposals,'Accuracy proposal budget exceeded')
            rows.append(row);g=quality[-1]['common_gauge'];error=g['non_anchor_probe_errors_reference_px']['p95'] if g['whole_page_error_available'] else None
            cells.append((esc(cid),esc(variant),esc(quality[0]['component_sizes']),esc(quality[1]['component_sizes']),
                number(quality[1]['selected_neighbor_identity_precision']),number(error),number(row['consumer_seconds'][1])))
            visuals.append('<h2>'+esc(cid+' / '+variant)+'</h2><p>'+link(target.parent/'consumer.log',report,'Consumer log')+' | '+
                link(integration/'cases'/cid/'stage3',report,'Reused Stage 3 evidence')+'</p><div class="visuals">'+
                image(prior/'preview.png',report,'Frozen speed baseline')+image(target/'preview.png',report,'Cold accuracy candidate')+'</div>')
    atomic_json(report/'summary.json',{'schema_version':'stage4.v7-accuracy-comparison.v1','request_sha256':request['request_sha256'],
        'cases':rows,'diagnostic_implementation':implementation_identity(),'construction_requires_visual_review':True,
        'timing_scope':'Consumer only; upstream evidence reused','default_promoted':False})
    content='<h1>Local continuation accuracy experiment</h1><p>Cold consumer runs on the same saved evidence and execution policy. A separate soft colour/ink cue applies to underconstrained, text-weak bridges. Blank, insufficiently visible or unreliable-normalization samples abstain. Connectivity and scores do not establish construction success; compare the accepted twelve-piece arrangement and every primary preview.</p>'
    content+='<p>'+link(integration/'report/index.html',report,'Upstream integration and original stage outputs')+'</p>'
    content+=table(['Case','Variant','Baseline components','Candidate components','True-neighbor fraction','Whole-page p95 ref px','Candidate s'],cells)+''.join(visuals)
    (report/'index.html').write_text(document('Accuracy experiment',content),encoding='utf-8')
    atomic_json(root/'experiment_receipt.json',{'schema_version':'stage4.v7-accuracy-receipt.v1','request_sha256':request['request_sha256'],
        'summary_sha256':file_hash(report/'summary.json'),'cold_primaries':True,'default_promoted':False,'status':'development_not_promoted'})
    return 0
