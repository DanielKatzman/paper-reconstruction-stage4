"""Consumer-only shared-scale experiment; historical controllers stay frozen."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from .common import WORKSPACE, DESKTOP, atomic_json, digest, file_hash, read_json
from .snapshots import freeze, verify_snapshot
from .runner import execute
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.constraints import bind_profile
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.outputs import implementation_identity, read_result
from paper_reconstruction_stage4.assessment import verify_package

SOURCE = DESKTOP/'paper-reconstruction-stage3/outputs/unit_scale_experiments/unit_20261024T125138_043110Z'
SOURCE_REQUEST = '393d1b2a6ed811d943dbf56df39ec891201e7605e7f0f2446c7376be1096feae'
CONTROL_IMPL = 'ea0cd2e4c26477b8ff9eff49dd56cfff4b812f6ef93634439b3173faef5c38b5'
CONFIG = WORKSPACE/'configs/joint_v13_component_closure_development.json'
PROFILE = WORKSPACE/'configs/shared_scale_complete_rectangle_v1.json'
SNAPSHOT = WORKSPACE/'controls/stage4_v13_component_closure_development_r1'
THREADS = {'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
CASES = ('puzzle_000330_c783e29dd569','puzzle_000450_dbaf91d313a9',
         'puzzle_000678_004586b006b3','puzzle_000917_5da663844c03')
V12_BASELINE = WORKSPACE/'outputs/v12_ambiguity_repeat'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_digest(value, field):
    require(value[field] == digest({k:v for k,v in value.items() if k != field}), 'Changed digest: '+field)


def controllers():
    # Binding the complete local controller/evaluation dependency set prevents a
    # resumed comparison from silently using changed review code.
    paths = [*WORKSPACE.glob('benchmark/*.py'), *WORKSPACE.glob('evaluation/*.py'),
             WORKSPACE/'scripts/run_component_closure_v13.py', WORKSPACE/'scripts/diagnose_component_closure_v13.py',
             WORKSPACE/'scripts/run_source_snapshot.py']
    return {str(p):file_hash(p) for p in sorted(paths)}


def preflight(source, config, profile, case_ids):
    source = Path(source).resolve(); request = read_json(source/'request.json')
    checked_digest(request, 'request_sha256')
    require(request['request_sha256'] == SOURCE_REQUEST, 'This adapter accepts only the reviewed unit-scale source run')
    barrier = read_json(source/'predictions_sealed.json'); checked_digest(barrier, 'barrier_sha256')
    require(barrier['request_sha256'] == SOURCE_REQUEST and barrier['references_opened'] is False,
            'Upstream prediction barrier differs')
    for name, sha in {**request['source_files'], **barrier['files']}.items():
        require(file_hash(name) == sha, 'Upstream sealed input/source changed: '+name)
    old = verify_snapshot(request['consumer_snapshot'])
    require(old['implementation'] == request['consumer_implementation'] and old['implementation']['sha256'] == CONTROL_IMPL,
            'Frozen v9 consumer differs')
    require(file_hash(request['consumer_config']) == '094ba854017f71df7fdda81e089f356385355f6a16f507076500582563b6b2b4', 'Frozen v9 configuration changed')
    installed = implementation_identity()
    require({k:v for k,v in installed['files'].items() if k.startswith('contract_sdk/')} ==
            {k:v for k,v in old['implementation']['files'].items() if k.startswith('contract_sdk/')}, 'Contract SDK changed')
    c = SolverConfig.from_dict(read_json(config)); cases = []
    require(set(case_ids) <= set(CASES) and len(set(case_ids)) == len(case_ids) and case_ids, 'Select unique reviewed four/eight-piece cases')
    require([r['case_id'] for r in request['cases']] == list(CASES) and [r['case_id'] for r in barrier['cases']] == list(CASES), 'Source case inventory differs')
    for cid in case_ids:
        case = next(r for r in request['cases'] if r['case_id'] == cid)
        saved = next(r for r in barrier['cases'] if r['case_id'] == cid)
        bundle = Path(saved['producer'])/'bundle'; capture = Path(case['stage1'])/'capture_profile.json'
        problem = load_problem(bundle, c); bind_profile(problem, c, profile, capture)
        control = read_result(saved['consumer'])
        require(control['implementation'] == old['implementation'] and control['input']['manifest_sha256'] == problem.manifest['manifest_sha256'] and
                control['configuration'] == read_json(request['consumer_config']) and control['evaluation_truth_opened'] is False,
                'Control result input/configuration binding differs')
        cases.append({'case_id':cid, 'pieces':len(problem.pieces), 'bundle':str(bundle),
            'bundle_sha256':problem.manifest['manifest_sha256'], 'capture_profile':str(capture),
            'constraint_binding':problem.constraint_profile, 'control_result':saved['consumer'],
            'control_result_sha256':file_hash(Path(saved['consumer'])/'result.json'),
            # This path is bound for a separate post-barrier evaluator; never read here.
            'evaluation_reference':str(Path(case['case_input'])/'evaluation_only/reference_geometry.json'),
            'evaluation_reference_sha256':next(r['sha256'] for r in case['evaluation_receipts'] if r['path']=='evaluation_only/reference_geometry.json')})
        print('Authenticate '+cid+': sealed rigid bundle and v9 control verified', flush=True)
    from .ambiguity_experiment_v12 import verify_predictions as verify_v12
    prior=read_json(V12_BASELINE/'request.json');checked_digest(prior,'request_sha256')
    prior_barrier=read_json(V12_BASELINE/'predictions_sealed.json');checked_digest(prior_barrier,'barrier_sha256')
    require(prior['request_sha256']=='0cecbe45755a3cb6046cf2b7b00b50ff57c6aa43c11a7c082b995ae7cd18b72d' and
            verify_v12(V12_BASELINE,prior)==prior_barrier['files'],'V12 comparison control differs')
    prior_receipt=read_json(V12_BASELINE/'experiment_receipt.json')
    require(prior_receipt['request_sha256']==prior['request_sha256'] and
            all(file_hash(V12_BASELINE/p)==sha for p,sha in prior_receipt['files'].items()),'V12 completed receipt changed')
    for case in cases:
        case['source_v9_control_result']=case['control_result']
        path=V12_BASELINE/'cases'/case['case_id']/'stage4';value=read_result(path)
        require(value['input']['manifest_sha256']==case['bundle_sha256'] and value['implementation']==prior['implementation'],
                'V12 baseline input/implementation differs')
        case.update(control_result=str(path),control_result_sha256=file_hash(path/'result.json'))
    return {'schema_version':'stage4.v13-unit-scale-reuse-preflight.v1', 'source':str(source),
        'source_request_sha256':SOURCE_REQUEST, 'source_barrier_sha256':barrier['barrier_sha256'],
        'sealed_source_files_checked':len(barrier['files']), 'source_controllers_checked':len(request['source_files']),
        'control_snapshot':request['consumer_snapshot'], 'control_implementation_sha256':CONTROL_IMPL,
        'comparison_control_snapshot':prior['snapshot'],'comparison_control_request_sha256':prior['request_sha256'],
        'comparison_control_receipt_sha256':file_hash(V12_BASELINE/'experiment_receipt.json'),
        'config_sha256':file_hash(config), 'profile_sha256':file_hash(profile), 'cases':cases,
        'upstream_inference_run':False, 'training_performed':False, 'evaluation_references_parsed':False}


def verify_predictions(root, request):
    frozen = verify_snapshot(request['snapshot'])
    require(frozen['implementation'] == request['implementation'], 'Experiment snapshot implementation differs')
    for name, sha in request['controllers'].items():
        require(file_hash(name) == sha, 'Experiment controller changed: '+name)
    for name, sha in request['bound_files'].items():
        require(file_hash(name) == sha, 'Experiment config/profile changed')
    files = {}
    for case in request['preflight']['cases']:
        cid = case['case_id']; target = root/'cases'/cid/'stage4'
        result = read_result(target)
        require(result['implementation'] == request['implementation'] and result['input']['manifest_sha256'] == case['bundle_sha256'] and
                result['configuration'] == request['configuration'] and result['evaluation_truth_opened'] is False and
                result['search']['constraint_profile'] == case['constraint_binding'], 'New prediction binding differs')
        require(file_hash(Path(case['control_result'])/'result.json') == case['control_result_sha256'], 'Frozen control result changed')
        for side in (target.with_name('stage4.assessment'),root/'cases'/cid/'baseline_assessment'):
            manifest = verify_package(side)
            expected = file_hash(target/'result.json') if side.name == 'stage4.assessment' else case['control_result_sha256']
            require(manifest['binding']['result_sha256'] == expected and
                    manifest['binding']['constraint_profile'] == case['constraint_binding'] and
                    manifest['binding']['assessment_implementation_sha256'] == request['implementation']['sha256'] and
                    manifest['binding']['configuration_sha256'] == digest(request['configuration']), 'Assessment binding differs')
            for path in [side/'manifest.json', *(side/name for name in manifest['files'])]:
                files[str(path)] = file_hash(path)
        for name in ('manifest.json','result.json','preview.png'):
            files[str(target/name)] = file_hash(target/name)
    return files


def main(argv=None):
    parser=argparse.ArgumentParser(description='V13: reuse four sealed unit-scale bundles, run only Stage 4')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--case-id',action='append',choices=CASES)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv); output=args.output.resolve()
    require(output.is_relative_to((WORKSPACE/'outputs').resolve()), 'V13 output must be under workspace outputs')
    check=preflight(args.source,CONFIG,PROFILE,args.case_id or list(CASES))
    frozen=freeze(SNAPSHOT,[CONFIG,PROFILE])
    config=SNAPSHOT/'configs'/CONFIG.name; profile=SNAPSHOT/'configs'/PROFILE.name
    request={'schema_version':'stage4.v13-page-experiment-request.v1','output':str(output),
        'snapshot':str(SNAPSHOT), 'implementation':frozen['implementation'], 'preflight':check,
        'configuration':SolverConfig.from_dict(read_json(config)).to_dict(), 'config':str(config),'profile':str(profile),
        'bound_files':{str(p):file_hash(p) for p in (CONFIG,PROFILE,config,profile)},
        'controllers':controllers(), 'saved_layout_seeds':False, 'fresh_pipeline':False,
        'timing_scope':'Fresh single-thread CPU Stage 4 only; Stage 1/2/3 evidence reused; assessment separately measured',
        'evaluation_truth_opened_during_predictions':False}
    request['request_sha256']=digest(request)
    if output.exists() and (output/'request.json').exists():
        require(args.resume and read_json(output/'request.json') == request, 'Existing request differs or --resume missing')
    elif output.exists():
        require(not any(output.iterdir()), 'Nonempty unrelated V13 output')
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as stream:stream.write(str(os.getpid()))
    progress=read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'preflight','cases':{}}
    try:
        atomic_json(output/'request.json',request);atomic_json(output/'preflight.json',check)
        if args.preflight_only:
            progress['phase']='preflight_complete';atomic_json(output/'progress.json',progress)
            print('V13 preflight verified; no inference/search ran:',output/'preflight.json',flush=True);return 0
        progress['phase']='predictions';atomic_json(output/'progress.json',progress)
        for index,case in enumerate(check['cases'],1):
            cid=case['case_id'];target=output/'cases'/cid/'stage4'
            entry=progress['cases'].setdefault(cid,{})
            common=['--bundle',case['bundle'],'--config',config,'--constraint-profile',profile,'--capture-profile',case['capture_profile']]
            baseline=target.parent/'baseline_assessment'
            if not (baseline/'manifest.json').exists():
                print(f'[{index}/{len(check["cases"])}] {cid}: assessment of sealed v12 primary',flush=True)
                attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',SNAPSHOT,'assess',*common,
                    '--result',case['control_result'],'--output',baseline],target.parent/'baseline_assessment.log',env_overrides=THREADS,echo_progress=True)
                entry['baseline_assessment']=attempt;require(attempt['returncode']==0,'V9 assessment failed; inspect log')
            if not (target/'manifest.json').exists() or not (target.with_name('stage4.assessment')/'manifest.json').exists():
                print(f'[{index}/{len(check["cases"])}] {cid}: cold bounded component closure Stage 4',flush=True)
                attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',SNAPSHOT,'solve',*common,
                    '--output',target,'--resume'],target.parent/'consumer.log',env_overrides=THREADS,echo_progress=True)
                entry['consumer']=attempt;require(attempt['returncode']==0,'V13 consumer failed; inspect log')
            sha=file_hash(target/'result.json')
            require(entry.get('result_sha256',sha)==sha,'Committed prediction changed on resume')
            entry['result_sha256']=sha;atomic_json(output/'progress.json',progress)
        files=verify_predictions(output,request)
        barrier={'schema_version':'stage4.v13-prediction-barrier.v1','request_sha256':request['request_sha256'],
                 'files':files,'references_opened':False}
        barrier['barrier_sha256']=digest(barrier)
        if (output/'predictions_sealed.json').exists():
            require(read_json(output/'predictions_sealed.json')==barrier,'Sealed prediction barrier changed')
        else:atomic_json(output/'predictions_sealed.json',barrier)
        progress['phase']='diagnostics';atomic_json(output/'progress.json',progress)
        attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_component_closure_v13.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        progress['diagnostics']=attempt;require(attempt['returncode']==0,'V13 diagnostics failed; sealed predictions are preserved')
        progress['phase']='finished';atomic_json(output/'progress.json',progress)
        print('V13 comparison:',output/'report/index.html',flush=True);return 0
    except BaseException as exc:
        progress.update(phase='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        atomic_json(output/'progress.json',progress);raise
    finally:lock.unlink(missing_ok=True)


def diagnose(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args(argv).run.resolve();request=read_json(root/'request.json');checked_digest(request,'request_sha256')
    barrier=read_json(root/'predictions_sealed.json');checked_digest(barrier,'barrier_sha256')
    require(barrier['request_sha256']==request['request_sha256'] and barrier['references_opened'] is False and
            verify_predictions(root,request)==barrier['files'],'Prediction barrier does not match verified outputs')
    require(implementation_identity()==request['implementation'],'Diagnosis must run inside the verified frozen implementation')
    for path,sha in barrier['files'].items():require(file_hash(path)==sha,'Sealed prediction changed')
    if (root/'experiment_receipt.json').exists():
        receipt=read_json(root/'experiment_receipt.json')
        require(receipt['request_sha256']==request['request_sha256'] and all(file_hash(root/p)==sha for p,sha in receipt['files'].items()),'Completed comparison changed')
        print('V13 completed comparison verified',flush=True);return 0
    # References are first parsed only here, after every selected prediction and
    # reference-free assessment was authenticated and sealed.
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from .report import document,esc,image,table,number,link
    report=root/'report';report.mkdir(exist_ok=True);rows=[];cells=[];visuals=[]
    for case in request['preflight']['cases']:
        cid=case['case_id'];target=root/'cases'/cid/'stage4';paths={'v12':Path(case['control_result']),'v13':target}
        values={name:read_result(path) for name,path in paths.items()}
        require(file_hash(case['evaluation_reference'])==case['evaluation_reference_sha256'],'Bound evaluation reference changed')
        reference={**read_json(case['evaluation_reference']),'input_manifest_sha256':case['bundle_sha256']}
        quality={name:assembly_quality(value,reference,evaluate(value,reference)) for name,value in values.items()}
        assessments={'v12':read_json(target.parent/'baseline_assessment/assessment.json'),
                     'v13':read_json(target.with_name('stage4.assessment')/'assessment.json')}
        feedback=read_json(target.with_name('stage4.assessment')/'pose_recovery_request.json')
        row={'case_id':cid,'pieces':case['pieces'],'quality':quality,
             'assessments':{k:v['layouts'][0] for k,v in assessments.items()},
             'new_search':values['v13']['search'],'new_resources':values['v13']['resources'],
             'fresh_stage4_subprocess':read_json(root/'progress.json')['cases'][cid].get('consumer'),
             'reused_stage3_bundle':case['bundle'],'recovery_regions':len(feedback['regions']),
             'reference_sha256':file_hash(case['evaluation_reference']),
             'visual_review':None,'four_pixel_metrics_are_diagnostic_only':True}
        gallery=write_gallery(report,case,values['v13'],request)
        closure_gallery=write_closure_gallery(report,case,values['v13'])
        row['geometry_gallery']=gallery
        row['component_placement_gallery']=closure_gallery
        rows.append(row)
        for name in ('v12','v13'):
            q=quality[name];a=assessments[name]['layouts'][0];g=q['common_gauge']
            error=g['non_anchor_probe_errors_reference_px']['p95'] if g['whole_page_error_available'] else None
            cells.append((esc(cid),esc(name),esc(q['component_sizes']),esc(a['status']),
                number(a['features'].get('fill_fraction')),number(error),esc(', '.join(a['reason_codes']))))
        visuals.append('<h2>'+esc(cid)+'</h2><div class="visuals">'+''.join(image(p/'preview.png',report,n) for n,p in paths.items())+'</div><p>'+
            link(target.with_name('stage4.assessment')/'report.md',report,'Observed assessment / recovery')+' | '+
            link(Path(gallery['index']),report,'Top-k geometry candidates and rejection reasons')+' | '+
            link(Path(closure_gallery['index']),report,'New component placements and native seam checks')+' | '+
            link(target.parent/'consumer.log',report,'Fresh CPU Stage 4 log')+'</p>')
    atomic_json(report/'summary.json',{'schema_version':'stage4.v13-page-comparison.v1','request_sha256':request['request_sha256'],
        'barrier_sha256':barrier['barrier_sha256'],'cases':rows,'reference_access':'post-prediction diagnostic process only',
        'default_promoted':False,'construction_requires_visual_review':True,'prepared_12_16_cases_run':False})
    content='<h1>Shared-scale complete-page comparison</h1><p>Fresh cold Stage 4 search on sealed unit-scale inputs. Page format and DPI are unknown. V12 predictions are unchanged; their assessments are prospective replays. Correct overall construction with modest seams is the goal. A supported assessment is uncalibrated and needs visual review; rejecting a bad assembly is not a solved puzzle. Stage 2/3 inference was reused.</p>'
    content+=table(['Case','Policy','Components','Assessment','Mask fill','Page p95 ref px','Reasons'],cells)+''.join(visuals)
    (report/'index.html').write_text(document('V13 component closure comparison',content),encoding='utf-8')
    receipt={'schema_version':'stage4.v13-page-comparison-receipt.v1','request_sha256':request['request_sha256'],
        'barrier_sha256':barrier['barrier_sha256'],'status':'development_not_promoted',
        'files':{str(p.relative_to(root)).replace('\\','/'):file_hash(p) for p in report.rglob('*') if p.is_file()}}
    atomic_json(root/'experiment_receipt.json',receipt);return 0


def write_gallery(report,case,result,request):
    """Render sealed candidate records only; no reference ranking or new solve."""
    from paper_reconstruction_stage4.page_geometry import record_state
    from paper_reconstruction_stage4.preview import render
    from .report import document,table,esc,number,image,link
    config=SolverConfig.from_dict(request['configuration']);problem=load_problem(case['bundle'],config)
    from paper_reconstruction_stage4.native_bridge import install_result_bridges
    problem.constraint_profile=case['constraint_binding'];install_result_bridges(problem,result,config)
    rows=result['search']['geometry_candidate_audit']['candidates']
    selected=result['search']['geometry_candidate_audit']['selected_candidate_id']
    ranked=sorted(rows,key=lambda r:(r['candidate_id']!=selected,r['disposition']!='eligible',not r['complete'],
        -(r.get('geometry',{}).get('score') or 0.),-r.get('selection_utility',r['source_score']),r['candidate_id']))
    path=report/'candidates'/case['case_id'];path.mkdir(parents=True,exist_ok=True)
    cells=[];visuals=[]
    history={r['candidate_id']:r for r in result['search'].get('page_candidate_retention',{}).get('retained_candidates',[])}
    for index,row in enumerate(ranked,1):
        geometry=row.get('geometry',{});pic=path/f'candidate_{index:03d}.png'
        if index<=50:
            render(problem,record_state(row),pic,max_side=1000)
            visuals.append('<details><summary>'+esc(f"Candidate {index}: {row['disposition']}, geometry {geometry.get('score')}")+'</summary>'+image(pic,path,'Sealed candidate '+str(index))+'</details>')
        oldrank=history.get(row['candidate_id'],{}).get('old_score_rank_among_measured_completions',row.get('source_rank_in_final_union'))
        cells.append((esc(index),esc(row['disposition']),number(geometry.get('score')),esc(geometry.get('status')),
            esc(oldrank),number(row['source_score']),number(row.get('selection_utility')),number(row.get('ambiguity',{}).get('risk_cost')),
            esc(len(row.get('ambiguity',{}).get('unresolved_geometry_ids',[]))),esc(', '.join(geometry.get('reason_codes',[]))),
            link(pic,path,'preview') if index<=50 else esc('recorded in JSON')))
    atomic_json(path/'candidates.json',{'source_run_id':result['run_id'],'candidates':ranked,
        'rank_semantics':'Old rank among measured streaming completions when retained there, otherwise among final source states; not global k-best',
        'reference_used_for_gallery_order':False,'evaluated_count':len(ranked)})
    content='<h1>'+esc(case['case_id'])+' geometry candidate review</h1><p>The union retains up to 50 solver-score families, 50 geometry families, extra topology diversity and 16 uncertainty-aware partial alternatives before old pruning. All reported checks use observations only. Non-page complete candidates are rejected. Budget-abstained candidates remain explicitly unverified. This is a bounded generated pool, not a certified global top-k. Geometry cannot guarantee correct content order. The first 50 records have previews.</p>'
    content+=table(['Candidate','Disposition','Geometry score','Geometry gate','Old score rank','Solver score','Selection utility','Ambiguity risk','Uncertain joins','Reasons','Preview'],cells)+''.join(visuals)
    (path/'index.html').write_text(document('Geometry candidates',content),encoding='utf-8')
    return {'index':str(path/'index.html'),'candidates_json':str(path/'candidates.json'),
        'records':len(ranked),'previews':min(50,len(ranked)),
        'geometry_rejected':sum(r['disposition']=='rejected_geometry' for r in ranked),
        'unverified':sum(r['disposition']=='unverified_budget' for r in ranked)}


def write_closure_gallery(report,case,result):
    from .report import document,table,esc,number,link
    audit=result['search']['component_closure']
    path=report/'component_placements'/case['case_id'];path.mkdir(parents=True,exist_ok=True)
    atomic_json(path/'placements.json',{'source_run_id':result['run_id'],'audit':audit,
        'reference_used_for_order_or_verification':False})
    rows=[]
    for rank,row in enumerate(audit['proposals'],1):
        bridges=row.get('bridges',[])
        rows.append((esc(rank),esc(row['disposition']),number(row['coarse_geometry']['score']),
            number(row.get('fine_geometry',{}).get('fill_fraction')),esc(len(bridges)),
            esc(', '.join('/'.join(b['pieces']) for b in bridges)),
            esc(row.get('reason','not native-verified' if row['disposition']=='coarse_only' else ''))))
    content='<h1>'+esc(case['case_id'])+' bounded component placement</h1>'
    content+='<p>These are newly generated relative poses from observed masks and a complete rectangular-page assumption with shared native scale. No page format, DPI or reference coordinates were used. Coarse-only candidates are unverified hypotheses. A join requires at least two distinct, unoccupied, contiguous native seam pairs, opposing interiors and final collision checks. This first version places two existing components and preserves their internal poses.</p>'
    content+=table(['Generated','Fine checked','Verified closures','Route seconds','Source families omitted','Budget abstention'],[
        (esc(audit['generated_proposals']),esc(audit['fine_checks']),esc(audit['verified_closures']),
         number(audit['elapsed_seconds']),esc(audit['sources_omitted_by_limit']),esc(audit['budget_abstained']))])
    content+='<p>'+link(path/'placements.json',path,'Sealed poses, source groups, shape scores and native bridge measurements')+'</p>'
    content+=table(['Coarse rank','Disposition','Coarse geometry','Fine fill','Native seam pairs','Pairs','Reason'],rows)
    (path/'index.html').write_text(document('Component placement verification',content),encoding='utf-8')
    return {'index':str(path/'index.html'),'generated':audit['generated_proposals'],
            'fine_checked':audit['fine_checks'],'verified_closures':audit['verified_closures']}
