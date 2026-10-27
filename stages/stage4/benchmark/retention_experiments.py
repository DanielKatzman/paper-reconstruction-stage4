"""Cold v6 retention/attachment ablations using authenticated imported v3 inputs."""
import argparse
import os
from pathlib import Path
import sys
from .common import WORKSPACE, atomic_json, digest, file_hash, load_cohort, read_json
from .candidate import THREADS
from .qualified_v3 import require, select_cases
from .quality_experiments import checked_integration
from .runner import execute
from .snapshots import verify_snapshot
from .validation import safe_output
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.outputs import read_result

VARIANTS = {
    'retention': 'joint_cold_retention_development.json',
    'exploration': 'joint_component_diversity_development.json',
    'attachment': 'joint_attachment_uncertainty_development.json',
    'combined': 'joint_retention_attachment_development.json',
    'conservative': 'joint_conservative_retention_development.json',
}


def validate_variant(name, value, baseline):
    bindings = {'retention':('observable-joint.v5-scale','cold-portfolio.v1'),
        'exploration':('observable-joint.v5-scale','component-beam.v2'),
        'attachment':('observable-joint.v6-attachment','component-beam.v1'),
        'combined':('observable-joint.v6-attachment','cold-portfolio.v1'),
        'conservative':('observable-joint.v6-attachment','cold-portfolio.v1')}
    require((value['objective_version'],value.get('joint_search_policy')) == bindings[name], 'Retention variant binding differs')
    changed = {k for k in set(value)|set(baseline) if value.get(k) != baseline.get(k)}
    allowed = {'objective_version','joint_search_policy'}
    if name in ('retention','combined','conservative'):
        allowed |= {'joint_portfolio_fraction','joint_portfolio_component_policy'}
        require(value.get('joint_portfolio_component_policy') == ('component-beam.v1' if name in ('retention','conservative') else 'component-beam.v2'), 'Portfolio component arm binding differs')
    if name in ('attachment','combined','conservative'):
        allowed |= {'attachment_uncertainty_cost','attachment_axis_weight','attachment_corroboration_discount','attachment_deletion_checks'}
    if name == 'conservative':
        require(value.get('attachment_uncertainty_cost') == 0.,'Conservative variant must disable the extra bridge cost')
    require(changed <= allowed,'Retention ablation changed budgets or unrelated policy')


def previous_quality(root, integration):
    request, state, receipt = (read_json(root/name) for name in ('request.json','progress.json','experiment_receipt.json'))
    require(state['phase'] == 'finished' and request['schema_version'] == 'stage4.v3-quality-request.v1', 'Previous quality run is incomplete')
    require(request['request_sha256'] == digest({k:v for k,v in request.items() if k != 'request_sha256'}) == receipt['request_sha256'], 'Previous quality request changed')
    require(Path(request['integration']).resolve() == integration.resolve() and 'combined' in request['variants'], 'Previous quality input/variant differs')
    require(request['integration_receipt_sha256'] == file_hash(integration/'integration_receipt.json'), 'Previous quality integration receipt differs')
    require(file_hash(root/'report/summary.json') == receipt['summary_sha256'], 'Previous quality report changed')
    verify_snapshot(request['snapshot'])
    return request,state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration',type=Path,default=WORKSPACE/'outputs/v3_development_full/integration')
    parser.add_argument('--previous-quality',type=Path,default=WORKSPACE/'outputs/v3_development_full/quality')
    parser.add_argument('--snapshot',type=Path,default=WORKSPACE/'controls/stage4_retention_v6_conservative_development')
    parser.add_argument('--case-id',action='append'); parser.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=['conservative'])
    parser.add_argument('--output',type=Path); parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--deep-diagnostics',action='store_true')
    args = parser.parse_args(argv)
    integration, previous, snapshot = (p.resolve() for p in (args.integration,args.previous_quality,args.snapshot))
    original = checked_integration(integration)
    prior, prior_state = previous_quality(previous,integration)
    cases = select_cases(original['import']['cases'],args.case_id)
    require(len(args.variants) == len(set(args.variants)), 'Duplicate retention variant')
    frozen = verify_snapshot(snapshot)
    configs = {name:snapshot/'configs'/VARIANTS[name] for name in args.variants}
    for case in cases:
        cid = case['case_id']
        require(cid in prior['case_ids'],'Previous comparison is missing a selected case')
        previous_result = read_result(previous/'cases'/cid/'combined/stage4')
        require(previous_result['implementation'] == prior['implementation'] and
            previous_result['input']['manifest_sha256'] == read_result(integration/'cases'/cid/'coarse/stage4')['input']['manifest_sha256'] and
            file_hash(previous/'cases'/cid/'combined/stage4/result.json') == prior_state['cases'][cid]['combined']['result_sha256'], 'Previous prediction binding changed')
        for name, path in configs.items():
            validate_variant(name,SolverConfig.from_dict(read_json(path)).to_dict(),previous_result['configuration'])
    if args.check:
        print('Retention preflight passed:',[c['case_id'] for c in cases],args.variants,'(no inference or output)')
        return 0
    require(args.output is not None,'--output is required')
    output = args.output.resolve()
    safe_output(output,[integration,previous,snapshot,WORKSPACE/'src',WORKSPACE/'scripts',WORKSPACE/'benchmark',WORKSPACE/'controls',Path(original['benchmark_manifest']).parent])
    request = {'schema_version':'stage4.v6-retention-request.v1','integration':str(integration),'previous_quality':str(previous),
        'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),'previous_receipt_sha256':file_hash(previous/'experiment_receipt.json'),
        'output':str(output),'case_ids':[c['case_id'] for c in cases],'variants':args.variants,'snapshot':str(snapshot),
        'implementation':frozen['implementation'],'configurations':{k:{'path':str(v),'sha256':file_hash(v)} for k,v in configs.items()},
        'controllers':{str(p):file_hash(p) for p in (Path(__file__),WORKSPACE/'benchmark/snapshot_diagnosis.py',WORKSPACE/'scripts/run_v3_retention_experiments.py',WORKSPACE/'scripts/diagnose_v3_retention.py',WORKSPACE/'scripts/run_source_snapshot.py')},
        'cold_primaries':True,'upstream_reused':True,'fresh_pipeline':False,'deep_diagnostics':args.deep_diagnostics}
    request['request_sha256'] = digest(request)
    require(not output.exists() or args.resume,'Existing retention run: use matching --resume or new output')
    output.mkdir(parents=True,exist_ok=True); lock = output/'run.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as stream:
        stream.write(str(os.getpid()))
    state = read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase':'predictions','cases':{}}
    try:
        if (output/'request.json').exists():
            require(read_json(output/'request.json') == request,'Retention run context changed')
        else:
            atomic_json(output/'request.json',request)
        state['phase'] = 'predictions'; atomic_json(output/'progress.json',state)
        for case in cases:
            cid = case['case_id']; entry = state['cases'].setdefault(cid,{})
            manifest = read_result(integration/'cases'/cid/'coarse/stage4')['input']['manifest_sha256']
            for variant, path in configs.items():
                target = output/'cases'/cid/variant/'stage4'
                if not (target/'manifest.json').exists():
                    print(cid,variant,': cold shared-budget consumer',flush=True)
                    command = [sys.executable,'-B',WORKSPACE/'scripts/run_source_snapshot.py',snapshot,'solve',
                        '--bundle',integration/'cases'/cid/'stage3','--config',path,'--output',target,'--resume']
                    attempt = execute(command,target.parent/'consumer.log',env_overrides=THREADS,echo_progress=True)
                    entry[variant] = attempt; atomic_json(output/'progress.json',state)
                    require(attempt['returncode'] == 0,'Retention consumer failed')
                result = read_result(target)
                require(result['implementation'] == frozen['implementation'] and result['input']['manifest_sha256'] == manifest and
                    result['configuration'] == SolverConfig.from_dict(read_json(path)).to_dict(),'Retention consumer binding differs')
                entry.setdefault(variant,{}).update(status='predicted',result_sha256=file_hash(target/'result.json'))
                atomic_json(output/'progress.json',state)
        state['phase'] = 'diagnostics'; atomic_json(output/'progress.json',state)
        state['diagnostics'] = execute([sys.executable,'-B',WORKSPACE/'scripts/diagnose_v3_retention.py','--run',output],
            output/'diagnostics.log',env_overrides=THREADS,echo_progress=True)
        state['phase'] = 'finished' if state['diagnostics']['returncode'] == 0 else 'diagnostics_failed'
        atomic_json(output/'progress.json',state)
        print('Retention report:',output/'report/index.html',flush=True)
        return int(state['phase'] != 'finished')
    except KeyboardInterrupt:
        state['phase'] = 'interrupted'; atomic_json(output/'progress.json',state); return 130
    except (ValueError,OSError,KeyError) as error:
        state.update(phase='failed',error=str(error)); atomic_json(output/'progress.json',state); raise
    finally:
        lock.unlink(missing_ok=True)


def diagnose(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True,type=Path)
    root = parser.parse_args(argv).run.resolve()
    request,state = read_json(root/'request.json'),read_json(root/'progress.json')
    require(state['phase'] in {'diagnostics','finished','diagnostics_failed'},'Retention serving barrier not reached')
    require(request['schema_version'] == 'stage4.v6-retention-request.v1' and request['request_sha256'] == digest({k:v for k,v in request.items() if k != 'request_sha256'}),'Retention request checksum differs')
    require(all(state['cases'].get(cid,{}).get(v,{}).get('status') == 'predicted' for cid in request['case_ids'] for v in request['variants']),'Retention serving barrier incomplete')
    for path,sha in request['controllers'].items():
        require(file_hash(Path(path)) == sha,'Retention diagnostic controller changed')
    integration,previous = Path(request['integration']),Path(request['previous_quality'])
    original = checked_integration(integration); prior,prior_state = previous_quality(previous,integration)
    require(file_hash(integration/'integration_receipt.json') == request['integration_receipt_sha256'] and file_hash(previous/'experiment_receipt.json') == request['previous_receipt_sha256'],'Retention source receipt changed')
    verify_snapshot(request['snapshot'])
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from .diagnostics import selected_support_roles
    from .report import document, esc, image, table, number
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
    from paper_reconstruction_stage4.joint_score import JointScorer
    from evaluation.pose_coverage import state_from_layout
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity() == request['implementation'],'Diagnostic must use the prediction source snapshot')
    cohort_root,cohort = load_cohort(original['benchmark_manifest'])
    report = root/'report'; report.mkdir(exist_ok=True); rows=[]; cells=[]; visuals=[]
    for cid in request['case_ids']:
        case = next(c for c in cohort['cases'] if c['case_id'] == cid)
        reference = read_json(cohort_root/case['path']/'evaluation_only/reference_geometry.json')
        controls = [read_result(integration/'cases'/cid/'coarse/stage4'),read_result(previous/'cases'/cid/'combined/stage4')]
        require(file_hash(previous/'cases'/cid/'combined/stage4/result.json') == prior_state['cases'][cid]['combined']['result_sha256'],'Previous sealed prediction changed')
        for variant in request['variants']:
            target = root/'cases'/cid/variant/'stage4'; result=read_result(target)
            require(file_hash(target/'result.json') == state['cases'][cid][variant]['result_sha256'],'Retention prediction changed')
            config=SolverConfig.from_dict(result['configuration'])
            require(result['implementation'] == request['implementation'] and config.to_dict() == SolverConfig.from_dict(read_json(request['configurations'][variant]['path'])).to_dict(),'Retention prediction configuration/implementation differs')
            require(all(r['input']['manifest_sha256'] == result['input']['manifest_sha256'] for r in controls),'Retention input comparison differs')
            ref={**reference,'input_manifest_sha256':result['input']['manifest_sha256']}
            qualities=[assembly_quality(r,ref,evaluate(r,ref)) for r in [*controls,result]]
            problem=load_problem(integration/'cases'/cid/'stage3',config)
            support=ActiveSupportIndex(problem,config); problem.structural=JointScorer(problem,config,support)
            rescored=[problem.structural.evaluate(state_from_layout(r['layouts'][0])) for r in [*controls,result]]
            row={'case_id':cid,'variant':variant,'quality_v4':qualities[0],'quality_v5_combined':qualities[1],'quality_experiment':qualities[2],
                'consumer_seconds':[r['resources']['elapsed_before_publication_seconds'] for r in [*controls,result]],
                'same_objective_scores':[s['score'] for s in rescored],'objective':config.objective_version,'score_terms':rescored[-1],
                'support_verification':verify_result_support(problem,result,config),'selected_support_roles':selected_support_roles(problem,result,ref),
                'search':result['search'],'termination':result['termination'],'proposals':result['search']['proposals'],
                'cold_primary':True,'visual_review':None,'evaluation_truth_opened':True}
            if request['deep_diagnostics']:
                from .joint_diagnostics import audit_candidates
                row['existing_candidate_audit']=audit_candidates(result,problem,config,ref,maximum=24,seconds=10.)
            rows.append(row); q=qualities[-1]; g=q['common_gauge']
            error=g['non_anchor_probe_errors_reference_px']['p95'] if g['whole_page_error_available'] else None
            cells.append((esc(cid),esc(variant),esc(q['component_sizes']),number(q['selected_neighbor_identity_precision']),number(error),number(row['consumer_seconds'][-1]),esc(row['termination'])))
            visuals.append('<h2>'+esc(cid+' / '+variant)+'</h2><div class="visuals">'+
                image(integration/'cases'/cid/'coarse/stage4/preview.png',report,'Frozen v4 primary')+
                image(previous/'cases'/cid/'combined/stage4/preview.png',report,'Previous combined primary')+
                image(target/'preview.png',report,'New cold primary')+'</div>')
    atomic_json(report/'summary.json',{'schema_version':'stage4.v6-retention-comparison.v1','request_sha256':request['request_sha256'],
        'cases':rows,'construction_requires_visual_review':True,'diagnostic_implementation':implementation_identity(),
        'timing_scope':'Consumer only; upstream evidence reused','default_promoted':False})
    content='<h1>Cold retention and attachment experiments</h1><p>Same imported evidence; fixed total proposal/time budgets. Each result starts cold. No reference seeds or upstream inference. Connectivity and neighbor identity do not establish construction success.</p>'
    content+=table(['Case','Variant','Components','True-neighbor fraction','Whole-page p95 reference px','Consumer seconds','Termination'],cells)+''.join(visuals)
    (report/'index.html').write_text(document('Cold retention experiments',content),encoding='utf-8')
    atomic_json(root/'experiment_receipt.json',{'schema_version':'stage4.v6-retention-receipt.v1','request_sha256':request['request_sha256'],
        'summary_sha256':file_hash(report/'summary.json'),'cold_primaries':True,'default_promoted':False,'status':'development_not_promoted'})
    return 0
