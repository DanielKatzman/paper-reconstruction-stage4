"""Separate upstream preparation and fixed-solver capacity experiment."""
import argparse
import os
from pathlib import Path
import sys
from .common import WORKSPACE, atomic_json, digest, file_hash, inventory, read_json, verify_files, load_cohort
from .scale_pipeline_v14 import STAGE2, STAGE3, RUNTIME, THREADS, checked, require, commands_for as pipeline_commands
from .runner import execute, completed_stage2, producer_identity
from .snapshots import verify_snapshot
from .scale_cohort_v20 import COHORT, COUNTS, prepare, validate

BASELINE = WORKSPACE/'outputs/v19_baseline_confirmation'
SNAPSHOT = WORKSPACE/'controls/stage4_v19_geometry_expansion_development_r2'
IMPLEMENTATION = 'f8fd285b589e64ab926d82781d995b502ed79ad47134a7a30e33be64e7460d68'
NEW_CONTROLLERS = ['benchmark/scale_extension_v20.py', 'benchmark/scale_cohort_v20.py',
    'benchmark/scale_extension_diagnosis_v20.py', 'benchmark/constructibility_v20.py',
    'scripts/prepare_scale_extension_v20.py', 'scripts/run_scale_extension_v20.py',
    'scripts/diagnose_scale_extension_v20.py']
DEPENDENCIES = ['benchmark/common.py', 'benchmark/runner.py', 'benchmark/snapshots.py',
    'benchmark/scale_pipeline_v14.py', 'scripts/scale_pipeline_worker_v14.py', 'scripts/run_source_snapshot.py',
    'benchmark/diagnostics.py', 'benchmark/quality.py', 'benchmark/report.py', 'benchmark/scale_pipeline_diagnosis_v14.py',
    'benchmark/recovery_diagnosis_v19_r2.py', 'benchmark/recovery_experiment_v19_r2.py',
    'benchmark/recovery_gallery_v18_r2.py', 'benchmark/recovery_report_v18_r2_fix.py',
    'benchmark/native_repair_experiment_v14.py', 'evaluation/evaluate.py']


def controllers():
    return {str(WORKSPACE/p): file_hash(WORKSPACE/p) for p in [*NEW_CONTROLLERS, *DEPENDENCIES]}


def output_path(path):
    path = Path(path).resolve(); parent = WORKSPACE/'outputs'
    require(path.is_relative_to(parent) and path != parent, 'Output must stay inside Stage 4 outputs')
    return path


def baseline_binding():
    """Authenticate completed controls without parsing reference diagnoses."""
    from .recovery_experiment_v19_r2 import authenticate
    request = checked(read_json(BASELINE/'request.json'), 'request_sha256'); authenticate(request)
    barrier = checked(read_json(BASELINE/'predictions_sealed.json'), 'barrier_sha256')
    require(barrier['request_sha256'] == request['request_sha256'] and barrier['references_opened'] is False,
        'Baseline is not sealed')
    require(request['implementation']['sha256'] == IMPLEMENTATION and len(request['cases']) == 8 and
        read_json(BASELINE/'progress.json')['phase'] == 'finished', 'Use completed V19 baseline confirmation')
    for path, sha in barrier['files'].items():
        require(file_hash(path) == sha, 'Baseline prediction changed')
    receipt = read_json(BASELINE/'experiment_receipt.json'); verify_files(BASELINE, receipt['files'])
    return {'root': str(BASELINE), 'request_sha256': request['request_sha256'], 'barrier_sha256': barrier['barrier_sha256'],
        'report_receipt_sha256': file_hash(BASELINE/'experiment_receipt.json')}


def create_upstream(output, device, threads):
    root, cohort = prepare(); validate(cohort)
    frozen = verify_snapshot(SNAPSHOT)
    require(frozen['implementation']['sha256'] == IMPLEMENTATION, 'Fixed solver changed')
    runtime = read_json(RUNTIME/'candidate.json')
    require(runtime['candidate_implementation_sha256'] == '6b0ac07074ff3ecf340aa7c30deae6760d8a8cd4807e3dd9b71348d72e86e66a',
        'Use the same Stage 3 unit-scale runtime as the passing baseline')
    paths = list((STAGE3/'scripts').glob('*.py'))
    paths += [p for folder in (STAGE3/'models', RUNTIME, STAGE3/'contracts/stage3_to_stage4/v2/sdk/src')
        for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    config = SNAPSHOT/'configs/joint_v19_geometry_expansion_development.json'
    profile = SNAPSHOT/'configs/shared_scale_complete_rectangle_v1.json'
    value = {'schema_version': 'stage4.scale-extension-upstream.v1', 'output': str(output),
        'cohort_root': str(root), 'cohort_manifest_sha256': file_hash(root/'benchmark.json'),
        'cohort': cohort, 'cases': cohort['cases'], 'stage3_root': str(STAGE3), 'stage3_runtime': str(RUNTIME),
        'stage3_device': device, 'stage2_cpu_threads': threads, 'stage2_producer': producer_identity(STAGE2),
        'snapshot': str(SNAPSHOT), 'implementation': frozen['implementation'], 'configuration': read_json(config),
        'config': str(config), 'profile': str(profile), 'baseline': baseline_binding(),
        'bound_files': {str(p): file_hash(p) for p in [config, profile, *paths]}, 'controllers': controllers(),
        'stage1_inference': False, 'stage4_inference': False, 'fresh_stage2_stage3': True,
        'evaluation_truth_opened_during_predictions': False}
    value['request_sha256'] = digest(value)
    return value


def authenticate(request):
    checked(request, 'request_sha256')
    require(request['schema_version'] == 'stage4.scale-extension-upstream.v1', 'Foreign upstream request')
    validate(request['cohort'])
    require(verify_snapshot(request['snapshot'])['implementation'] == request['implementation'], 'Solver snapshot changed')
    require(request['implementation']['sha256'] == IMPLEMENTATION, 'Use the passing V19 r2 solver unchanged')
    for path, sha in {**request['controllers'], **request['bound_files']}.items():
        require(file_hash(path) == sha, 'Bound file changed: '+path)
    require(request['baseline'] == baseline_binding(), 'Baseline changed')
    require(file_hash(Path(request['cohort_root'])/'benchmark.json') == request['cohort_manifest_sha256'], 'Cohort changed')
    for case in request['cases']:
        verify_files(Path(request['cohort_root'])/case['path'], [r for r in case['files'] if r['path'].startswith('stage1/')])
    for path, sha in request['stage2_producer']['files'].items():
        require(file_hash(STAGE2/path) == sha, 'Stage 2 source/model changed')


def upstream_commands(case, output, request):
    return pipeline_commands(case, output, request)[:4]


def verify_upstream(output, request, progress):
    from paper_reconstruction_contracts import load_bundle
    authenticate(request); files = {}; rows = []
    for case in request['cases']:
        cid = case['case_id']; folder = output/'cases'/cid; entry = progress['cases'][cid]
        require(entry['status'] == 'ready', 'Upstream case is unfinished or failed: '+cid)
        require(file_hash(folder/'stage2/edge_hypotheses_manifest.json') ==
            read_json(folder/'stage2_commit.json')['manifest_sha256'], 'Stage 2 manifest changed')
        bundle = load_bundle(folder/'stage3', full=True)
        require(len(bundle.manifest['pieces']) == case['pieces'], 'Upstream physical inventory differs: '+cid)
        for item in inventory(folder):
            if not item['path'].endswith(('.log', '.attempt.json')):
                files[str(folder/item['path'])] = item['sha256']
        rows.append({'case_id': cid, 'pieces': case['pieces'], 'bundle_sha256': bundle.manifest['manifest_sha256'],
            'attempts_sha256': digest(entry['attempts'])})
    return {'cases': rows, 'files': files}


def upstream_binding(output):
    output = output_path(output); request = read_json(output/'request.json')
    barrier = checked(read_json(output/'upstream_sealed.json'), 'barrier_sha256')
    require(barrier['request_sha256'] == request['request_sha256'] and barrier['references_opened'] is False,
        'Prepare and seal all six upstream cases first')
    verified = verify_upstream(output, request, read_json(output/'progress.json'))
    require(all(barrier[k] == v for k, v in verified.items()), 'Upstream seal differs')
    return request, barrier


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare exact generator cuts and run Stage 2/3 only at 20/24/30 pieces.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage3-device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--stage2-cpu-threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv); output = output_path(args.output)
    require(args.stage2_cpu_threads > 0, 'Stage 2 threads must be positive')
    if (output/'request.json').exists():
        require(args.resume, 'Use --resume for unchanged upstream preparation')
        request = read_json(output/'request.json'); authenticate(request)
        require(request['stage3_device'] == args.stage3_device and request['stage2_cpu_threads'] == args.stage2_cpu_threads,
            'Resume runtime settings differ')
    else:
        require(not output.exists() or not any(output.iterdir()), 'Unrelated nonempty output')
        request = create_upstream(output, args.stage3_device, args.stage2_cpu_threads)
    output.mkdir(parents=True, exist_ok=True); lock = output/'run.lock'
    with os.fdopen(os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY), 'w') as f:
        f.write(str(os.getpid()))
    try:
        atomic_json(output/'request.json', request)
        progress = read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase': 'preflight', 'cases': {}}
        if (output/'upstream_sealed.json').exists():
            upstream_binding(output); print('Sealed Stage 2/3 preparation verified:', output, flush=True); return 0
        if not (output/'preflight.json').exists():
            attempt = execute([STAGE3/'.venv/Scripts/python.exe', '-B', WORKSPACE/'scripts/scale_pipeline_worker_v14.py',
                'check', '--request', output/'request.json'], output/'preflight.log', env_overrides=THREADS, echo_progress=True)
            require(attempt['returncode'] == 0, 'Stage 3 model/device check failed; inspect preflight.log')
        require(read_json(output/'preflight.json')['request_sha256'] == request['request_sha256'], 'Foreign preflight')
        if args.preflight_only:
            progress['phase'] = 'preflight_complete'; atomic_json(output/'progress.json', progress)
            print('Extension preflight ready; no Stage 2/3 inference ran:', output/'request.json', flush=True); return 0
        progress['phase'] = 'upstream_predictions'; atomic_json(output/'progress.json', progress)
        for index, case in enumerate(request['cases'], 1):
            cid = case['case_id']; folder = output/'cases'/cid
            entry = progress['cases'].setdefault(cid, {'status': 'pending', 'attempts': []})
            if entry['status'] == 'ready':
                continue
            entry.update(status='working'); entry.pop('failed_stage', None)
            for stage, command in upstream_commands(case, output, request):
                entry['active_stage'] = stage; atomic_json(output/'progress.json', progress)
                print(f'[{index}/6] {cid} ({case["pieces"]} pieces): {stage}', flush=True)
                commit = folder/(stage+'_commit.json'); attempt = None
                if stage == 'stage2':
                    attempt = completed_stage2(folder, Path(request['cohort_root'])/case['path']/'stage1')
                elif stage in ('stage3_native', 'stage3_pool') and commit.exists():
                    value = read_json(commit); require(value['request_sha256'] == request['request_sha256'], 'Foreign upstream commit')
                    verify_files(folder, value['files']); attempt = {'returncode': 0, 'status': 'reused_complete'}
                if attempt is None:
                    try:
                        attempt = execute(command, folder/(stage+'.log'), env_overrides=THREADS, echo_progress=True)
                    except KeyboardInterrupt:
                        attempt_path = folder/(stage+'.attempt.json')
                        if attempt_path.exists():
                            entry['attempts'].append({**read_json(attempt_path), 'stage': stage})
                        entry['status'] = 'interrupted'; atomic_json(output/'progress.json', progress); raise
                entry['attempts'].append({**attempt, 'stage': stage})
                if attempt['returncode'] != 0:
                    entry.update(status='failed', failed_stage=stage); break
                if stage == 'stage2':
                    atomic_json(commit, {'manifest_sha256': file_hash(folder/'stage2/edge_hypotheses_manifest.json')})
                if stage in ('stage3_native', 'stage3_pool') and attempt['status'] != 'reused_complete':
                    folders = ['stage3_native', 'stage3_native.work'] if stage == 'stage3_native' else ['stage3']
                    files = [{**r, 'path': name+'/'+r['path']} for name in folders for r in inventory(folder/name)]
                    atomic_json(commit, {'request_sha256': request['request_sha256'], 'files': files})
                atomic_json(output/'progress.json', progress)
            else:
                entry['status'] = 'ready'
            atomic_json(output/'progress.json', progress)
        require(all(e['status'] == 'ready' for e in progress['cases'].values()), 'Upstream failures remain; inspect progress and per-stage logs before Stage 4')
        barrier = {'schema_version': 'stage4.scale-extension-upstream-seal.v1', 'request_sha256': request['request_sha256'],
            **verify_upstream(output, request, progress), 'references_opened': False}
        barrier['barrier_sha256'] = digest(barrier); atomic_json(output/'upstream_sealed.json', barrier)
        progress['phase'] = 'upstream_ready'; atomic_json(output/'progress.json', progress)
        print('Six Stage 2/3 bundles sealed; ready for Stage 4:', output, flush=True); return 0
    finally:
        lock.unlink()


def create_stage4(output, upstream):
    request, barrier = upstream_binding(upstream); config = request['config']; profile = request['profile']
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.constraints import bind_profile
    cases = []
    for case in request['cases']:
        inputs = Path(request['cohort_root'])/case['path']; bundle = upstream/'cases'/case['case_id']/'stage3'
        problem = load_problem(bundle, SolverConfig.from_dict(request['configuration']))
        require(len(problem.pieces) == case['pieces'], 'Stage 4 physical inventory differs: '+case['case_id'])
        capture = inputs/'stage1/capture_profile.json'; bind_profile(problem, SolverConfig.from_dict(request['configuration']), profile, capture)
        cases.append({**case, 'bundle': str(bundle), 'bundle_sha256': problem.manifest['manifest_sha256'],
            'stage1': str(inputs/'stage1'), 'capture_profile': str(capture), 'capture_profile_sha256': file_hash(capture),
            'evaluation_reference': str(inputs/'evaluation_only/reference_geometry.json'),
            'evaluation_reference_sha256': next(r['sha256'] for r in case['files'] if r['path'] == 'evaluation_only/reference_geometry.json'),
            'constraint_binding': problem.constraint_profile})
    value = {'schema_version': 'stage4.scale-extension-predictions.v1', 'output': str(output), 'upstream': str(upstream),
        'upstream_request_sha256': request['request_sha256'], 'upstream_barrier_sha256': barrier['barrier_sha256'],
        'snapshot': request['snapshot'], 'implementation': request['implementation'], 'config': config, 'profile': profile,
        'configuration': request['configuration'], 'controllers': controllers(), 'cases': cases, 'baseline': request['baseline'],
        'evaluation_truth_opened_during_predictions': False, 'saved_layout_seeds': False,
        'constructibility_probe_seconds_per_tolerance': 40., 'constructibility_probe_tolerances_px': [8., 16.]}
    value['request_sha256'] = digest(value); return value


def authenticate_stage4(request):
    checked(request, 'request_sha256')
    for path, sha in request['controllers'].items():
        require(file_hash(path) == sha, 'Stage 4 controller changed: '+path)
    upstream, barrier = upstream_binding(Path(request['upstream']))
    require(upstream['request_sha256'] == request['upstream_request_sha256'] and
        barrier['barrier_sha256'] == request['upstream_barrier_sha256'], 'Upstream changed')
    require(request['configuration'] == upstream['configuration'] and request['implementation'] == upstream['implementation'],
        'Consumer configuration/source changed')


def stage4_command(output, request, case):
    return [sys.executable, '-B', WORKSPACE/'scripts/run_source_snapshot.py', request['snapshot'], 'solve',
        '--bundle', case['bundle'], '--config', request['config'], '--constraint-profile', request['profile'],
        '--capture-profile', case['capture_profile'], '--output', output/'cases'/case['case_id']/'stage4', '--resume']


def verify_stage4(output, request, progress):
    from paper_reconstruction_stage4.outputs import read_result
    from paper_reconstruction_stage4.assessment import verify_package
    authenticate_stage4(request); files = {}; rows = []
    for case in request['cases']:
        cid = case['case_id']; folder = output/'cases'/cid; entry = progress['cases'][cid]
        require(entry['status'] in ('predicted', 'failed'), 'Stage 4 case still active')
        if entry['status'] == 'predicted':
            value = read_result(folder/'stage4'); a = verify_package(folder/'stage4.assessment')
            require(value['implementation'] == request['implementation'] and value['configuration'] == request['configuration'] and
                value['input']['manifest_sha256'] == case['bundle_sha256'] and value['evaluation_truth_opened'] is False and
                value['search']['constraint_profile'] == case['constraint_binding'], 'Stage 4 result differs')
            require(a['binding']['result_sha256'] == file_hash(folder/'stage4/result.json'), 'Assessment binding differs')
        files.update({str(folder/r['path']): r['sha256'] for r in inventory(folder)})
        rows.append({'case_id': cid, 'pieces': case['pieces'], 'status': entry['status']})
    return {'cases': rows, 'files': files}


def run_main(argv=None):
    parser = argparse.ArgumentParser(description='Run fixed V19 r2 on six new larger puzzles and report all 14 cases.')
    parser.add_argument('--upstream', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv); output = output_path(args.output); upstream = output_path(args.upstream)
    require(output != upstream and not output.is_relative_to(upstream) and not upstream.is_relative_to(output), 'Keep upstream and Stage 4 outputs separate')
    if (output/'request.json').exists():
        require(args.resume, 'Use --resume for the unchanged experiment'); request = read_json(output/'request.json')
        require(request['upstream'] == str(upstream), 'Different upstream on resume'); authenticate_stage4(request)
    else:
        require(not output.exists() or not any(output.iterdir()), 'Unrelated nonempty output')
        request = create_stage4(output, upstream)
    output.mkdir(parents=True, exist_ok=True); lock = output/'run.lock'
    with os.fdopen(os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY), 'w') as f:
        f.write(str(os.getpid()))
    try:
        atomic_json(output/'request.json', request)
        progress = read_json(output/'progress.json') if (output/'progress.json').exists() else {'phase': 'preflight', 'cases': {}}
        if args.preflight_only:
            progress['phase'] = 'preflight_complete'; atomic_json(output/'progress.json', progress)
            print('Stage 4 preflight complete; six new cold solves pending:', output, flush=True); return 0
        if not (output/'predictions_sealed.json').exists():
            progress['phase'] = 'predictions'; atomic_json(output/'progress.json', progress)
            from .recovery_experiment_v19_r2 import run_attempt
            for i, case in enumerate(request['cases'], 1):
                cid = case['case_id']; folder = output/'cases'/cid
                entry = progress['cases'].setdefault(cid, {'status': 'pending', 'attempts': []})
                if entry['status'] == 'predicted':
                    require(file_hash(folder/'stage4/result.json') == entry['result_sha256'], 'Committed result changed'); continue
                entry['status'] = 'working'; atomic_json(output/'progress.json', progress)
                print(f'[{i}/6] {cid} ({case["pieces"]} pieces): fixed V19 r2 Stage 4', flush=True)
                attempt = run_attempt(stage4_command(output, request, case), folder, entry,
                    lambda: atomic_json(output/'progress.json', progress))
                entry['status'] = 'predicted' if attempt['returncode'] == 0 else 'failed'
                if entry['status'] == 'predicted':
                    entry['result_sha256'] = file_hash(folder/'stage4/result.json')
                atomic_json(output/'progress.json', progress)
            barrier = {'schema_version': 'stage4.scale-extension-prediction-seal.v1', 'request_sha256': request['request_sha256'],
                **verify_stage4(output, request, progress), 'references_opened': False}
            barrier['barrier_sha256'] = digest(barrier); atomic_json(output/'predictions_sealed.json', barrier)
        progress['phase'] = 'diagnostics'; atomic_json(output/'progress.json', progress)
        attempt = execute([sys.executable, '-B', WORKSPACE/'scripts/diagnose_scale_extension_v20.py', '--run', output],
            output/'diagnostics.log', env_overrides=THREADS, echo_progress=True)
        progress['diagnostics'] = attempt; atomic_json(output/'progress.json', progress)
        require(attempt['returncode'] == 0, 'Diagnostics failed; predictions preserved; inspect diagnostics.log')
        progress['phase'] = 'finished'; atomic_json(output/'progress.json', progress)
        print('14-case capacity report:', output/'report/index.html', flush=True); return 0
    finally:
        lock.unlink()
