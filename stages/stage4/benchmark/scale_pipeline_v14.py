"""Fresh eight-case shared-scale pipeline; frozen V14 consumer, no reference reads."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

from .common import WORKSPACE, DESKTOP, atomic_json, digest, file_hash, inventory, load_cohort, read_json, verify_files
from .runner import completed_stage2, execute, producer_identity
from .snapshots import verify_snapshot

STAGE2 = DESKTOP / 'paper-reconstruction-stage2'
STAGE3 = DESKTOP / 'paper-reconstruction-stage3'
COHORT = STAGE3 / 'generated/clean_benchmarks/clean_shared_scale_v1'
RUNTIME = STAGE3 / 'generated/stage3_candidates/unit-scale-v5_6b0ac07074ff3ecf_0e97a1aa'
SNAPSHOT = WORKSPACE / 'controls/stage4_v14_native_repair_development_r1'
IMPLEMENTATION = 'd060278d6308c8035c3efea5dfc58cab7d1a77bc08883739978e48259ed972b1'
THREADS = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
CONTROLLERS = ['benchmark/scale_pipeline_v14.py', 'benchmark/scale_pipeline_diagnosis_v14.py',
    'benchmark/common.py', 'benchmark/runner.py', 'benchmark/snapshots.py', 'benchmark/qualified_v3.py',
    'benchmark/report.py', 'benchmark/diagnostics.py', 'benchmark/quality.py',
    'benchmark/native_repair_experiment_v14.py', 'evaluation/evaluate.py',
    'scripts/run_scale_pipeline_v14.py', 'scripts/scale_pipeline_worker_v14.py',
    'scripts/diagnose_scale_pipeline_v14.py', 'scripts/run_source_snapshot.py']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(value, field):
    require(value[field] == digest({k: v for k, v in value.items() if k != field}), 'Changed digest: ' + field)
    return value


def validate_cohort(cohort):
    cases = cohort['cases']
    require(Counter(c['pieces'] for c in cases) == {4: 2, 8: 2, 12: 2, 16: 2}, 'Require exactly two cases at each of 4/8/12/16 pieces')
    require(len({c['case_id'] for c in cases}) == 8, 'Duplicate case IDs')
    profile = cohort['capture_profile']
    require(profile.get('shared_native_pixel_scale') is True and profile.get('per_piece_resize') is False and
            profile.get('perspective') is False and profile.get('rotation_family') == 'quarter_turn' and
            profile.get('stage1_masks') == 'generator_exact_bypass', 'Need reviewed exact-mask shared-scale quarter-turn capture')
    return sorted(cases, key=lambda c: (c['pieces'], c['case_id']))


def commands_for(case, run, request):
    cid = case['case_id']; output = run / 'cases' / cid
    stage1 = Path(request['cohort_root']) / case['path'] / 'stage1'
    py2 = STAGE2 / '.venv/Scripts/python.exe'; py3 = STAGE3 / '.venv/Scripts/python.exe'
    worker = WORKSPACE / 'scripts/scale_pipeline_worker_v14.py'
    common = ['--request', run / 'request.json', '--case', cid]
    return [
        ('stage2', [py2, '-B', STAGE2 / 'scripts/run_stage2_inference.py', '--input', stage1,
                    '--output', output / 'stage2', '--release', STAGE2 / 'models/release.json',
                    '--cpu-threads', str(request['stage2_cpu_threads']), '--resume']),
        ('stage2_verify', [py2, '-B', STAGE2 / 'scripts/verify_stage2_output.py', '--input', output / 'stage2',
                           '--release', STAGE2 / 'models/release.json']),
        ('stage3_native', [py3, '-B', worker, 'infer', *common]),
        ('stage3_pool', [py3, '-B', worker, 'pool', *common]),
        ('stage4', [sys.executable, '-B', WORKSPACE / 'scripts/run_source_snapshot.py', request['snapshot'], 'solve',
                    '--bundle', output / 'stage3', '--config', request['config'], '--constraint-profile', request['profile'],
                    '--capture-profile', stage1 / 'capture_profile.json', '--output', output / 'stage4', '--resume'])]


def create_request(run, device, cpu_threads):
    root, cohort = load_cohort(COHORT, serving_only=True)
    cases = validate_cohort(cohort)
    frozen = verify_snapshot(SNAPSHOT)
    require(frozen['implementation']['sha256'] == IMPLEMENTATION, 'Frozen V14 implementation differs')
    candidate = read_json(RUNTIME / 'candidate.json')
    require(candidate['candidate_implementation_sha256'] == '6b0ac07074ff3ecf340aa7c30deae6760d8a8cd4807e3dd9b71348d72e86e66a', 'Unit-scale runtime differs')
    config = SNAPSHOT / 'configs/joint_v14_native_repair_development.json'
    profile = SNAPSHOT / 'configs/shared_scale_complete_rectangle_v1.json'
    upstream_scripts = [p for p in sorted((STAGE3 / 'scripts').glob('*.py'))]
    model_files = [p for p in sorted((STAGE3 / 'models').rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    runtime_files = [p for p in sorted(RUNTIME.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    sdk_files = [p for p in sorted((STAGE3 / 'contracts/stage3_to_stage4/v2/sdk/src').rglob('*'))
                 if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts]
    request = {'schema_version': 'stage4.fresh-scale-pipeline.v1', 'output': str(run),
        'cohort_root': str(root), 'cohort_manifest_sha256': file_hash(root / 'benchmark.json'),
        'cohort': cohort, 'cases': cases, 'snapshot': str(SNAPSHOT), 'implementation': frozen['implementation'],
        'config': str(config), 'profile': str(profile), 'configuration': read_json(config),
        'stage3_root': str(STAGE3), 'stage3_runtime': str(RUNTIME), 'stage3_device': device,
        'stage2_cpu_threads': cpu_threads, 'stage2_producer': producer_identity(STAGE2),
        'bound_files': {str(p): file_hash(p) for p in [config, profile, *upstream_scripts, *model_files, *runtime_files, *sdk_files]},
        'controllers': {str(WORKSPACE / p): file_hash(WORKSPACE / p) for p in CONTROLLERS},
        'fresh_pipeline': True, 'stage1_inference': False, 'stage1_source': 'generator_exact_bypass',
        'stage2_inference': True, 'stage3_fresh_learned_ranking_and_unit_scale_geometry': True,
        'saved_rankings_or_pair_caches_imported': False, 'frozen_v14_consumer': True,
        'evaluation_truth_opened_during_predictions': False,
        'scope': 'Exposed synthetic development cohort, complete rectangular page, unknown format/DPI; quarter turns and shared pixel scale.'}
    request['request_sha256'] = digest(request)
    return request


def authenticate(request):
    checked(request, 'request_sha256')
    require(verify_snapshot(request['snapshot'])['implementation'] == request['implementation'], 'Frozen consumer changed')
    for path, sha in {**request['controllers'], **request['bound_files']}.items():
        require(file_hash(path) == sha, 'Bound file changed: ' + path)
    producer = request['stage2_producer']
    for path, sha in producer['files'].items():
        require(file_hash(Path(producer['root']) / path) == sha, 'Stage 2 source/model changed: ' + path)
    root = Path(request['cohort_root'])
    require(file_hash(root / 'benchmark.json') == request['cohort_manifest_sha256'], 'Cohort manifest changed')
    for case in request['cases']:
        verify_files(root / case['path'], [f for f in case['files'] if f['path'].startswith('stage1/')])


def verify_predictions(run, request, progress):
    """Seal exact downstream bundles/work caches and independent ready commitments."""
    authenticate(request)
    from paper_reconstruction_contracts import load_bundle
    from paper_reconstruction_stage4.outputs import read_result
    from paper_reconstruction_stage4.assessment import verify_package
    records = []; files = {}
    for case in request['cases']:
        cid = case['case_id']; output = run / 'cases' / cid; entry = progress['cases'][cid]
        require(entry['status'] in ('predicted', 'failed'), 'Unfinished prediction attempt: ' + cid)
        if entry['status'] == 'predicted':
            bundle = load_bundle(output / 'stage3', full=True); result = read_result(output / 'stage4')
            require(result['implementation'] == request['implementation'] and result['configuration'] == request['configuration'] and
                    result['input']['manifest_sha256'] == bundle.manifest['manifest_sha256'] and result['evaluation_truth_opened'] is False,
                    'Stage 4 prediction binding differs: ' + cid)
            assessment = verify_package(output / 'stage4.assessment')
            require(assessment['binding']['result_sha256'] == file_hash(output / 'stage4/result.json') and
                    assessment['binding']['assessment_implementation_sha256'] == request['implementation']['sha256'], 'Assessment differs')
        # Include attempted stages and pair caches even if a later stage failed.
        for item in inventory(output):
            if item['path'].endswith('.log') or item['path'].endswith('.attempt.json'):
                continue
            files[str(output / item['path'])] = item['sha256']
        records.append({'case_id': cid, 'pieces': case['pieces'], 'status': entry['status'], 'failed_stage': entry.get('failed_stage')})
    return {'cases': records, 'files': files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage3-device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--stage2-cpu-threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv); run = args.output.resolve()
    require(run.is_relative_to((WORKSPACE / 'outputs').resolve()) and run != (WORKSPACE / 'outputs').resolve(), 'Output must be a new directory inside Stage 4 outputs')
    require(args.stage2_cpu_threads > 0, 'Stage 2 CPU threads must be positive')
    if (run / 'request.json').exists():
        require(args.resume, 'Existing run requires --resume')
        request = read_json(run / 'request.json')
        require(request['output'] == str(run) and request['stage3_device'] == args.stage3_device and
                request['stage2_cpu_threads'] == args.stage2_cpu_threads, 'Resume settings differ')
        authenticate(request)
    else:
        require(not run.exists() or not any(run.iterdir()), 'Unrelated nonempty output')
        request = create_request(run, args.stage3_device, args.stage2_cpu_threads)
    run.mkdir(parents=True, exist_ok=True)
    lock = run / 'run.lock'
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        atomic_json(run / 'request.json', request)
        progress = read_json(run / 'progress.json') if (run / 'progress.json').exists() else {'phase': 'preflight', 'cases': {}}
        if not (run / 'preflight.json').exists():
            attempt = execute([STAGE3 / '.venv/Scripts/python.exe', '-B', WORKSPACE / 'scripts/scale_pipeline_worker_v14.py',
                               'check', '--request', run / 'request.json'], run / 'preflight.log', env_overrides=THREADS, echo_progress=True)
            require(attempt['returncode'] == 0, 'Stage 3 runtime/model/device preflight failed; inspect preflight.log')
        if args.preflight_only:
            print('Fresh eight-case pipeline preflight passed:', run / 'preflight.json', flush=True); return 0
        if not (run / 'predictions_sealed.json').exists():
            progress['phase'] = 'predictions'; atomic_json(run / 'progress.json', progress)
            for index, case in enumerate(request['cases'], 1):
                cid = case['case_id']; output = run / 'cases' / cid
                entry = progress['cases'].setdefault(cid, {'status': 'pending', 'attempts': []})
                if entry['status'] == 'predicted':
                    require(file_hash(output / 'stage4/result.json') == entry['result_sha256'], 'Committed prediction changed')
                    continue
                entry.pop('failed_stage', None); entry['status'] = 'working'
                for stage, command in commands_for(case, run, request):
                    entry['active_stage'] = stage; atomic_json(run / 'progress.json', progress)
                    print(f'[{index}/8] {cid} ({case["pieces"]} pieces): {stage}', flush=True)
                    attempt = None
                    if stage == 'stage2':
                        attempt = completed_stage2(output, Path(request['cohort_root']) / case['path'] / 'stage1')
                    elif stage in ('stage3_native', 'stage3_pool', 'stage4'):
                        commit = output / (stage + '_commit.json')
                        if commit.exists():
                            value = read_json(commit)
                            require(value['request_sha256'] == request['request_sha256'], 'Foreign stage commitment')
                            verify_files(output, value['files'])
                            attempt = {'status': 'reused_complete', 'returncode': 0, 'stage': stage}
                    if attempt is None:
                        attempt = execute(command, output / (stage + '.log'), env_overrides=THREADS, echo_progress=True)
                    attempt['stage'] = stage; entry['attempts'].append(attempt)
                    if attempt['returncode'] != 0:
                        entry.update(status='failed', failed_stage=stage)
                        atomic_json(run / 'progress.json', progress); break
                    if stage == 'stage2':
                        atomic_json(output / 'stage2_commit.json', {'manifest_sha256': file_hash(output / 'stage2/edge_hypotheses_manifest.json')})
                    if stage in ('stage3_native', 'stage3_pool', 'stage4') and attempt['status'] != 'reused_complete':
                        folders = {'stage3_native': ['stage3_native', 'stage3_native.work'], 'stage3_pool': ['stage3'], 'stage4': ['stage4', 'stage4.assessment']}[stage]
                        receipts = [{**f, 'path': folder + '/' + f['path']} for folder in folders for f in inventory(output / folder)]
                        atomic_json(output / (stage + '_commit.json'), {'request_sha256': request['request_sha256'], 'files': receipts})
                    atomic_json(run / 'progress.json', progress)
                else:
                    entry.update(status='predicted', result_sha256=file_hash(output / 'stage4/result.json'))
                    atomic_json(run / 'progress.json', progress)
            sealed = verify_predictions(run, request, progress)
            barrier = {'schema_version': 'stage4.fresh-scale-predictions-sealed.v1', 'request_sha256': request['request_sha256'],
                       **sealed, 'references_opened': False}
            barrier['barrier_sha256'] = digest(barrier); atomic_json(run / 'predictions_sealed.json', barrier)
        progress['phase'] = 'diagnostics'; atomic_json(run / 'progress.json', progress)
        attempt = execute([sys.executable, '-B', WORKSPACE / 'scripts/diagnose_scale_pipeline_v14.py', '--run', run],
                          run / 'diagnostics.log', env_overrides=THREADS, echo_progress=True)
        require(attempt['returncode'] == 0, 'Post-seal diagnostics failed; predictions preserved; inspect diagnostics.log')
        progress.update(phase='finished', diagnostics=attempt); atomic_json(run / 'progress.json', progress)
        print('Fresh eight-case pipeline report:', run / 'report/index.html', flush=True)
        return 0
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
