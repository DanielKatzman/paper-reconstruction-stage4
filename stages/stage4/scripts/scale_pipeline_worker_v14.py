"""Stage 3 interpreter adapter. Reads serving inputs only; all writes stay in Stage 4."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'infer', 'pool'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args(argv); request = read(args.request)
    if request['request_sha256'] != digest({k: v for k, v in request.items() if k != 'request_sha256'}):
        raise ValueError('Request checksum differs')
    for path, expected in {**request['controllers'], **request['bound_files']}.items():
        if sha(path) != expected:
            raise ValueError('Bound file changed: ' + path)
    root = Path(request['stage3_root']); runtime = Path(request['stage3_runtime'])
    sys.path.insert(0, str(root / 'scripts'))
    from stage3_unit_runtime import validate_candidate
    candidate = validate_candidate(runtime)
    sys.path.insert(0, str(root / 'contracts/stage3_to_stage4/v2/sdk/src'))
    sys.path.insert(0, str(runtime / 'src'))
    from paper_reconstruction_stage3.alpha_package import load_package
    from paper_reconstruction_stage3.contracts import atomic_json
    from paper_reconstruction_contracts import load_bundle
    _, models = load_package(root / 'models', full=True)
    output = Path(request['output'])
    workspace = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(workspace / 'outputs'):
        raise ValueError('Worker writes must stay inside Stage 4 outputs')
    if args.mode == 'check':
        import torch
        device = request['stage3_device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        atomic_json(output / 'preflight.json', {'request_sha256': request['request_sha256'],
            'runtime_implementation_sha256': candidate['candidate_implementation_sha256'],
            'model_manifest_sha256': models['manifest_sha256'], 'device': device,
            'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
            'cases': len(request['cases']), 'stage1_bypassed': True, 'references_opened': False})
        print('Fresh Stage 3 runtime/model/device verified', flush=True); return 0
    case = next(c for c in request['cases'] if c['case_id'] == args.case)
    case_root = output / 'cases' / args.case
    stage1 = Path(request['cohort_root']) / case['path'] / 'stage1'
    if args.mode == 'infer':
        from paper_reconstruction_stage3.alpha_inference import infer
        from stage3_profile_tools import Measurements, instrument, process_memory
        metrics = Measurements(); metrics.started = time.perf_counter()
        # High-level timers only: avoid instrumentation of millions of primitive calls.
        targets = {
            'alpha_inference': [('production_inputs', 'inputs.production_inputs'), ('retrieve_parents', 'ranking.retrieve_parents'),
                ('fuse_parents', 'ranking.fuse_parents'), ('FrozenRankers.__init__', 'ranking.model_setup'), ('FrozenRankers.score', 'ranking.score')],
            'pose_geometry': [('fit_parent', 'poses.fit_parent')],
            'pose_inference_geometry': [('generate', 'poses.generate')],
            'alpha_bundle': [('export_bundle', 'export.bundle')],
            'contracts': [('atomic_json', 'io.atomic_json')]}
        def on_write(path, value):
            if path.name.startswith('pair_') and 'pieces' in value:
                metrics.pairs.append({'pieces': value['pieces'], 'elapsed_since_start_seconds': time.perf_counter() - metrics.started})
        with instrument(metrics, on_write, targets=targets):
            result = infer(root / 'models', stage1, case_root / 'stage2', case_root / 'stage3_native',
                           device=request['stage3_device'], puzzle_id=args.case, progress=lambda msg: print(msg, flush=True))
        atomic_json(case_root / 'stage3_profile.json', {'request_sha256': request['request_sha256'], 'result': result,
            'elapsed_seconds': time.perf_counter() - metrics.started, 'functions': metrics.functions(), 'pairs': metrics.pairs,
            'peak_memory': process_memory(), 'timer_scope': 'High-level non-mutating process instrumentation; inclusive times overlap',
            'fresh_ranking_requested': True, 'imported_external_rankings_or_pair_caches': False})
    else:
        from stage3_pool_replay_tools import publish_replay, POLICY
        source = load_bundle(case_root / 'stage3_native', full=True)
        target = case_root / 'stage3'
        implementation = {'experiment_request_sha256': request['request_sha256'],
                          'pool_tool_sha256': sha(root / 'scripts/stage3_pool_replay_tools.py')}
        if (target / 'bundle.json').exists():
            existing = load_bundle(target, full=True); provenance = existing.manifest['provenance']
            if provenance['source_manifest_sha256'] != source.manifest['manifest_sha256'] or provenance['implementation'] != implementation or existing.manifest['configuration']['pool_replay'] != POLICY:
                raise ValueError('Existing pool belongs to different inputs/policy')
            if not (case_root / 'stage3_pool_audit.json').exists():
                atomic_json(case_root / 'stage3_pool_audit.json', {'adopted_after_interruption': True, 'manifest_sha256': existing.manifest['manifest_sha256']})
        else:
            audit = publish_replay(source.root, target, implementation, policy=POLICY)
            atomic_json(case_root / 'stage3_pool_audit.json', audit)
        bundle = load_bundle(target, full=True)
        print('Fresh native archive pooled:', bundle.summary(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
