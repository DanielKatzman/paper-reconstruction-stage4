"""Regenerate the V18 r2 report from sealed predictions; never run a solver."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bootstrap(run):
    """Authenticate original source and seals before loading a frozen reader."""
    request = read(run/'request.json'); barrier = read(run/'predictions_sealed.json')
    if request.get('schema_version') != 'stage4.v18-recovery-ablation-request.v1':
        raise ValueError('Use this repair only with a sealed V18 recovery run')
    if barrier.get('request_sha256') != request.get('request_sha256') or barrier.get('references_opened') is not False:
        raise ValueError('V18 prediction seal is missing or belongs to another run')
    for path, expected in request['controllers'].items():
        if sha(path) != expected:
            raise ValueError('Original controller changed: '+path)
    snapshot = Path(request['snapshot']).resolve(); receipt = read(snapshot/'receipt.json')
    for relative, expected in receipt['files'].items():
        path = (snapshot/relative).resolve()
        if not path.is_relative_to(snapshot) or sha(path) != expected:
            raise ValueError('Original snapshot changed: '+str(path))
    for path, expected in barrier['files'].items():
        if sha(path) != expected:
            raise ValueError('Sealed prediction changed: '+path)
    return request, snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv); run = args.run.resolve()
    if not run.is_relative_to(ROOT/'outputs') or run == ROOT/'outputs':
        raise ValueError('Run must be inside this workspace outputs folder')
    request, snapshot = bootstrap(run)
    sys.path.insert(0, str(ROOT))
    if args.worker:
        sys.path.insert(0, str(snapshot/'src'))
        from benchmark.recovery_report_v18_r2_fix import main as report_main
        return report_main(['--run', str(run)])

    from benchmark.common import atomic_json, digest, file_hash, inventory, verify_files
    from benchmark.runner import execute
    folder = run/'report_repair_v1'; folder.mkdir(exist_ok=True)
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['request_sha256'] != request['request_sha256']:
            raise ValueError('Report repair belongs to another run')
        if receipt['receipt_sha256'] != digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'}):
            raise ValueError('Changed report repair receipt')
        for path, expected in receipt['repair_sources'].items():
            if file_hash(path) != expected:
                raise ValueError('Report repair source changed: '+path)
        verify_files(run, receipt['files'])
        print('Repaired report verified:', run/'report/index.html', flush=True)
        return 0
    # Keep the crash evidence and original progress rather than replacing it.
    for name in ('diagnostics.log', 'diagnostics.attempt.json', 'progress.json'):
        source = run/name; target = folder/('original_'+name)
        if source.exists() and not target.exists():
            target.write_bytes(source.read_bytes())
    repair_sources = [Path(__file__), ROOT/'benchmark/recovery_report_v18_r2_fix.py']
    bindings = {str(p): file_hash(p) for p in repair_sources}
    started = time.perf_counter()
    index = len(list(folder.glob('diagnostics_*.log')))+1
    log = folder/f'diagnostics_{index:03d}.log'
    attempt = execute([sys.executable, '-B', Path(__file__).resolve(), '--run', run, '--worker'],
        log, env_overrides={'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        echo_progress=True)
    if attempt['returncode'] != 0:
        raise ValueError('Report-only repair failed; predictions remain intact. See '+str(log))
    bootstrap(run)  # Confirm every original sealed byte still agrees.
    for path, expected in bindings.items():
        if file_hash(path) != expected:
            raise ValueError('Report repair source changed during execution')
    report = read(run/'experiment_receipt.json'); verify_files(run, report['files'])
    progress = read(run/'progress.json')
    progress.setdefault('diagnostics_history', []).append(progress.get('diagnostics'))
    progress.update(phase='finished', diagnostics=attempt, report_repair='report_repair_v1/receipt.json')
    atomic_json(run/'progress.json', progress)
    evidence = ['request.json', 'predictions_sealed.json', 'experiment_receipt.json', 'progress.json']
    evidence += ['report_repair_v1/'+r['path'] for r in inventory(folder)]
    receipt = {'schema_version': 'stage4.v18-r2-report-repair-receipt.v1',
        'request_sha256': request['request_sha256'], 'prediction_seal_sha256': file_hash(run/'predictions_sealed.json'),
        'repair_sources': bindings, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter()-started, 'solver_run': False, 'predictions_changed': False,
        'files': [{**r, 'path': 'report/'+r['path']} for r in inventory(run/'report')] + [
            {'path': name, 'bytes': (run/name).stat().st_size, 'sha256': file_hash(run/name)} for name in evidence]}
    receipt['receipt_sha256'] = digest(receipt); atomic_json(receipt_path, receipt)
    print('Report repaired; all solver results preserved:', run/'report/index.html', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
