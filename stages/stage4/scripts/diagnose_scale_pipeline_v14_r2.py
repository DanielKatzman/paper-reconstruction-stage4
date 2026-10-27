"""Post-seal report adapter for an optional null native-generation audit.

The original bound controller stays byte-identical. This revision changes only
the human-readable findings formatter; predictions and diagnostic numbers stay
unchanged. Its own source receipt is included in the sealed report inventory.
"""
import hashlib
import json
from pathlib import Path
import sys


def normalize_optional_generation(row):
    stage4 = row['stages'].get('stage4', {})
    if stage4.get('status') != 'complete':
        return row
    search = stage4['search']; native = search['native_repair']
    if native.get('native_generation') is not None:
        return row
    return {**row, 'stages': {**row['stages'], 'stage4': {**stage4,
        'search': {**search, 'native_repair': {**native, 'native_generation': {}}}}}}


def main():
    run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
    request = json.loads((run / 'request.json').read_text())
    for name, sha in request['controllers'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError('Changed pipeline controller: ' + name)
    snapshot = Path(request['snapshot']); receipt = json.loads((snapshot / 'receipt.json').read_text())
    for name, sha in receipt['files'].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Changed frozen reader: ' + name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(snapshot / 'src'))
    from benchmark.common import atomic_json, read_json, file_hash, digest
    from benchmark.scale_pipeline_v14 import checked, authenticate, require
    from benchmark import scale_pipeline_diagnosis_v14 as base
    from benchmark import diagnostics
    from benchmark.quality import contact_path_diagnostics
    # The frozen first launcher imported this helper from diagnostics although
    # its implementation lives in quality. Alias the same bound function here.
    diagnostics.contact_path_diagnostics = contact_path_diagnostics
    barrier = checked(read_json(run / 'predictions_sealed.json'), 'barrier_sha256')
    authenticate(request)
    require(barrier['request_sha256'] == request['request_sha256'] and barrier['references_opened'] is False, 'Unverified prediction barrier')
    for name, sha in barrier['files'].items():
        require(file_hash(name) == sha, 'Sealed prediction bytes changed: ' + name)
    # base.main performs full bundle/result/assessment and inventory validation
    # before opening any reference; do not duplicate that expensive scan here.
    revision = {'schema_version': 'stage4.pipeline-diagnosis-revision.v1', 'request_sha256': request['request_sha256'],
        'source': str(Path(__file__).resolve()), 'source_sha256': file_hash(__file__),
        'change': 'Restore the bound contact-path helper import; format optional null native-generation audits and distinguish generation/merge budgets.',
        'timing_attempt_receipts': {p.relative_to(run).as_posix(): file_hash(p) for p in sorted((run / 'cases').glob('*/*.attempt.json'))},
        'serving_progress_cases_sha256': digest(read_json(run / 'progress.json')['cases']),
        'prediction_or_numeric_diagnostic_changes': False}
    path = run / 'report/diagnostic_revision.json'
    if path.exists():
        require(read_json(path) == revision, 'Diagnosis revision changed')
    else:
        require(not (run / 'experiment_receipt.json').exists(), 'Cannot alter an existing sealed report')
        atomic_json(path, revision)
    original = base.findings
    def findings(row):
        lines = original(normalize_optional_generation(row))
        generation = row['stages'].get('stage4', {}).get('search', {}).get('native_repair', {}).get('native_generation') or {}
        if generation:
            lines.append(f"Native proposal generation separately: {generation['native_checks']}/{generation['maximum_native_checks']} checks; {generation['untested_pair_count']} physical pairs untested; generation budget abstained {generation['budget_abstained']}. The merge-route budget flag is separate.")
        return lines
    base.findings = findings
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
