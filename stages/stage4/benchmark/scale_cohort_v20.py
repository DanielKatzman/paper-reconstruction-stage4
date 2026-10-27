"""Six additional canonical captures, selected without reconstruction outcomes."""
from collections import Counter
import importlib.util
from pathlib import Path
from uuid import uuid4
from .common import WORKSPACE, DESKTOP, read_json, digest, file_hash, atomic_json, load_cohort
from .scale_pipeline_v14 import checked, require

COUNTS = {20: 2, 24: 2, 30: 2}
SEED = 20261026
COHORT = WORKSPACE/'benchmarks/scale_extension20_24_30_v1'
CANONICAL = DESKTOP/'paper_reconstruction_stage3_inputs_v1/pilot_canonical'
COMPILER = DESKTOP/'paper-reconstruction-stage3/scripts/clean_capture_benchmark.py'


def compiler():
    spec = importlib.util.spec_from_file_location('clean_scale_extension_compiler', COMPILER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def select(rows, excluded_pages, existing_families=()):
    """Two distinct pages per size; prefer simple/moderate and family diversity."""
    chosen = []; pages = set(excluded_pages); families = Counter(existing_families)
    for size, count in COUNTS.items():
        for slot in range(count):
            candidates = [r for r in rows if r['actual_piece_count'] == size and r['source_id'] not in pages and r['split'] == 'train']
            require(candidates, 'No unused canonical page at size '+str(size))
            tier = 'simple' if slot == 0 else 'moderate'
            row = min(candidates, key=lambda r: (r['difficulty_tier'] != tier,
                families[r['source_family']], ('simple', 'moderate', 'hard', 'extreme').index(r['difficulty_tier']),
                digest([SEED, r['puzzle_id']])))
            chosen.append(row); pages.add(row['source_id']); families[row['source_family']] += 1
    return sorted(chosen, key=lambda r: (r['actual_piece_count'], r['puzzle_id']))


def validate(value):
    require(dict(Counter(c['pieces'] for c in value['cases'])) == COUNTS, 'Require 2 each at 20/24/30 pieces')
    require(len({c['case_id'] for c in value['cases']}) == 6 and len({c['source_page_id'] for c in value['cases']}) == 6,
        'Duplicate extension puzzle/source page')
    p = value['capture_profile']
    require(p['shared_native_pixel_scale'] and not p['per_piece_resize'] and not p['perspective'] and
        p['rotation_family'] == 'quarter_turn' and p['stage1_masks'] == 'generator_exact_bypass', 'Capture contract differs')


def prepare(output=COHORT):
    output = Path(output).resolve()
    require(output.is_relative_to(WORKSPACE/'benchmarks') and output != WORKSPACE/'benchmarks', 'Cohort must stay inside workspace benchmarks')
    if output.exists():
        _, value = load_cohort(output); validate(value)
        for path, sha in value['compiler']['files'].items():
            require(file_hash(path) == sha, 'Compiler changed; preserve cohort and choose a new version')
        require(value['selection']['seed'] == SEED, 'Selection seed differs')
        return output, value
    module = compiler(); dataset = read_json(CANONICAL/'dataset_manifest.json')
    require(dataset['dataset_fingerprint'] == module.CANONICAL_SHA, 'Canonical dataset differs from passing baseline')
    baseline = read_json(DESKTOP/'paper-reconstruction-stage3/generated/clean_benchmarks/clean_shared_scale_v1/benchmark.json')
    checked(baseline, 'benchmark_sha256')
    excluded = {c['source_page_id'] for c in baseline['cases']}
    rows = []
    for raw in dataset['puzzles']:
        if raw['split'] != 'train' or raw['actual_piece_count'] not in COUNTS:
            continue
        puzzle = read_json(module.within(CANONICAL, raw['puzzle_path']))
        rows.append({**raw, 'source_family': Path(puzzle['source']['origin']).name})
    selected = select(rows, excluded, [c['source_family'] for c in baseline['cases']])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name+'.building_'+uuid4().hex[:10]); staging.mkdir()
    cases = []
    for row in selected:
        print('Compile clean '+row['puzzle_id']+' ('+str(row['actual_piece_count'])+' pieces)', flush=True)
        cases.append(module.export_case(CANONICAL, row, staging/'cases'/row['puzzle_id'], SEED))
    value = {'schema_version': 'stage4.development-benchmark.v1', 'name': output.name,
        'input_adapter_version': 'canonical-shared-scale-quarter-turn.v1',
        'canonical_dataset_fingerprint': dataset['dataset_fingerprint'], 'capture_profile': module.PROFILE,
        'stage1': {'status': 'bypassed', 'segmentation_accuracy_measured': False},
        'compiler': {'files': {str(p): file_hash(p) for p in (Path(__file__), COMPILER)},
            'baseline_cohort_sha256': baseline['benchmark_sha256']},
        'selection': {'seed': SEED, 'puzzles_per_size': COUNTS, 'outcomes_used': False,
            'excluded_baseline_page_ids': sorted(excluded),
            'policy': 'Metadata-only seeded selection; distinct unused train pages; prefer simple/moderate and source-family diversity.'},
        'cases': cases, 'purpose': 'Capacity stress test of the unchanged V19 r2 solver; exposed synthetic development data.'}
    value['benchmark_sha256'] = digest(value); atomic_json(staging/'benchmark.json', value); validate(value)
    checks = {c['case_id']: module.validate_case(staging, c, full=True) for c in cases}
    atomic_json(staging/'VALIDATION.json', {'benchmark_sha256': value['benchmark_sha256'], 'cases': checks,
        'inference_performed': False, 'shared_scale_verified': True})
    require(staging.resolve().parent == output.parent and staging.name.startswith(output.name+'.building_') and
        not output.exists(), 'Unsafe cohort publication target')
    staging.rename(output)
    return output, value
