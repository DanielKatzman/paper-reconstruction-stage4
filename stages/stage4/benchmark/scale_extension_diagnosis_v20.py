"""Reference diagnosis after all six new fixed-solver attempts are sealed."""
import argparse
from pathlib import Path
import traceback
from .common import atomic_json, read_json, file_hash, inventory, verify_files
from .scale_extension_v20 import checked, require, verify_stage4


def timings(entry):
    values = {}
    for row in entry['attempts']:
        if 'elapsed_seconds' in row:
            key = row['stage']; values[key] = values.get(key, 0.)+row['elapsed_seconds']
    return values


def requires_probe(quality, evaluation, assessment):
    if not quality['all_pieces_connected'] or evaluation['false_neighbor_joins'] or assessment['status'] != 'supported':
        return True
    error = quality['common_gauge']['non_anchor_probe_errors_reference_px']['p95']
    return error is not None and error > 48.  # Offline approximate-placement diagnostic trigger only.


def case_review(output, request, case, progress, upstream_request, upstream_progress, barrier):
    from .diagnostics import stage1_audit, stage2_diagnosis, stage3_diagnosis, pose_error
    from .scale_pipeline_diagnosis_v14 import accurate_graph
    from .recovery_diagnosis_v19_r2 import retained_accuracy, native_diagnosis, write_native_gallery
    from .native_repair_experiment_v14 import write_gallery
    from .recovery_report_v18_r2_fix import write_closure_gallery
    from .constructibility_v20 import probe
    from .quality import assembly_quality, contact_path_diagnostics
    from evaluation.evaluate import evaluate
    from paper_reconstruction_stage4.outputs import read_result
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.constraints import bind_profile
    from paper_reconstruction_stage4.active_support import verify_result_support
    from paper_reconstruction_stage4.native_bridge import install_result_bridges
    cid = case['case_id']; source = Path(request['upstream'])/'cases'/cid; target = output/'cases'/cid
    require(file_hash(case['evaluation_reference']) == case['evaluation_reference_sha256'], 'Reference changed')
    reference = read_json(case['evaluation_reference']); stages = {}; errors = []
    stages['stage1'] = stage1_audit(Path(case['stage1']))
    for stage, fn in [('stage2', stage2_diagnosis), ('stage3', stage3_diagnosis)]:
        try:
            stages[stage] = fn(source/stage, reference, source/'stage3_native.work') if stage == 'stage3' else fn(source/stage, reference)
        except Exception as exc:
            stages[stage] = {'status': 'diagnostic_failed', 'error': str(exc)}
            errors.append({'stage': stage, 'error': str(exc), 'traceback': traceback.format_exc()})
    if stages['stage3']['status'] == 'complete':
        stages['stage3']['accuracy_graphs'] = {pool: [accurate_graph(stages['stage3']['contacts'], reference['pieces'], field, t)
            for t in (4, 8, 16, 32, 48)] for pool, field in [('archive', 'archive_best_p95_px'), ('working', 'working_best_p95_px')]}
    time = timings(upstream_progress['cases'][cid])
    time['stage4'] = sum(r['elapsed_seconds'] for r in progress['cases'][cid]['attempts'])
    row = {'case_id': cid, 'pieces': case['pieces'], 'reused_control': False, 'timing': time, 'stages': stages,
        'serving_status': progress['cases'][cid]['status'], 'errors': errors,
        'primary_result': str(target/'stage4'), 'visual_review': None, 'reference_used_during_prediction': False,
        'upstream_profile': read_json(source/'stage3_profile.json'), 'pool_audit': read_json(source/'stage3_pool_audit.json')}
    if row['serving_status'] != 'predicted':
        row['failure_category'] = 'stage4_process_failure'; return row
    result = read_result(target/'stage4'); config = SolverConfig.from_dict(result['configuration'])
    require(result['implementation'] == request['implementation'], 'Use frozen prediction reader')
    ref = {**reference, 'input_manifest_sha256': case['bundle_sha256']}
    evaluation = evaluate(result, ref); quality = assembly_quality(result, ref, evaluation)
    problem = load_problem(case['bundle'], config); bind_profile(problem, config, request['profile'], case['capture_profile'])
    install_result_bridges(problem, result, config); independent = verify_result_support(problem, result, config)
    assessment = read_json(target/'stage4.assessment/assessment.json')['layouts'][0]
    native = result['search']['native_repair']; searches = native['merge_searches']
    row.update(configuration=result['configuration'], quality=quality, evaluation=evaluation, assessment=assessment, search=result['search'],
        resources=result['resources'], independent_native_support_verification=independent,
        retained=retained_accuracy(result, reference), termination=result['termination'],
        contact_paths=list(contact_path_diagnostics(result['layouts'][0], reference['contacts'], pose_error).values()),
        native=native_diagnosis(result, reference, row),
        reference_bound_diagnostic={'scope': 'Original global observed-area bound is unchanged; unknown format/DPI.'},
        search_work={'native_fitting_seconds': sum((native.get(k) or {}).get('elapsed_seconds', 0.)
            for k in ('native_generation', 'expanded_native_generation')),
            'merge_seconds': sum(s['elapsed_seconds'] for s in searches),
            'native_complete_shape_candidates': sum(s['complete_shape_candidates'] for s in searches),
            'native_final_checks': len(native['final_checks']), 'deepest_native_growth': max(
                (s.get('completed_depth', 1) for s in searches), default=1),
            'growth_and_support_deadlines': native.get('route_stop_reason'),
            'geometry_expansion': native.get('geometry_expansion')})
    gallery_root = output/'report/new_cases'; context = {**case}
    row['galleries'] = {'geometry': write_gallery(gallery_root, context, result, {'configuration': request['configuration']}),
        'closure': write_closure_gallery(gallery_root, context, result),
        'native': write_native_gallery(gallery_root/'native_recovery'/cid, context, row)}
    if requires_probe(quality, evaluation, assessment):
        print('[constructibility] '+cid+': post-seal reference-assisted saved-bank check', flush=True)
        try:
            row['bank_constructibility'] = probe(case, result, reference, request,
                output/'report/constructibility'/cid, barrier['barrier_sha256'])
        except Exception as exc:
            row['bank_constructibility'] = {'status': 'diagnostic_failed', 'error': str(exc),
                'negative_result_proves_impossibility': False, 'counts_as_blind_success': False}
            errors.append({'stage': 'constructibility_diagnostic', 'error': str(exc), 'traceback': traceback.format_exc()})
    else:
        row['bank_constructibility'] = {'status': 'not_needed_for_approximately_correct_supported_primary',
            'counts_as_blind_success': False, 'negative_result_proves_impossibility': False}
    # These are evidence categories, never a blanket allocation of blame.
    witness = row['bank_constructibility'].get('approximately_correct_native_assembly_demonstrated', False)
    row['failure_category'] = ('compatible_bank_construction_demonstrated_search_or_selection_gap' if witness else
        'unresolved_compatible_construction_not_demonstrated' if requires_probe(quality, evaluation, assessment) else
        'supported_approximate_construction_pending_visual_review')
    return row


def report(output, request, rows):
    from .report import document, esc, table, number, link, image
    root = output/'report'; root.mkdir(exist_ok=True); cells = []; pictures = []
    for row in rows:
        cid = row['case_id']; q = row.get('quality', {}); e = row.get('evaluation', {}); a = row.get('assessment', {})
        p95 = q.get('common_gauge', {}).get('non_anchor_probe_errors_reference_px', {}).get('p95') if q.get('all_pieces_connected') else None
        t = row['timing']; proof = row.get('bank_constructibility', {})
        cells.append((esc(cid), esc(row['pieces']), esc('reused control' if row['reused_control'] else 'fresh larger case'),
            esc(q.get('component_sizes')), esc(a.get('status')), esc(len(e.get('false_neighbor_joins', []))) if e else esc(None),
            number(p95), number(t.get('stage2')), number(t.get('stage3_native')), number(t.get('stage4')),
            esc(proof.get('approximately_correct_native_assembly_demonstrated', proof.get('status')))))
        folder = root/cid; folder.mkdir(exist_ok=True); atomic_json(folder/'diagnosis.json', row)
        preview = Path(row['primary_result'])/'preview.png'
        text = '<h2>'+esc(cid)+'</h2>'
        text += image(preview, root, 'Selected primary assembly') if preview.exists() else '<p>No primary assembly was published.</p>'
        text += '<p>'+link(folder/'diagnosis.json', root, 'Complete stage diagnoses, bank coverage and search work')+'</p>'
        if row.get('galleries'):
            text += '<p>'+link(Path(row['galleries']['geometry']['index']), root, 'Retained complete/partial bank and rejection reasons')+' | '+link(
                Path(row['galleries']['native']), root, 'Native coverage, growth traces and final support failures')+'</p>'
        if proof.get('records'):
            text += '<p>Reference-assisted diagnostic only: '+link(root/'constructibility'/cid/'diagnostic.json', root, 'Saved-bank construction witness / inconclusive attempt')+'</p>'
            pic = root/'constructibility'/cid/'reference_assisted_diagnostic.png'
            if pic.exists():
                text += '<details><summary>Reference-assisted witness; excluded from blind success counts</summary>'+image(pic, root, 'Diagnostic only')+'</details>'
        pictures.append(text)
    intro = '<h1>Stage 4 capacity: two each at 4, 8, 12, 16, 20, 24 and 30 pieces</h1><p>The eight smaller controls reuse the sealed confirmation run; the six larger cases run fresh Stage 2/3 preparation and cold Stage 4 with the identical V19 r2 solver, configuration, capture assumptions and 600-second cooperative solve deadline. No old assembly seeds enter new solves. Reports open references only after all six attempts are sealed. Exact generator masks bypass Stage 1 inference.</p><p>A missed solve is not proof of an intrinsic size limit. Pair-pose connectivity is necessary evidence, not a compatible construction. Failed or inaccurate cases receive a reference-assisted saved-native-bank construction check. A successful independently verified witness demonstrates an available construction; a failed/expired check is inconclusive. The check neither supplies reference placements nor generates new pair fits, and never changes a primary result or counts as blind success. Four-pixel metrics are diagnostic. Supported pages still need visual review. New source pages are disjoint from the eight controls; this remains synthetic train-split development data.</p>'
    body = table(['Case', 'Pieces', 'Source', 'Groups', 'Assessment', 'False pairs', 'Whole-page p95 px',
        'Stage 2 s', 'Stage 3 s', 'Stage 4 worker s', 'Compatible native-bank witness'], cells)
    (root/'index.html').write_text(document('Capacity benchmark to 30 pieces', intro+body+''.join(pictures)), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    output = parser.parse_args(argv).run.resolve(); request = checked(read_json(output/'request.json'), 'request_sha256')
    barrier = checked(read_json(output/'predictions_sealed.json'), 'barrier_sha256'); progress = read_json(output/'progress.json')
    require(barrier['request_sha256'] == request['request_sha256'] and barrier['references_opened'] is False, 'Predictions are not sealed')
    verified = verify_stage4(output, request, progress)
    require(all(barrier[k] == v for k, v in verified.items()), 'Prediction seal differs')
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity() == request['implementation'], 'Use the exact frozen reader')
    receipt = output/'experiment_receipt.json'
    if receipt.exists():
        verify_files(output, read_json(receipt)['files']); print('Capacity report verified:', output/'report/index.html'); return 0
    # Baseline diagnoses contain references and are parsed only after the new seal.
    baseline = Path(request['baseline']['root']); old = read_json(baseline/'report/summary.json')
    rows = []
    for item in old['cases']:
        r = item['variants']['adaptive']; cid = item['case_id']
        rows.append({'case_id': cid, 'pieces': item['pieces'], 'reused_control': True,
            'primary_result': str(baseline/'cases'/cid/'adaptive/stage4'),
            'quality': r['quality'], 'evaluation': r['evaluation'], 'assessment': r['assessment'],
            'timing': {'stage4': r['stage4_subprocess_seconds']}, 'native': r['native'],
            'stages': {'stage2': item['unchanged_upstream_stage2'], 'stage3': item['unchanged_upstream_stage3']},
            'original_diagnosis': str(baseline/'report'/cid/'diagnosis.json'),
            'bank_constructibility': {'status': 'reused_passed_control'}, 'visual_review': None})
    upstream = Path(request['upstream']); up_request = read_json(upstream/'request.json'); up_progress = read_json(upstream/'progress.json')
    for case in request['cases']:
        print('[diagnostics] '+case['case_id'], flush=True)
        rows.append(case_review(output, request, case, progress, up_request, up_progress, barrier))
    rows.sort(key=lambda r: (r['pieces'], r['case_id'])); report(output, request, rows)
    atomic_json(output/'report/summary.json', {'schema_version': 'stage4.scale-extension-review.v1',
        'request_sha256': request['request_sha256'], 'barrier_sha256': barrier['barrier_sha256'],
        'cases': rows, 'controls_reused': 8, 'new_cold_stage4_cases': 6, 'references_opened_after_seal': True,
        'solver_changed': False, 'all_diagnostic_probes_excluded_from_blind_success_counts': True})
    atomic_json(receipt, {'request_sha256': request['request_sha256'], 'barrier_sha256': barrier['barrier_sha256'],
        'files': [{**r, 'path': 'report/'+r['path']} for r in inventory(output/'report')]})
    print('Capacity diagnosis complete:', output/'report/index.html', flush=True); return 0
