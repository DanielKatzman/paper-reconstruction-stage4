"""Reference-aware diagnostics, launched only after all eight attempts are sealed."""
import argparse
from collections import Counter
from pathlib import Path
import traceback

from .common import atomic_json, digest, file_hash, inventory, read_json, verify_files
from .scale_pipeline_v14 import authenticate, checked, require, verify_predictions


def accurate_graph(contacts, piece_ids, field, tolerance):
    adjacency = {p: set() for p in piece_ids}
    for row in contacts:
        error = row.get(field)
        if error is not None and error <= tolerance:
            a, b = row['piece_a'], row['piece_b']
            adjacency[a].add(b); adjacency[b].add(a)
    remaining = set(adjacency); components = []
    while remaining:
        todo = [min(remaining)]; found = set()
        while todo:
            piece = todo.pop()
            if piece not in found:
                found.add(piece); todo.extend(adjacency[piece] - found)
        components.append(sorted(found)); remaining -= found
    return {'tolerance_native_px': tolerance, 'component_sizes': sorted(map(len, components), reverse=True),
            'components': components, 'connected': len(components) == 1,
            'scope': 'Independent direct poses on true neighbor pairs only; compatible supports/global feasibility are not established.'}


def durations(entry):
    result = {}
    for attempt in entry.get('attempts', []):
        if 'elapsed_seconds' in attempt:
            result[attempt['stage']] = result.get(attempt['stage'], 0.) + attempt['elapsed_seconds']
    result['pipeline_total'] = sum(result.values())
    result['scope'] = 'Measured subprocess wall time, including model startup, validation and assessment; all local attempts included. Stage 1 inference bypassed.'
    return result


def case_diagnosis(run, request, case, entry):
    from .diagnostics import stage1_audit, stage2_diagnosis, stage3_diagnosis, contact_path_diagnostics, pose_error
    from .quality import assembly_quality
    from evaluation.evaluate import evaluate
    from paper_reconstruction_stage4.outputs import read_result
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.constraints import bind_profile
    from paper_reconstruction_stage4.native_bridge import install_result_bridges
    from paper_reconstruction_stage4.active_support import verify_result_support
    from .native_repair_experiment_v14 import write_gallery, write_closure_gallery, write_repair_gallery
    cid = case['case_id']; output = run / 'cases' / cid
    inputs = Path(request['cohort_root']) / case['path']
    # This function can be reached only after the parent verified the full barrier.
    verify_files(inputs, [f for f in case['files'] if f['path'].startswith('evaluation_only/')])
    reference = read_json(inputs / 'evaluation_only/reference_geometry.json')
    row = {'case_id': cid, 'pieces': case['pieces'], 'serving_status': entry['status'],
           'failed_stage': entry.get('failed_stage'), 'timing': durations(entry), 'stages': {},
           'errors': [], 'outputs': inventory(output), 'visual_review': None, 'four_pixel_metrics_are_diagnostic_only': True}
    row['stages']['stage1'] = stage1_audit(inputs / 'stage1')
    for stage, sentinel, fn in [('stage2', 'edge_hypotheses_manifest.json', stage2_diagnosis), ('stage3', 'bundle.json', stage3_diagnosis)]:
        if not (output / stage / sentinel).exists():
            row['stages'][stage] = {'status': 'not_published'}; continue
        try:
            row['stages'][stage] = fn(output / stage, reference, output / 'stage3_native.work') if stage == 'stage3' else fn(output / stage, reference)
        except Exception as exc:
            row['stages'][stage] = {'status': 'diagnostic_failed', 'error': str(exc)}
            row['errors'].append({'stage': stage, 'error': str(exc), 'traceback': traceback.format_exc()})
    s3 = row['stages']['stage3']
    if s3['status'] == 'complete':
        s3['accuracy_graphs'] = {pool: [accurate_graph(s3['contacts'], reference['pieces'], field, t) for t in (4, 16, 32, 48)]
                                 for pool, field in [('archive', 'archive_best_p95_px'), ('working', 'working_best_p95_px')]}
    for name in ('stage3_profile', 'stage3_pool_audit'):
        if (output / (name + '.json')).exists():
            row[name] = read_json(output / (name + '.json'))
    if entry['status'] == 'predicted':
        result = read_result(output / 'stage4'); reference['input_manifest_sha256'] = result['input']['manifest_sha256']
        evaluation = evaluate(result, reference); quality = assembly_quality(result, reference, evaluation)
        config = SolverConfig.from_dict(result['configuration']); problem = load_problem(output / 'stage3', config)
        bind_profile(problem, config, request['profile'], inputs / 'stage1/capture_profile.json')
        require(result['search']['constraint_profile'] == problem.constraint_profile, 'Capture constraint binding differs')
        install_result_bridges(problem, result, config)
        independent = verify_result_support(problem, result, config)
        assessment = read_json(output / 'stage4.assessment/assessment.json')['layouts'][0]
        row['stages']['stage4'] = {'status': 'complete', 'assembly_quality': quality, 'evaluation': evaluation,
            'assessment': assessment, 'search': result['search'], 'resources': result['resources'],
            'termination': result['termination'], 'independent_native_support_verification': independent,
            'contact_paths': list(contact_path_diagnostics(result['layouts'][0], reference['contacts'], pose_error).values())}
        case_context = {'case_id': cid, 'bundle': str(output / 'stage3'), 'constraint_binding': problem.constraint_profile}
        row['galleries'] = {'geometry': write_gallery(run / 'report', case_context, result, request),
            'component_placement': write_closure_gallery(run / 'report', case_context, result),
            'native_repair': write_repair_gallery(run / 'report', case_context, result)}
    else:
        row['stages']['stage4'] = {'status': 'not_published', 'accuracy': None}
    return row


def findings(row):
    stages = row['stages']; result = []
    if row['serving_status'] != 'predicted':
        result.append('Pipeline stopped at ' + str(row['failed_stage']) + '; downstream assembly quality is unavailable.')
    s2, s3, s4 = (stages.get(s, {}) for s in ('stage2', 'stage3', 'stage4'))
    if s2.get('status') == 'complete':
        contacts = s2['contacts']
        result.append(f"Stage 2 candidate seam containment on both sides: {sum(c['at_least_one_seam_contained_both_sides_0_8'] for c in contacts)}/{len(contacts)} true neighbor pairs (80% seam coverage).")
    if s3.get('status') == 'complete':
        for tolerance in (4, 32):
            contacts = s3['contacts']; a = sum(c['archive_best_p95_px'] is not None and c['archive_best_p95_px'] <= tolerance for c in contacts)
            w = sum(c['working_best_p95_px'] is not None and c['working_best_p95_px'] <= tolerance for c in contacts)
            result.append(f'Stage 3 direct true-neighbor poses at {tolerance}px: archive {a}/{len(contacts)}, working pool {w}/{len(contacts)}. This is availability, not a proof of compatible full assembly.')
    if s4.get('status') == 'complete':
        quality = s4['assembly_quality']; a = s4['assessment']; native = s4['search']['native_repair']
        result.append(f"Stage 4 components {quality['component_sizes']}; page assessment {a['status']}; true selected neighbors {quality['selected_true_neighbor_pairs']}/{quality['selected_physical_pairs']}.")
        result.append(f"Native repair: {native['verified_completions']} verified completions; budget abstained {native['budget_abstained']}; {native.get('native_generation', {}).get('native_checks', 0)} native checks. Inspect sealed trace for omitted sources/pairs.")
    return result


def write_report(run, request, rows):
    from .report import document, esc, table, number, link, image
    report = run / 'report'; cells = []; sections = []
    for row in rows:
        cid = row['case_id']; output = run / 'cases' / cid
        inputs = Path(request['cohort_root']) / next(c['path'] for c in request['cases'] if c['case_id'] == cid)
        s4 = row['stages']['stage4']; q = s4.get('assembly_quality', {}); a = s4.get('assessment', {})
        error = q.get('common_gauge', {}).get('non_anchor_probe_errors_reference_px', {}).get('p95') if q.get('all_pieces_connected') else None
        t = row['timing']
        cells.append((esc(cid), esc(row['pieces']), esc(row['serving_status']), esc(q.get('component_sizes')),
            esc(a.get('status')), number(a.get('features', {}).get('fill_fraction')), number(error),
            number(t.get('stage2')), number(t.get('stage3_native')), number(t.get('stage4')), number(t['pipeline_total'])))
        row['findings'] = findings(row)
        details = '<h2>' + esc(cid) + '</h2><ul>' + ''.join('<li>' + esc(f) + '</li>' for f in row['findings']) + '</ul>'
        details += '<div class="visuals">' + image(output / 'stage4/preview.png', report, 'Frozen V14 primary') + image(inputs / 'evaluation_only/reference_page.png', report, 'Reference: post-seal review only') + '</div>'
        artifacts = [('stage1/input_pieces.jpg', inputs, 'Generator input pieces'),
            ('stage2/edge_hypotheses_manifest.json', output, 'Stage 2 manifest'),
            ('stage2.log', output, 'Stage 2 log'), ('stage2_verify.log', output, 'Stage 2 independent verification'),
            ('stage3_native/bundle.json', output, 'Stage 3 native manifest'), ('stage3_native.work/rankings.json', output, 'Fresh learned rankings'),
            ('stage3_native.work/parent_selection.json', output, 'Stage 3 selected parents'), ('stage3_native.work/completion.json', output, 'Stage 3 completion'),
            ('stage3_profile.json', output, 'Stage 3 timing profile'), ('stage3_pool_audit.json', output, 'Pool replay audit'),
            ('stage3/bundle.json', output, 'Stage 4 input bundle'), ('stage3_native.log', output, 'Stage 3 full log'),
            ('stage4/result.json', output, 'Complete solver output / budget trace'), ('stage4.log', output, 'Stage 4 log'),
            ('stage4.assessment/report.md', output, 'Page geometry assessment'), ('stage4.assessment/pose_recovery_request.json', output, 'Recovery request'),
            (cid + '/diagnosis.json', report, 'All stage diagnoses, contacts and output inventory')]
        for gallery in row.get('galleries', {}).values():
            details += '<p>' + link(Path(gallery['index']), report, Path(gallery['index']).parent.parent.name + ' gallery') + '</p>'
        details += '<p>' + ' | '.join(link(base / rel, report, label) for rel, base, label in artifacts if (base / rel).is_file()) + '</p>'
        if row['stages']['stage3'].get('status') == 'complete':
            contacts = row['stages']['stage3']['contacts']
            details += table(['True pair', 'Retained for pose search', 'Best archive p95 px', 'Best pool p95 px', 'Strict 4px rank in pool'],
                [(esc(c['piece_a'] + ' / ' + c['piece_b']), esc(c['physical_pair_retained_for_pose_search']), number(c['archive_best_p95_px']),
                  number(c['working_best_p95_px']), esc(c['best_accurate_working_weight_rank'])) for c in contacts])
        sections.append(details)
        atomic_json(report / cid / 'diagnosis.json', row)
    intro = '<h1>Fresh pipeline scaling benchmark: 2 each at 4, 8, 12, 16 pieces</h1><p>Stage 1 uses exact generator cuts. Stage 2 inference, learned Stage 3 ranking, unit-scale pose search, archive pool selection and frozen V14 assembly ran in order. Shared scale, quarter turns, complete rectangular page, unknown format and DPI. These are exposed synthetic development cases. References were opened only after all eight prediction attempts were sealed.</p><p>Correct overall placement is the goal. The 4px tests are diagnostic only; page acceptance and connectivity do not independently prove correct content placement. Times are measured subprocess wall seconds, including initialization and validation. CUDA is used for learned ranking; geometry and Stage 4 run with one CPU thread. Missing downstream output is recorded as unavailable.</p>'
    (report / 'index.html').write_text(document('V14 full pipeline scaling benchmark', intro + table(
        ['Case', 'Pieces', 'Pipeline', 'Components', 'Page assessment', 'Fill', 'Whole-page p95 px', 'Stage 2 s', 'Stage 3 s', 'Stage 4 s', 'Pipeline s'], cells) + ''.join(sections)), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args(argv).run.resolve(); request = checked(read_json(run / 'request.json'), 'request_sha256')
    barrier = checked(read_json(run / 'predictions_sealed.json'), 'barrier_sha256'); progress = read_json(run / 'progress.json')
    authenticate(request)
    require(barrier['request_sha256'] == request['request_sha256'] and barrier['references_opened'] is False, 'Foreign or unsealed predictions')
    sealed = verify_predictions(run, request, progress)
    require(sealed['files'] == barrier['files'] and sealed['cases'] == barrier['cases'], 'Prediction barrier differs')
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity() == request['implementation'], 'Diagnostics require the frozen consumer')
    receipt_path = run / 'experiment_receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt['request_sha256'] == request['request_sha256'], 'Foreign diagnostic receipt')
        verify_files(run, receipt['files']); print('Fresh pipeline report verified', flush=True); return 0
    (run / 'report').mkdir(exist_ok=True); rows = []
    for case in request['cases']:
        print('[diagnostics] ' + case['case_id'], flush=True)
        folder = run / 'report' / case['case_id']; folder.mkdir(exist_ok=True)
        row = case_diagnosis(run, request, case, progress['cases'][case['case_id']]); rows.append(row)
    write_report(run, request, rows)
    atomic_json(run / 'report/summary.json', {'schema_version': 'stage4.fresh-scale-pipeline-diagnosis.v1',
        'request_sha256': request['request_sha256'], 'barrier_sha256': barrier['barrier_sha256'], 'cases': rows,
        'reference_access': 'post-seal diagnostic process only', 'default_promoted': False,
        'counts_by_size': dict(Counter(r['pieces'] for r in rows)), 'construction_requires_visual_review': True})
    receipt = {'schema_version': 'stage4.fresh-scale-pipeline-receipt.v1', 'request_sha256': request['request_sha256'],
        'barrier_sha256': barrier['barrier_sha256'], 'files': [{**f, 'path': 'report/' + f['path']} for f in inventory(run / 'report')]}
    atomic_json(receipt_path, receipt)
    print('Fresh pipeline diagnostics completed:', run / 'report/index.html', flush=True); return 0


if __name__ == '__main__':
    raise SystemExit(main())
