"""Post-seal reference-assisted construction witness, never a blind result."""
import time
from pathlib import Path
from .common import atomic_json, file_hash
from .diagnostics import pose_error
from .scale_pipeline_diagnosis_v14 import accurate_graph


def filter_bank(proposals, reference, tolerance):
    contacts = {(r['piece_a'], r['piece_b']): r for r in reference['contacts']}
    filtered = []; best = {pair: None for pair in contacts}
    for p in proposals:
        pair = tuple(p['pieces'])
        if pair not in contacts:
            continue
        error = pose_error(p['transform_b_to_a'], contacts[pair])
        if error is None:
            continue
        best[pair] = error if best[pair] is None else min(error, best[pair])
        if error <= tolerance:
            filtered.append(p)
    rows = [{'piece_a': a, 'piece_b': b, 'error': error} for (a, b), error in best.items()]
    return filtered, accurate_graph(rows, reference['pieces'], 'error', tolerance)


def probe(case, result, reference, request, folder, barrier_sha256):
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.constraints import bind_profile
    from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
    from paper_reconstruction_stage4.page import make_page_context
    from paper_reconstruction_stage4.model import singleton_state
    from paper_reconstruction_stage4.native_graph_v18 import search
    from paper_reconstruction_stage4.feasibility import Feasibility
    from paper_reconstruction_stage4.outputs import serialize_layout
    from paper_reconstruction_stage4.preview import render
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    config = SolverConfig.from_dict(result['configuration'])
    proposals = result['search']['native_repair'].get('generated_pose_proposals', [])
    records = []
    for tolerance in request['constructibility_probe_tolerances_px']:
        filtered, graph = filter_bank(proposals, reference, tolerance)
        row = {'reference_filter_tolerance_px': tolerance, 'retained_existing_proposals': len(filtered),
            'accurate_native_graph': graph, 'complete': False, 'approximately_correct_reference_construction': False}
        if not graph['connected']:
            row['status'] = 'filtered_native_graph_disconnected'; records.append(row); continue
        problem = load_problem(case['bundle'], config)
        bind_profile(problem, config, request['profile'], case['capture_profile'])
        ctx = make_page_context(problem, config, ActiveSupportIndex(problem, config))
        problem.page_context = ctx; problem.structural = ctx.scorer()
        basis = singleton_state(problem); physical = Feasibility(problem, config); finals = []
        started = time.perf_counter(); ctx.repair_deadline = started+request['constructibility_probe_seconds_per_tolerance']

        def complete(node, shape):
            candidate, verdict = ctx.finalize(basis, node, physical)
            finals.append({'accepted': candidate is not None, 'verdict': verdict})
            return candidate

        candidate, trace = search(ctx, basis, filtered, ctx.repair_deadline, complete, physical)
        row.update(complete=candidate is not None, search=trace, final_checks=finals,
            search_and_verification_seconds=time.perf_counter()-started,
            status='compatible_native_assembly_demonstrated' if candidate is not None else 'inconclusive_filtered_search')
        if candidate is not None:
            layout = serialize_layout(problem, candidate, 0, physical.state(candidate, final=True), {'attempted': False})
            value = {'layouts': [layout]}
            row['independent_native_support_verification'] = verify_result_support(problem, value, config)
            row['layout'] = layout
            # evaluate() needs the ordinary envelope. No serving layout is changed.
            evaluated = {**result, 'layouts': [layout], 'run_id': 'reference-assisted-diagnostic'}
            ref = {**reference, 'input_manifest_sha256': case['bundle_sha256']}
            evaluation = evaluate(evaluated, ref)
            row['reference_measurement'] = assembly_quality(evaluated, ref, evaluation)
            row['false_neighbor_joins'] = evaluation['false_neighbor_joins']
            error = row['reference_measurement']['common_gauge']['non_anchor_probe_errors_reference_px']['p95']
            row['approximately_correct_reference_construction'] = bool(not evaluation['false_neighbor_joins'] and
                error is not None and error <= 48.)
            if not row['approximately_correct_reference_construction']:
                row['status'] = 'physical_candidate_demonstrated_reference_alignment_insufficient'
            render(problem, candidate, folder/'reference_assisted_diagnostic.png', max_side=1200)
        records.append(row)
        if row['approximately_correct_reference_construction']:
            break
    value = {'policy': 'saved-native-bank-constructibility-diagnostic.v1',
        'prediction_barrier_sha256': barrier_sha256,
        'reference_sha256': case['evaluation_reference_sha256'], 'records': records,
        'compatible_native_assembly_demonstrated': any(r['complete'] for r in records),
        'approximately_correct_native_assembly_demonstrated': any(r['approximately_correct_reference_construction'] for r in records),
        'offline_approximate_reference_p95_threshold_px': 48.,
        'reference_used_to_filter_existing_proposals': True, 'reference_poses_supplied_to_growth': False,
        'new_pose_fits_generated': False, 'old_assembly_poses_inherited': False,
        'primary_changed': False, 'counts_as_blind_success': False,
        'negative_result_proves_impossibility': False,
        'scope': 'A successful independently replayed construction is a positive witness from existing native modes. Failure, deadline, or a disconnected tolerance-filtered graph is inconclusive about other poses/supports or continuous search.'}
    atomic_json(folder/'diagnostic.json', value)
    return value
