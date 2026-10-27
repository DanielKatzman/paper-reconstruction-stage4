"""Post-seal inspection of coarse completions, native growth and support failures."""
from pathlib import Path
import numpy as np
from paper_reconstruction_contracts import map_points
from .common import atomic_json, read_json
from .report import document, table, image, link, esc, number


def unaccepted_gallery(folder, case, row):
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.page_geometry import record_state
    from paper_reconstruction_stage4.preview import render
    from paper_reconstruction_stage4.outputs import native_probes
    folder.mkdir(parents=True, exist_ok=True)
    audit = row['search']['native_repair']; records = []
    checked = {}
    for verdict in audit['final_checks']:
        checked.setdefault(verdict.get('coarse_candidate_id'), []).append(verdict['disposition'])
    for record in audit.get('rejected_complete_candidates', []):
        records.append({**record, 'record_kind': 'rejected_final', 'state': record['diagnostic_state']})
    for search in audit['merge_searches']:
        for record in search.get('coarse_complete_records', []):
            records.append({**record, 'record_kind': 'coarse_generated', 'lane': search['lane'],
                            'final_dispositions': checked.get(record.get('candidate_id'), [])})
        for record in search.get('observed_growth_states', []):
            records.append({**record, 'record_kind': 'unverified_native_growth', 'lane': search['lane']})
    problem = load_problem(case['bundle'], SolverConfig.from_dict(row['configuration'])) if records else None
    reference = read_json(case['evaluation_reference']) if records else None
    refposes = {p: np.asarray(v['native_to_reference']) for p, v in reference['pieces'].items()} if reference else {}
    result_rows = []; cells = []; visuals = []
    for index, record in enumerate(records, 1):
        state = record_state(record['state'])
        groups = sorted(state.components(), key=lambda g: (-len(g), g))
        scope = groups[0]; anchor = min(scope)
        alignment = refposes[anchor]@np.linalg.inv(state.poses[anchor]); errors = []
        for pid in scope:
            if pid == anchor:
                continue
            probes = np.asarray(native_probes(problem.pieces[pid]))
            errors.extend(np.linalg.norm(map_points(probes, alignment@state.poses[pid])-map_points(probes, refposes[pid]), axis=1).tolist())
        error = float(np.quantile(errors, .95)) if errors else 0.
        measured = {**record, 'placement_p95_reference_px': error, 'alignment_scope_piece_ids': scope,
                    'whole_page_error_available': len(groups) == 1, 'reference_used_for_selection': False}
        result_rows.append(measured)
        pic = folder/f'candidate_{index:03d}.png'
        if index <= 50:
            state.assignments = {}; state.active_support = {}
            render(problem, state, pic, max_side=1000)
            visuals.append('<details><summary>'+esc(f'Record {index}: {record["record_kind"]}')+'</summary>'+image(pic, folder, 'Pose-only diagnostic')+'</details>')
        cells.append((esc(index), esc(record['record_kind']), esc(record.get('lane')), esc(list(map(len, groups))),
            number(error), esc('whole page' if len(groups) == 1 else 'largest group only'),
            esc(record.get('reason', record.get('final_dispositions'))), link(pic, folder, 'pose preview') if index <= 50 else esc('JSON only')))
    atomic_json(folder/'candidates.json', {'records': result_rows, 'post_seal_only': True, 'primary_changed': False,
        'reference_used_for_gallery_order': False, 'preview_limit': 50,
        'coarse_geometry_is_an_accepted_assembly': False})
    body = '<h1>Generated pose and final support diagnosis</h1><p>Records are in serving-attempt order. Coarse generated and native growth states are diagnostic proposals; final_dispositions records their later support check when identifiable. Partial-state errors use one shared alignment of the largest group, never a whole-page fit. Pose-only previews omit seams; JSON preserves native supports, cuts and source bindings. References are post-seal diagnostics and never solver seeds or ranking input.</p>'
    body += table(['Record', 'Kind', 'Lane', 'Components', 'Placement p95 px', 'Error scope', 'Final check / reason', 'Preview'], cells)
    body += '<p>'+link(folder/'candidates.json', folder, 'Full pose and support records')+'</p>'+''.join(visuals)
    (folder/'index.html').write_text(document('Generated pose diagnosis', body), encoding='utf-8')
    return folder/'index.html'
