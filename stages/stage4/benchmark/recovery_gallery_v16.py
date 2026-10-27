"""Post-seal pose-only previews of failed complete candidates; never solver input."""
from pathlib import Path
import numpy as np
from paper_reconstruction_contracts import map_points
from .common import atomic_json, read_json
from .report import document, table, image, link, esc, number


def rejected_gallery(folder, case, row):
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.page_geometry import record_state
    from paper_reconstruction_stage4.preview import render
    folder.mkdir(parents=True, exist_ok=True)
    records = row['search']['native_repair'].get('rejected_complete_candidates', [])
    problem = load_problem(case['bundle'], SolverConfig.from_dict(row['configuration'])) if records else None
    # The cohort reference is opened only by the caller's post-seal diagnosis.
    reference = read_json(case['evaluation_reference']) if records else None
    reference_poses = {p: np.asarray(v['native_to_reference']) for p, v in reference['pieces'].items()} if reference else {}
    result_rows = []; cells = []; visuals = []
    for index, record in enumerate(records, 1):
        state = record_state(record['diagnostic_state']); anchor = min(state.poses)
        alignment = reference_poses[anchor] @ np.linalg.inv(state.poses[anchor]); errors = []
        for pid, pose in state.poses.items():
            if pid == anchor:
                continue
            from paper_reconstruction_stage4.outputs import native_probes
            probes = np.asarray(native_probes(problem.pieces[pid]))
            errors.extend(np.linalg.norm(map_points(probes, alignment@pose)-map_points(probes, reference_poses[pid]), axis=1).tolist())
        error = float(np.quantile(errors, .95)) if errors else 0.
        result_rows.append({**record, 'whole_page_p95_reference_px': error,
                            'reference_diagnostic_used_for_selection': False})
        # No failed derived edge is installed as valid support for this preview.
        state.assignments = {}; state.active_support = {}
        pic = folder/f'candidate_{index:03d}.png'; render(problem, state, pic, max_side=1000)
        cells.append((esc(index), esc(record.get('lane')), esc(record['disposition']),
                      number(error), esc(record.get('reason')), esc(len(record.get('unsupported_bridges', []))), link(pic, folder, 'pose preview')))
        visuals.append('<details><summary>'+esc(f'Rejected candidate {index}')+'</summary>'+image(pic, folder, 'Pose-only diagnostic')+'</details>')
    atomic_json(folder/'candidates.json', {'records': result_rows, 'post_seal_only': True,
                                          'primary_changed': False, 'reference_used_for_gallery_order': False})
    body = '<h1>Rejected complete candidate diagnosis</h1><p>Sealed failed candidate poses, shown in original attempt order. These are unaccepted diagnostic layouts. Preview seams are omitted; JSON preserves exact proposed native supports, offending bridge IDs, graph cuts and occupied source arcs. Reference placement errors are post-seal diagnostics, never ranking or solver seeds.</p>'
    body += table(['Attempt', 'Lane', 'Disposition', 'Whole-page p95 px', 'Reason', 'Unsupported bridges', 'Preview'], cells)
    body += '<p>'+link(folder/'candidates.json', folder, 'Full pose and support records')+'</p>'+''.join(visuals)
    (folder/'index.html').write_text(document('Rejected complete candidates', body), encoding='utf-8')
    return folder/'index.html'
