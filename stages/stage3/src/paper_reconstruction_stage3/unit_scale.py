"""Declared common-scale geometry. Copied into an isolated runtime snapshot.

No page coordinates, puzzle IDs, reference seams or learned weights enter here.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

POLICY = {'name': 'declared-unit-scale.v1', 'motion_model': 'SE2', 'relative_scale': 1.,
          'reflection': False, 'projective': False, 'overlap_lengths': 'shared_native_pixels',
          'rotation': 'continuous', 'matrix_tolerance': 1e-7}


def require_rigid(matrix):
    h = np.asarray(matrix, float); tolerance = POLICY['matrix_tolerance']
    if (h.shape != (3, 3) or not np.isfinite(h).all()
            or not np.allclose(h[2], [0., 0., 1.], atol=tolerance, rtol=0.)
            or not np.allclose(h[:2, :2].T @ h[:2, :2], np.eye(2), atol=tolerance, rtol=0.)
            or abs(np.linalg.det(h[:2, :2])-1.) > tolerance):
        raise ValueError('Unit-scale mode requires a proper rigid native-pixel transform')
    return h


def configuration(stage1, original, receipts):
    from .alpha_inputs import Files
    path = Path(stage1)/'capture_profile.json'; files = Files(); profile = files.read(path)
    if (profile.get('schema_version') != 'stage1.capture-profile.v1'
            or type(profile.get('relative_piece_scale')) not in (int, float) or profile.get('relative_piece_scale') != 1.
            or profile.get('shared_native_pixel_scale') is not True
            or profile.get('motion_model') != 'rotation_translation'
            or profile.get('per_piece_resize') is not False
            or profile.get('perspective') is not False):
        raise ValueError('Unit-scale mode requires an explicit shared native-pixel capture profile')
    config = deepcopy(original)
    config['pose_configuration'].update(family='rigid', minimum_scale=1., maximum_scale=1.)
    config['unit_scale_geometry'] = {**POLICY, 'capture_profile_sha256': files.receipt()[0]['sha256']}
    return config, [*receipts, *files.receipt()]


def selection_rows(parents, gallery):
    return [{'query': p['query'], 'target': p['target'], 'pair': sorted(
        gallery[i][1]['entry']['piece_id'] for i in (p['query'], p['target'])),
        'parent_rank': p['parent_rank'], 'retrieval_score': p['retrieval_score'],
        'ranker_scores': p['ranker_scores'], 'windows': deepcopy(p['windows'])} for p in parents]


def native_parameters(first, second, base, search):
    """Same native overlap length on both curves, with unknown start positions."""
    lengths = [float(f['observed_length_px'][0]) for f in (first, second)]
    if not np.isfinite(lengths).all() or min(lengths) <= 0:
        raise ValueError('Invalid observed native curve lengths')
    minimum = base['minimum_fit_length_px']; maximum = min(lengths)
    if maximum < minimum:
        return []
    shared = sorted({min(maximum, fraction*length) for length in lengths
                     for fraction in search['overlap_fractions'] if fraction*length >= minimum}, reverse=True)
    if maximum >= minimum:
        shared = sorted(set(shared+[maximum, float(minimum)]), reverse=True)
    values = []
    for length in shared:
        fractions = [length/x for x in lengths]; starts = []
        for fraction in fractions:
            count = max(1, int(np.ceil((1-fraction)*search['overlap_position_divisions'])))
            starts.append(np.linspace(0., max(0., 1-fraction), count+1) if fraction < 1.-1e-12 else [0.])
        values.extend((float(a), fractions[0], float(b), fractions[1]) for a in starts[0] for b in starts[1])
    return values


def refinement_parameters(current, lengths, minimum, step):
    """Translate either subarc or change its shared native length; never scale."""
    parameters, owners = [], []
    for i, pose in enumerate(current):
        for axis in (0, 2, 1):
            for direction in (-1, 1):
                trial = np.asarray(pose['_parameters']).copy()
                if axis == 1:
                    delta = direction*step*min(lengths)
                    trial[1] += delta/lengths[0]; trial[3] += delta/lengths[1]
                else:
                    trial[axis] += direction*step
                if any(trial[k] < -1e-12 or trial[k+1]*lengths[k//2] < minimum-1e-10
                       or trial[k]+trial[k+1] > 1.+1e-12 for k in (0, 2)):
                    continue
                parameters.append(trial); owners.append(i)
    return parameters, owners


def interpolate_rigid(first, second, fraction):
    """Interpolate in SE2; a linear blend of rotations would shrink the paper."""
    a, b = require_rigid(first), require_rigid(second)
    start = np.arctan2(a[1, 0], a[0, 0]); end = np.arctan2(b[1, 0], b[0, 0])
    delta = (end-start+np.pi) % (2*np.pi)-np.pi; angle = start+fraction*delta
    h = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    h[:2, 2] = a[:2, 2]+fraction*(b[:2, 2]-a[:2, 2])
    return require_rigid(h)


def rigid_preflight(first, second, pose, policy):
    from .pose_perspective_geometry import blocks, distances
    require_rigid(pose['alignment']['transform'])
    da, db, *_ = distances(first, second, pose['alignment']['transform'])
    counts = [int(np.count_nonzero((blocks(len(d), policy['validation_blocks']) % 2 == 1-side)
                                   & (d <= policy['refinement_distance_px']))) for side, d in enumerate((da, db))]
    eligible = min(counts) >= policy['minimum_validation_samples']
    return {'eligible': eligible, 'reason': 'rigid_fixed_support' if eligible else 'insufficient_fixed_validation',
            'validation_samples': counts, 'optimization_steps': 0, 'homography_fitted': False}


def refine_rigid(first, second, pose, initial, family, base, policy, source_shape, target_shape):
    from .alignment import apply_transform, fit_transforms
    from .pose_perspective_geometry import blocks, distances, validate_fixed
    from .projective import grid, warp_check
    if family != 'rigid' or base['family'] != 'rigid':
        raise ValueError('Only rigid refinement is available in unit-scale mode')
    origin = require_rigid(pose['alignment']['transform']); current = require_rigid(initial).copy()
    da, db, *_ = distances(first, second, origin)
    fixed = [(blocks(len(d), policy['validation_blocks']) % 2 == 1-side)
             & (d <= policy['refinement_distance_px']) for side, d in enumerate((da, db))]
    before = validate_fixed(first, second, origin, fixed, policy); best = before; accepted = 0; rejects = []
    if min(before['samples']) < policy['minimum_validation_samples']:
        return origin, {'accepted_iterations': 0, 'reason': 'insufficient_fixed_validation', 'before': before, 'after': before}
    native_a = np.asarray(first['points_xy_crop']); native_b = np.asarray(second['points_xy_crop'])
    probes = grid(source_shape); diagonal = float(np.hypot(*source_shape))
    bound = min(policy['maximum_probe_displacement_px'], max(policy['minimum_refinement_displacement_px'],
                policy['maximum_probe_displacement_fraction']*diagonal))
    details = {}
    for _ in range(policy['refinement_iterations']):
        da, db, pa, pb, ja, jb = distances(first, second, current)
        ta = blocks(len(da), policy['validation_blocks']) % 2 == 0
        tb = blocks(len(db), policy['validation_blocks']) % 2 == 1
        ia = ta & (da <= policy['refinement_distance_px']) & tb[ja] & tb[ja+1]
        ib = tb & (db <= policy['refinement_distance_px']) & ta[jb] & ta[jb+1]
        x = np.concatenate((native_a[ia], pb[ib]))
        y = np.concatenate((apply_transform(pa[ia], np.linalg.inv(current)), native_b[ib]))
        errors = np.concatenate((da[ia], db[ib]))
        if len(x) < policy['minimum_fit_pairs']:
            rejects.append('insufficient_fit_pairs'); break
        keep = errors <= max(float(np.quantile(errors, policy['robust_keep_fraction'])), 1e-8); x, y = x[keep], y[keep]
        if len(x) < policy['minimum_fit_pairs'] or min(np.linalg.norm(np.ptp(x, axis=0)), np.linalg.norm(np.ptp(y, axis=0))) < policy['minimum_fit_length_px']:
            rejects.append('short_fit_support'); break
        fitted = fit_transforms(x[None], y[None], similarity=False)[0][0]; require_rigid(fitted)
        update = None; reason = 'fixed_validation_not_improved'
        for fraction in policy['refinement_step_fractions']:
            candidate = interpolate_rigid(current, fitted, fraction)
            check = warp_check(candidate, source_shape, target_shape, policy)
            if not check['valid']:
                reason = check['reason']; continue
            if np.max(np.linalg.norm(apply_transform(probes, candidate)-apply_transform(probes, origin), axis=1)) > bound:
                reason = 'whole_crop_displacement'; continue
            value = validate_fixed(first, second, candidate, fixed, policy)
            if value['score'] <= best['score']+policy['validation_score_improvement'] or value['support_fraction'] < policy['validation_support_ratio']*before['support_fraction']:
                continue
            if any(a < b-policy['validation_block_score_slack'] for a, b in zip(value['block_scores'], before['block_scores'])):
                reason = 'validation_block_regressed'; continue
            update = candidate, value; details = {'fit_pairs': len(x), 'step_fraction': fraction}; break
        if update is None:
            rejects.append(reason); break
        current, best = update; accepted += 1
    return require_rigid(current), {'accepted_iterations': accepted, 'before': before, 'after': best, 'fit': details,
        'rejections': rejects, 'displacement_bound_px': bound, 'motion_model': 'SE2', 'projective_fitting_performed': False,
        'maximum_probe_displacement_px': float(np.max(np.linalg.norm(apply_transform(probes, current)-apply_transform(probes, origin), axis=1)))}


def initial_trace(owners, matrices, rms, lengths, normals, config):
    from . import pose_trace
    writer = pose_trace.ACTIVE
    if writer is None or owners is None:
        return
    from .contracts import digest
    q, t = owners; n = len(matrices)
    passed = (rms/lengths <= config['maximum_normalized_residual']) & (normals <= config['maximum_normal_penalty'])
    pair = sorted(writer.gallery[i][1]['entry']['piece_id'] for i in owners)
    writer.save('initial', pair, {'matrix': matrices, 'query': np.full(n, q, np.int32), 'target': np.full(n, t, np.int32),
        'parent_index': np.full(n, -1, np.int32), 'stage_bits': (1+2*passed).astype(np.uint16),
        'proposal_id': np.asarray([digest([q, t, h.tolist()])[:24] for h in matrices], 'S24')},
        {'stages': ['initial_generated', 'initial_geometry_pass'], 'origins': ['initial_subarc'], 'windows_observed_separately': True})
