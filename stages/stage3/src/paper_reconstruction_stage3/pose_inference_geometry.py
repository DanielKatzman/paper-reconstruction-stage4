"""Bounded observable proposal search shared by inference and its smoke check."""
import copy
import time
from collections import Counter
import numpy as np
from .alignment import apply_transform, collision_fraction
from .contracts import digest
from .pose_boundary_evidence import evidence
from .pose_overlap_geometry import sliding_fits
from .pose_perspective_geometry import refine
from .pose_retention_selection import portfolio
from .projective import warp_check
from .pose_inference_contracts import STAGES, ORIGINS
from .pose_trace import overlap_trace


def generate(parents, feature, mask, mask_samples, base, correction, policy, search):
    """Parents/curves/masks only. Return guarded pools and sealed numeric traces.

    A directed parent is an existing candidate pair, never a saved seam. All
    bounded prefilter candidates receive guards; survivor count cannot stop it.
    No image model is evaluated until the <=128 branch shortlists are fixed.
    """
    started = time.perf_counter()
    trace = []
    lookup = {}
    screened = []
    failures = Counter()
    searches = []
    def remember(p, origin, parent):
        h = np.asarray(p['alignment']['transform'], float)
        token = digest([p['query'], p['target'], h.tolist()])[:24]
        if token not in lookup:
            lookup[token] = len(trace)
            trace.append({'matrix': h, 'query': p['query'], 'target': p['target'],
                          'parent_index': parent['parent_index'], 'stage_bits': 1,
                          'origin': ORIGINS.index(origin), 'proposal_id': token,
                          'observable_score': -1.})
        p['proposal_id'] = token
        p['_trace_index'] = lookup[token]
        p['_source_record_id'] = parent['record_id']
        return p
    def mark(p, stage):
        trace[p['_trace_index']]['stage_bits'] |= 1 << STAGES.index(stage)
    for parent in parents[:search['candidate_parents_per_pair']]:
        q, t = parent['pose']['query'], parent['pose']['target']
        first, second = feature(q, 512), feature(t, 512)
        coarse, fine, stats = sliding_fits(first, second, base, search)
        overlap_trace(q, t, 'generation', coarse, fine, base, parent['parent_index'])
        searches.append({'parent_index': parent['parent_index'], 'query': q, 'target': t, **stats})
        good = []
        for origin, values in (('sliding_coarse', coarse), ('sliding_fine', fine)):
            for p in values:
                p.update(query=q, target=t)
                remember(p, origin, parent)
                fit = p['alignment']
                if fit['normalized_residual'] <= base['maximum_normalized_residual'] and fit['inward_normal_penalty'] <= base['maximum_normal_penalty']:
                    mark(p, 'geometry_pass')
                    p['_selection_origin'] = origin
                    good.append(p)
        values, _ = portfolio(good, search['geometry_screen_per_parent'], second['points_xy_crop'][::64],
                              score=lambda p: p['alignment']['geometry_score'],
                              secondary=lambda p: p['alignment']['fitted_overlap_length_query_px'], distance=search['duplicate_px'])
        screened.extend(values)
    generation_seconds = time.perf_counter() - started
    guard_start = time.perf_counter()
    survivors = []
    for p in screened:
        q, t = p['query'], p['target']
        mark(p, 'screened'); mark(p, 'warp_checked')
        h = p['alignment']['transform']
        if not warp_check(h, mask(t).shape, mask(q).shape, policy)['valid']:
            failures['warp'] += 1; continue
        mark(p, 'warp_pass'); mark(p, 'collision_checked')
        collision = collision_fraction(mask(q), mask(t), h, base['collision_grid'], (mask_samples(q), mask_samples(t)))
        if collision > base['maximum_pair_collision']:
            failures['collision'] += 1; continue
        mark(p, 'collision_pass')
        p['alignment']['mask_collision_proxy'] = collision
        p['_broad'] = evidence(feature(q, 128), feature(t, 128), h, p, correction)
        trace[p['_trace_index']]['observable_score'] = p['_broad']['score']
        if not p['_broad']['credible']:
            failures['broader_support'] += 1; continue
        mark(p, 'broader_support_pass')
        survivors.append(p)
    # Deduplicate in canonical piece coordinates supplied by the caller. Curves
    # from different candidate IDs are not directly comparable native probes.
    def choose(values, cap):
        chosen = []; mapped = []
        for p in sorted(values, key=lambda p: (-p['_broad']['score'], -p['alignment']['geometry_score'], p['proposal_id'])):
            xy = p['_canonical_probes'](p['alignment']['transform'])
            if any(np.max(np.linalg.norm(xy - other, axis=1)) < search['duplicate_px'] for other in mapped):
                continue
            chosen.append(p); mapped.append(xy)
            if len(chosen) == cap: break
        return chosen
    parent_lookup = {p['record_id']: p for p in parents}
    for p in survivors:
        p['_canonical_probes'] = parent_lookup[p['_source_record_id']]['_canonical_probes']
    screened_pool = choose(survivors, search['broad_screen_cap'])
    sliding = choose(screened_pool, search['novel_pool_cap'])
    for p in sliding: mark(p, 'sliding_pool')
    # Include native observable starts as well as new overlap fits. They have
    # identical full-crop/fixed-block guards, and no reference fit is injected.
    starts = list(screened_pool)
    for parent in parents:
        p = copy.deepcopy(parent['pose']); p['_source_record_id'] = parent['record_id']
        p['_canonical_probes'] = parent['_canonical_probes']
        q, t = p['query'], p['target']
        p['_broad'] = evidence(feature(q, 128), feature(t, 128), p['alignment']['transform'], p, correction)
        p['proposal_id'] = digest([q, t, p['alignment']['transform']])[:24]
        starts.append(p)
    from .pose_inference_portfolio import start_portfolio
    seeds, start_audit = start_portfolio(starts, feature, search, policy)
    projective = []; attempts = []
    for source in seeds:
        p = copy.deepcopy(source); q, t = p['query'], p['target']
        h, details = refine(feature(q, 192), feature(t, 192), p, p['alignment']['transform'], 'rigid', base, policy, mask(t).shape, mask(q).shape)
        attempt = {'source_proposal_id': p['proposal_id'], 'source_record_id': p['_source_record_id'], **details}
        attempts.append(attempt)
        if not details['accepted_iterations']: continue
        parent = parent_lookup[p['_source_record_id']]
        p['inference_source_alignment'] = copy.deepcopy(p['alignment'])
        p['alignment'].update(transform=np.asarray(h).tolist(), family='rigid')
        p['inference_rigid_refinement'] = details
        remember(p, 'rigid_refinement', parent)
        mark(p, 'geometry_pass'); mark(p, 'screened'); mark(p, 'warp_checked')
        if not warp_check(h, mask(t).shape, mask(q).shape, policy)['valid']:
            failures['projective_warp'] += 1; continue
        mark(p, 'warp_pass'); mark(p, 'collision_checked')
        collision = collision_fraction(mask(q), mask(t), h, base['collision_grid'], (mask_samples(q), mask_samples(t)))
        if collision > base['maximum_pair_collision']:
            failures['projective_collision'] += 1; continue
        mark(p, 'collision_pass')
        p['alignment']['mask_collision_proxy'] = collision
        p['_broad'] = evidence(feature(q, 128), feature(t, 128), h, p, correction)
        trace[p['_trace_index']]['observable_score'] = p['_broad']['score']
        if p['_broad']['credible']:
            mark(p, 'broader_support_pass'); projective.append(p)
        else: failures['projective_broader_support'] += 1
    combined = choose(screened_pool + projective, search['novel_pool_cap'])
    for p in combined: mark(p, 'rigid_refined_pool')
    arrays = {
        'matrix': np.asarray([r['matrix'] for r in trace], dtype=np.float64).reshape(-1, 3, 3),
        **{k: np.asarray([r[k] for r in trace], dtype=dtype) for k, dtype in
           [('query', np.int32), ('target', np.int32), ('parent_index', np.int32), ('stage_bits', np.uint16),
            ('origin', np.uint8), ('proposal_id', 'S24'), ('observable_score', np.float64)]}}
    counts = {s: int(np.count_nonzero(arrays['stage_bits'] & (1 << i))) for i, s in enumerate(STAGES)}
    return {'sliding_joint': sliding, 'projective_joint': combined}, arrays, {
        'searches': searches, 'stage_counts': counts, 'rejections': dict(failures),
        'projective_start_selection': start_audit,
        'projective_attempts': attempts, 'full_prefilter_checked': True,
        'generation_seconds': generation_seconds, 'guards_and_refinement_seconds': time.perf_counter() - guard_start,
        'reference_anchors_used': False, 'learned_model_used_for_generation': False}
