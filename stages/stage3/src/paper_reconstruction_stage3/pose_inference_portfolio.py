"""Observable parent and start portfolios. No scorer, labels or saved anchors."""
import copy
from collections import Counter
import numpy as np
from .alignment import apply_transform
from .pose_perspective_geometry import blocks, distances


def curve_support(first, second):
    ratios = []; lengths = []
    for curve in (first, second):
        xy = np.asarray(curve['points_xy_crop'], float)
        eig = np.linalg.eigvalsh(np.cov(xy.T)) if len(xy) >= 3 else [0., 0.]
        ratios.append(float(max(eig[0], 0.) / max(eig[-1], 1e-12)))
        lengths.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
    return {'minimum_curve_anisotropy': min(ratios), 'minimum_curve_length_px': min(lengths)}


def parent_portfolio(saved, feature, canonical_map, search, policy):
    """Four slots: source quality, curved support, long support, edge diversity.

    Full native curves give a cheap support proxy. Exact projective fit support
    is checked separately; curve anisotropy is not an identifiability proof.
    """
    ordered = sorted(saved, key=lambda r: (min(d['parent_rank'] for d in r['methods'].values()),
                     -r.get('selection_evidence', {}).get('geometry', {}).get('score', r['pose']['alignment']['geometry_score']), r['record_id']))
    candidates = {}; metrics = {}
    for r in ordered:
        pair = r['pose']['query'], r['pose']['target']
        if pair in candidates: continue
        candidates[pair] = r
        metrics[pair] = {**curve_support(feature(pair[0], 64), feature(pair[1], 64)),
                         'geometry_score': float(r.get('selection_evidence', {}).get('geometry', {}).get('score', r['pose']['alignment']['geometry_score']))}
    remaining = list(candidates); chosen = []; decisions = {}
    threshold = policy['minimum_fit_anisotropy']
    for slot in range(min(search['candidate_parents_per_pair'], len(remaining))):
        if slot == 0:
            pair = remaining[0]; reason = 'source_quality'
        else:
            def score(pair):
                m = metrics[pair]
                curved = m['minimum_curve_anisotropy'] >= threshold
                diversity = min((len(set(pair) - set(p)) for p in chosen), default=2)
                if slot == 1:
                    return curved, m['minimum_curve_anisotropy'], m['minimum_curve_length_px'], m['geometry_score']
                if slot == 2:
                    return curved, m['minimum_curve_length_px'], m['geometry_score'], m['minimum_curve_anisotropy']
                return diversity, curved, m['geometry_score'], m['minimum_curve_length_px']
            pair = max(remaining, key=score)
            reason = ('curved_support', 'long_support', 'candidate_edge_diversity')[min(slot-1, 2)]
        chosen.append(pair); remaining.remove(pair); decisions[pair] = reason
    parents = []
    for pair in chosen:
        row = copy.deepcopy(candidates[pair]); row['_canonical_probes'] = canonical_map(row)
        parents.append(row)
    inventory = [{'query': q, 'target': t, 'parent_index': candidates[q, t]['parent_index'],
                  'record_id': candidates[q, t]['record_id'], **metrics[q, t],
                  'selected': (q, t) in decisions, 'selection_reason': decisions.get((q, t)),
                  'search_slot': chosen.index((q, t)) if (q, t) in chosen else None}
                 for q, t in candidates]
    return parents, {'policy': search['parent_policy'], 'candidate_inventory': inventory,
                     'selected_parent_indices': [p['parent_index'] for p in parents],
                     'reference_anchors_used': False}


from .unit_scale import rigid_preflight as projective_preflight


def start_portfolio(starts, feature, search, policy):
    """At most 32 conditioning checks, then at most eight guarded refinements.

    Round-robin native parent identities prevent a high-scoring, straight
    contour family from consuming the whole preflight or refinement budget.
    """
    groups = {}; seen = {}
    ordered = sorted(starts, key=lambda p: (-p['_broad']['score'], p['proposal_id']))
    for p in ordered:
        xy = p['_canonical_probes'](p['alignment']['transform'])
        directed = p['query'], p['target']
        if any(np.max(np.linalg.norm(xy-other, axis=1)) < search['duplicate_px'] for other in seen.get(directed, [])): continue
        seen.setdefault(directed, []).append(xy)
        groups.setdefault(p['_source_record_id'], []).append(p)
    portfolio = []
    while any(groups.values()) and len(portfolio) < search['projective_preflight_cap']:
        for group in groups.values():
            if group and len(portfolio) < search['projective_preflight_cap']: portfolio.append(group.pop(0))
    eligible = {}; audit = []; rejections = Counter()
    for p in portfolio:
        detail = projective_preflight(feature(p['query'], 192), feature(p['target'], 192), p, policy)
        audit.append({'proposal_id': p['proposal_id'], 'source_record_id': p['_source_record_id'], **detail, 'selected': False})
        if detail['eligible']: eligible.setdefault(p['_source_record_id'], []).append(p)
        else: rejections[detail['reason']] += 1
    selected = []
    while any(eligible.values()) and len(selected) < search['projective_seeds']:
        for group in eligible.values():
            if group and len(selected) < search['projective_seeds']: selected.append(group.pop(0))
    tokens = {p['proposal_id'] for p in selected}
    for item in audit: item['selected'] = item['proposal_id'] in tokens
    return selected, {'policy': search['projective_start_policy'], 'candidates': len(starts),
                      'preflight_checked': len(portfolio), 'selected': len(selected),
                      'preflight_rejections': dict(rejections), 'checks': audit,
                      'reference_anchors_used': False, 'optimization_steps': 0}
