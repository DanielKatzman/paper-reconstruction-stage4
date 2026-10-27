"""Observable, bounded archive replay. No producer, model or evaluation imports."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from paper_reconstruction_contracts import load_bundle, map_points

POLICY = {
    'name': 'support-pose-strata.v1',
    'maximum_per_physical_pair': 24,
    'score_anchors': 6,
    'relative_score_margin': .2,
    'support_fraction_bins': 4,
    'support_span_bin_px': 128.,
    'translation_cell_diagonal_fraction': .05,
    'rotation_cell_degrees': 15.,
    'log_scale_cell': .1,
    'distance_saturation_diagonal_fraction': .15,
    'slot_cycle': ['support', 'origin', 'pose'],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value)+b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def disjoint_output(output, protected):
    output = Path(output).resolve()
    for item in protected:
        item = Path(item).resolve()
        if output.is_relative_to(item) or item.is_relative_to(output):
            raise ValueError('Output overlaps a protected input: '+str(item))
    return output


def bundle_files(data):
    paths = [data.root/'bundle.json', data.root/data.manifest['archive']['path'],
             data.root/data.manifest['working_pool']['path']]
    paths.extend(data.root/r['path'] for r in data.manifest['assets'].values())
    return {str(p.resolve()): file_hash(p) for p in paths}


def verify_files(receipts):
    for name, checksum in receipts.items():
        if file_hash(name) != checksum:
            raise ValueError('Protected source changed: '+name)


def occurrence_score(occurrence):
    for method in ('mlp_start', 'alpha_fused', 'cnn_budget1', 'reference'):
        value = occurrence.get('methods', {}).get(method, {}).get('weight')
        if isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
            return float(value), method
    return float(occurrence['pose']['alignment'].get('geometry_score', 0.)), 'geometry_only'


def archive_edge(group):
    candidates = [o for o in group['occurrences'] if o['variant'] == 'source_control'] or group['occurrences']
    representative = min(candidates, key=lambda o: (-occurrence_score(o)[0], o['occurrence_id']))
    score, method = occurrence_score(representative)
    edge = {k: deepcopy(group[k]) for k in ('hypothesis_id', 'piece_a', 'piece_b',
            'observation_a', 'observation_b', 'transform_b_to_a')}
    edge.update(weight=score, score_method=method, score_variant=representative['variant'],
                representative_occurrence=representative['occurrence_id'],
                occurrence_ids=[o['occurrence_id'] for o in group['occurrences']],
                pose=deepcopy(representative['pose']))
    return edge


def strata(group, pieces, policy):
    """Support bins are coverage proxies, never replacements for exact supports."""
    supports, origins = set(), set()
    for occurrence in group['occurrences']:
        pose = occurrence['pose']; fit = pose['alignment']
        sides = []
        for side, fraction, span in (('query', 'query_arc_fraction', 'span_a'),
                                     ('target', 'target_arc_fraction', 'span_b')):
            interval = fit.get(fraction, [0., 1.])
            bins = tuple(int(np.floor(float(x)*policy['support_fraction_bins'])) for x in interval)
            window = pose.get(span)
            span_bins = None if window is None else tuple(int(np.floor(float(window[k])/policy['support_span_bin_px'])) for k in ('start_px', 'end_px'))
            sides.append((occurrence[side+'_piece'], occurrence[side+'_key'], bins, span_bins))
        supports.add(tuple(sorted(sides)))
        origins.add((occurrence['variant'], fit.get('family', 'unknown')))
    a, b = group['piece_a'], group['piece_b']
    height, width = pieces[b]['size_hw']
    probes = np.array([(x, y) for y in (0., (height-1)/2, height-1) for x in (0., (width-1)/2, width-1)])
    mapped = map_points(probes, group['transform_b_to_a'])
    diagonal = max(float(np.linalg.norm(pieces[a]['size_hw'])), 1.)
    center = np.array([(width-1)/2, (height-1)/2])
    basis = map_points([center, center+[1., 0.]], group['transform_b_to_a'])
    vector = basis[1]-basis[0]
    angle = np.arctan2(vector[1], vector[0])
    pose_cell = (*np.floor(basis[0]/(diagonal*policy['translation_cell_diagonal_fraction'])).astype(int).tolist(),
                 int(np.floor(angle/np.deg2rad(policy['rotation_cell_degrees']))),
                 int(np.floor(np.log(max(float(np.linalg.norm(vector)), 1e-12))/policy['log_scale_cell'])))
    return {'support': supports, 'origin': origins, 'pose': {pose_cell}, 'mapped': mapped/diagonal}


def select_pool(groups, pieces, policy=None, source_cap=24):
    policy = deepcopy(POLICY if policy is None else policy)
    maximum = min(policy['maximum_per_physical_pair'], source_cap)
    if type(maximum) is not int or not 1 <= maximum <= 24:
        raise ValueError('Replay must retain a bounded pool of at most 24 geometries per pair')
    if policy['relative_score_margin'] <= 0 or policy['slot_cycle'] != ['support', 'origin', 'pose']:
        raise ValueError('Unsupported replay policy')
    pairs = defaultdict(list)
    for group in groups:
        pairs[group['piece_a'], group['piece_b']].append((archive_edge(group), strata(group, pieces, policy)))
    selected, audits = [], []
    for pair, rows in sorted(pairs.items()):
        rows.sort(key=lambda r: (-r[0]['weight'], r[0]['hypothesis_id']))
        floor = rows[0][0]['weight']-policy['relative_score_margin']
        eligible = [r for r in rows if r[0]['weight'] >= floor]
        counts = {k: Counter() for k in ('support', 'origin', 'pose')}
        mapped, chosen, decisions = [], [], []

        def add(row, reason):
            edge, features = row
            chosen.append(edge)
            decisions.append({'geometry_id': edge['hypothesis_id'], 'slot': len(chosen), 'reason': reason,
                              'source_weight': edge['weight']})
            for key in counts:
                counts[key].update(features[key])
            mapped.append(features['mapped'])
            eligible.remove(row)

        for row in list(eligible[:min(maximum, policy['score_anchors'])]):
            add(row, 'source_score_anchor')
        turn = 0
        while eligible and len(chosen) < maximum:
            mode = policy['slot_cycle'][turn % len(policy['slot_cycle'])]

            def priority(row):
                edge, features = row
                coverage = {key: sum(1./(1+counts[key][token]) for token in sorted(features[key], key=repr))/max(1, len(features[key])) for key in counts}
                distance = min((float(np.sqrt(np.mean(np.sum((features['mapped']-xy)**2, axis=1)))) for xy in mapped), default=1.)
                distance = min(distance, policy['distance_saturation_diagonal_fraction'])
                quality = np.clip((edge['weight']-floor)/policy['relative_score_margin'], 0., 1.)
                first = distance*(.1+quality) if mode == 'pose' else coverage[mode]
                return (first, coverage['support'], coverage['origin'], distance, edge['weight'])

            # Stable ID tie break comes from the original deterministic ordering.
            add(max(eligible, key=priority), mode+'_coverage')
            turn += 1
        selected.extend(chosen)
        audits.append({'pieces': list(pair), 'available_geometries': len(rows), 'working_geometries': len(chosen),
                       'resource_truncated': len(chosen) < len(rows), 'score_floor': floor,
                       'eligible_geometries': sum(r[0]['weight'] >= floor for r in rows),
                       'covered_strata': {k: len(v) for k, v in counts.items()}, 'decisions': decisions})
    pool = {'schema_version': 'stage3.working-pool.v1', 'maximum_per_physical_pair': maximum,
            'selection': policy['name']+'; source-score anchors then support/origin/pose coverage; original scores and all occurrence IDs retained',
            'scores_are_calibrated_probabilities': False,
            'pairs': [{k: row[k] for k in ('pieces', 'available_geometries', 'working_geometries', 'resource_truncated')} for row in audits],
            'edges': sorted(selected, key=lambda e: (-e['weight'], e['hypothesis_id']))}
    return pool, {'policy': policy, 'pairs': audits, 'evaluation_truth_opened': False,
                  'scores_changed': False, 'archive_occurrences_changed': False}


def publish_replay(source, output, implementation, policy=None):
    source = load_bundle(source, full=True)
    receipts = bundle_files(source)
    output = disjoint_output(output, [source.root])
    if output.exists():
        raise ValueError('Replay output is immutable; choose a new directory')
    groups = list(source.hypotheses())
    original = {edge['hypothesis_id']: edge for edge in source.pool['edges']}
    for group in groups:
        if group['hypothesis_id'] in original and archive_edge(group) != original[group['hypothesis_id']]:
            raise ValueError('Archived representative/score semantics differ from the source working pool')
    pool, audit = select_pool(groups, source.manifest['pieces'], policy, source.pool['maximum_per_physical_pair'])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name+'.building_'+uuid4().hex[:12])
    staging.mkdir()  # Inherit workspace ACLs, including when invoked in the sandbox.
    for receipt in [source.manifest['archive'], *source.manifest['assets'].values()]:
        target = staging/receipt['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.root/receipt['path'], target)
        if target.stat().st_size != receipt['size'] or file_hash(target) != receipt['sha256']:
            raise ValueError('Byte-preserving archive/asset copy failed')
    pool_path = staging/source.manifest['working_pool']['path']
    atomic_json(pool_path, pool)
    manifest = deepcopy(source.manifest)
    manifest['algorithm'] = 'stage3.archive-pool-replay.v1'
    manifest['configuration']['pool_replay'] = audit['policy']
    manifest['provenance'] = {'kind': 'observable_saved_archive_pool_replay',
        'source_manifest_sha256': source.manifest['manifest_sha256'], 'source_bundle_identity': source.manifest['identity'],
        'source_algorithm': source.manifest['algorithm'], 'source_provenance': source.manifest['provenance'],
        'implementation': implementation, 'pool_policy_changed': True, 'scores_changed': False,
        'pose_search_repeated': False, 'training_performed': False, 'optimizer_steps': 0, 'validation_truth_opened': False}
    manifest['working_pool'] = {'path': source.manifest['working_pool']['path'], 'size': pool_path.stat().st_size, 'sha256': file_hash(pool_path)}
    manifest['counts']['working_edges'] = len(pool['edges'])
    manifest['identity'] = digest([source.manifest['manifest_sha256'], audit['policy'], implementation])
    manifest['manifest_sha256'] = digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    atomic_json(staging/'bundle.json', manifest)
    replay = load_bundle(staging, full=True)
    if list(replay.hypotheses()) != groups:
        raise ValueError('Replay changed archived geometry/support content')
    verify_files(receipts)
    staging.rename(output)
    audit.update(source_manifest_sha256=source.manifest['manifest_sha256'],
                 replay_manifest_sha256=manifest['manifest_sha256'], sources_verified_unchanged=True,
                 source_working_edges=len(source.edges), replay_working_edges=len(pool['edges']),
                 archive_receipt_unchanged=manifest['archive'] == source.manifest['archive'],
                 asset_receipts_unchanged=manifest['assets'] == source.manifest['assets'])
    return audit
