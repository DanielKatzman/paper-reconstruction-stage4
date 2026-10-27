"""Portable observable evidence; geometry grouping never replaces occurrences."""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import io
import json
import shutil

import numpy as np

from .contracts import GEOMETRY, atomic_json, atomic_replace, digest, require

SCHEMA = 'stage3.assembly-evidence.v2'
ALGORITHM = 'stage3.alpha-evidence.v0931'
ROW_FIELDS = ('record_id', 'source_record_id', 'parent_index', 'pose_id', 'pose_index',
              'new_proposal', 'methods', 'content_evidence', 'learned_evidence',
              'selection_evidence', 'tangent_observability', 'visual_evidence')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def seal_file(root, path):
    return {'path': Path(path).relative_to(root).as_posix(), 'size': Path(path).stat().st_size,
            'sha256': file_hash(path)}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def normalized_matrix(value):
    h = np.asarray(value, dtype=np.float64)
    require(h.shape == (3, 3) and np.isfinite(h).all(), 'Invalid native transform')
    require(abs(np.linalg.det(h)) > 1e-12 and abs(h[2, 2]) > 1e-12, 'Singular native transform')
    return h / h[2, 2]


def group_records(gallery, variants):
    """Return exact transform groups and every original support/scoring occurrence.

    The grouping key uses an unrounded normalized matrix. Approximate deduplication
    is confined to the working shortlist and never deletes archive evidence.
    """
    groups = {}
    for variant, trace in sorted(variants.items()):
        for row_index, row in enumerate(trace['records']):
            pose = row['pose']; q, t = pose['query'], pose['target']
            require(type(q) is int and type(t) is int and 0 <= q < len(gallery) and 0 <= t < len(gallery),
                    'Occurrence references an unknown candidate')
            qkey, qa = gallery[q]; tkey, ta = gallery[t]
            a, b = sorted((qa['entry']['piece_id'], ta['entry']['piece_id']))
            require(a != b and row['piece_a'] == a and row['piece_b'] == b, 'Occurrence physical owners disagree')
            qo, to = qa['entry']['observation_key'], ta['entry']['observation_key']
            obs_a, obs_b = (qo, to) if qa['entry']['piece_id'] == a else (to, qo)
            raw = row['canonical_transform_target_b_to_query_a']
            matrix = normalized_matrix(raw)
            directed = normalized_matrix(pose['alignment']['transform'])
            expected = directed if qa['entry']['piece_id'] == a else normalized_matrix(np.linalg.inv(directed))
            require(np.allclose(matrix, expected, atol=1e-6, rtol=1e-8), 'Canonical/query transform direction mismatch')
            identity = digest([a, b, obs_a, obs_b, matrix.tolist()])
            group = groups.setdefault(identity, {'hypothesis_id': identity, 'piece_a': a, 'piece_b': b,
                'observation_a': obs_a, 'observation_b': obs_b, 'transform_b_to_a': matrix.tolist(), 'occurrences': []})
            occurrence = {k: row[k] for k in ROW_FIELDS if k in row}
            occurrence.update(occurrence_id=digest([variant, row_index, row.get('record_id')]),
                variant=variant, row_index=row_index, query_key=qkey, target_key=tkey,
                query_observation=qo, target_observation=to, query_piece=qa['entry']['piece_id'],
                target_piece=ta['entry']['piece_id'], pose=pose,
                original_canonical_transform_b_to_a=raw)
            # Force strict JSON validation here, before opening the output archive.
            json_bytes(occurrence)
            group['occurrences'].append(occurrence)
    return sorted(groups.values(), key=lambda g: g['hypothesis_id'])


def score_occurrence(occurrence):
    """One declared branch/method; correlated branches do not add votes."""
    methods = occurrence.get('methods', {})
    for method in ('mlp_start', 'alpha_fused', 'cnn_budget1', 'reference'):
        value = methods.get(method, {})
        score = value.get('weight')
        if isinstance(score, (int, float)) and np.isfinite(score):
            return float(score), method
    return float(occurrence['pose']['alignment'].get('geometry_score', 0.)), 'geometry_only'


def working_pool(groups, pieces, maximum=24):
    require(type(maximum) is int and 1 <= maximum <= 128, 'Working pool cap must be 1..128 per physical pair')
    pairs = defaultdict(list)
    for group in groups:
        # Prefer the exact frozen control to its later correlated rescoring branches.
        occurrences = [o for o in group['occurrences'] if o['variant'] == 'source_control'] or group['occurrences']
        occurrence = min(occurrences, key=lambda o: (-score_occurrence(o)[0], o['occurrence_id']))
        score, method = score_occurrence(occurrence)
        edge = {k: group[k] for k in ('hypothesis_id', 'piece_a', 'piece_b', 'observation_a', 'observation_b', 'transform_b_to_a')}
        edge.update(weight=score, score_method=method, score_variant=occurrence['variant'],
                    representative_occurrence=occurrence['occurrence_id'],
                    occurrence_ids=[o['occurrence_id'] for o in group['occurrences']], pose=occurrence['pose'])
        pairs[group['piece_a'], group['piece_b']].append(edge)
    selected = []; audit = []
    for (a, b), values in sorted(pairs.items()):
        values.sort(key=lambda e: (-e['weight'], e['hypothesis_id']))
        h, w = pieces[b]['size_hw']; probes = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], float)
        probes = np.column_stack((probes, np.ones(4))); chosen = []; deferred = []; mapped = []
        for edge in values:
            xy = probes @ np.asarray(edge['transform_b_to_a']).T
            require((np.abs(xy[:, 2]) > 1e-12).all(), 'Transform has a pole at native crop probes')
            xy = xy[:, :2] / xy[:, 2:]
            if any(np.max(np.linalg.norm(xy-old, axis=1)) < 2. for old in mapped):
                deferred.append(edge); continue
            chosen.append(edge); mapped.append(xy)
            if len(chosen) == maximum: break
        # Fill unused slots without changing the complete archive.
        chosen += deferred[:maximum-len(chosen)]
        selected.extend(chosen)
        audit.append({'pieces': [a, b], 'available_geometries': len(values), 'working_geometries': len(chosen),
                      'resource_truncated': len(chosen) < len(values)})
    return {'schema_version': 'stage3.working-pool.v1', 'maximum_per_physical_pair': maximum,
            'selection': 'frozen-control score, native-crop transform diversity at 2 px, then score fill',
            'scores_are_calibrated_probabilities': False, 'pairs': audit,
            'edges': sorted(selected, key=lambda e: (-e['weight'], e['hypothesis_id']))}


def export_bundle(output, puzzle_id, entries, gallery, variants, *, provenance, configuration, maximum=24):
    """Sanitize native arrays and export one observation per physical piece.

    Entry inventory is independent of the candidate gallery, retaining isolated
    pieces and pieces for which Stage 2 emitted no candidates.
    """
    from .data import load_contour, check_images
    from paper_reconstruction_contracts import load_bundle
    output = Path(output).resolve()
    require(entries, 'Empty physical-piece inventory')
    by_piece = {}; by_obs = {}
    for entry in entries:
        pu, obs = entry['piece_id'], entry['observation_key']
        require(pu not in by_piece and obs not in by_obs, 'Alpha requires one registered native observation per physical piece')
        require(entry.get('split') not in ('test', 'holdout'), 'Protected evidence is closed')
        by_piece[pu] = entry; by_obs[obs] = entry
    require(len({k for k, _ in gallery}) == len(gallery), 'Duplicate candidate key')
    for key, item in gallery:
        e = item['entry']; require(e['observation_key'] in by_obs and e['piece_id'] in by_piece, 'Gallery inventory mismatch')
        owner = by_piece[e['piece_id']]
        require(all(e.get(k) == owner.get(k) for k in ('observation_key', 'image_path', 'mask_path', 'arrays_path', 'offset_start', 'offset_end')),
                'Candidate native frame differs from physical inventory')
    groups = group_records(gallery, variants)
    identity = digest({'puzzle_id': puzzle_id, 'entries': entries, 'gallery': gallery, 'variants': variants,
                       'provenance': provenance, 'configuration': configuration, 'maximum': maximum, 'algorithm': ALGORITHM})
    if (output/'bundle.json').exists():
        old = load_bundle(output)
        require(old.manifest['identity'] == identity, 'Completed alpha bundle belongs to different inputs/configuration')
        return old.manifest
    marker = output/'export_request.json'
    if output.exists() and any(output.iterdir()):
        from .contracts import read_json
        require(marker.exists() and read_json(marker) == {'identity': identity}, 'Nonempty bundle output has a different/unknown export identity')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, {'identity': identity})
    asset_root = output/'assets'; asset_root.mkdir(exist_ok=True)
    assets = {}; pieces = {}
    def adopt(path):
        source = Path(path); checksum = file_hash(source)
        suffix = source.suffix.lower()
        require(suffix in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npz'), 'Unsupported native asset extension')
        target = asset_root/(checksum+suffix)
        if not target.exists():
            temporary = target.with_suffix(suffix+'.tmp'); shutil.copyfile(source, temporary); atomic_replace(temporary, target)
        require(file_hash(target) == checksum, 'Portable asset copy changed')
        receipt = seal_file(output, target); assets[receipt['path']] = receipt
        return receipt['path']
    for pu, entry in sorted(by_piece.items()):
        contour = load_contour({**entry, 'candidates': entry.get('candidates', [])}); check_images(entry, contour)
        arrays = {k: contour.arrays[k] for k in ('points_xy', *GEOMETRY, 'role_probabilities', 'transition_probabilities')}
        buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays); data = buffer.getvalue()
        target = asset_root/(hashlib.sha256(data).hexdigest()+'.npz')
        if not target.exists():
            temporary = target.with_suffix('.tmp'); temporary.write_bytes(data); atomic_replace(temporary, target)
        rc = seal_file(output, target); assets[rc['path']] = rc
        pieces[pu] = {'piece_id': pu, 'observation_key': entry['observation_key'], 'size_hw': entry['size_hw'],
                      'image': adopt(entry['image_path']), 'mask': adopt(entry['mask_path']), 'contour': rc['path']}
    candidates = [{'candidate_key': k, 'piece_id': p['entry']['piece_id'], 'observation_key': p['entry']['observation_key'],
                   'candidate': p['candidate']} for k, p in gallery]
    working = working_pool(groups, pieces, maximum)
    atomic_json(output/'working_pool.json', working)
    archive = output/'hypotheses.jsonl.gz'; temporary = archive.with_suffix('.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            for group in groups: zipped.write(json_bytes(group)+b'\n')
    atomic_replace(temporary, archive)
    manifest = {'schema_version': SCHEMA, 'algorithm': ALGORITHM, 'identity': identity, 'puzzle_id': puzzle_id,
        'pieces': pieces, 'candidates': candidates, 'assets': assets,
        'archive': seal_file(output, archive), 'working_pool': seal_file(output, output/'working_pool.json'),
        'counts': {'pieces': len(pieces), 'candidates': len(candidates), 'geometries': len(groups),
                   'occurrences': sum(len(g['occurrences']) for g in groups), 'working_edges': len(working['edges'])},
        'coordinate_frame': 'Native cropped image pixels; x right, y down; homogeneous 3x3 maps piece_b into piece_a.',
        'occurrence_contract': 'Every variant/row retains native candidate keys, complete pose/support, original matrix, method metadata and observable evidence; no score summation across occurrences.',
        'provenance': provenance, 'configuration': configuration,
        'correction_scorer': {'enabled': False, 'reason': 'v0930_no_calibration_operating_point'},
        'training_performed': False, 'optimizer_steps': 0, 'validation_truth_opened': False, 'protected_truth_opened': False,
        'scores_are_calibrated_probabilities': False, 'global_assembly_performed': False,
        'capacity': {'demonstrated_reliable_piece_capacity': None, 'baseline_target_pieces': 20, 'milestone_pieces': 30}}
    manifest['manifest_sha256'] = digest(manifest)
    atomic_json(output/'bundle.json', manifest)
    load_bundle(output)
    return manifest
