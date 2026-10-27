"""Observable-only inference on an arbitrary frozen Stage 2 production handoff.

This is an explicitly versioned composition of frozen components, not a claim
to replay every historical experiment coordinator on an unseen puzzle.
"""
from collections import OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
import copy
import os
import time

import numpy as np

from .alpha_bundle import ALGORITHM, export_bundle, file_hash, normalized_matrix
from .alpha_package import load_package
from .contracts import atomic_json, digest, read_json, require
from .unit_scale import configuration, selection_rows, require_rigid
from .pose_trace import overlap_trace, bootstrap_stage, generated_trace


def production_inputs(stage1, stage2):
    from .alpha_inputs import Files, production_header, production_piece
    config = {'stage1': str(Path(stage1).resolve()), 'stage2': str(Path(stage2).resolve()),
              'geometry_only': False, 'allow_legacy_stage2': False}
    files = Files(); warnings = []
    manifest = production_header(config, files, warnings); entries = []
    for piece in manifest['pieces']: entries.extend(production_piece(config, manifest, piece, files))
    require(len(entries) == manifest['piece_count'] and sum(len(e['candidates']) for e in entries) == manifest['candidate_count'],
            'Production inventory totals changed')
    require(len({e['piece_id'] for e in entries}) == len(entries), 'Multiple captures per physical piece need registration before alpha inference')
    gallery = []
    for entry in sorted(entries, key=lambda e: (e['piece_id'], e['observation_key'])):
        for candidate in sorted(entry['candidates'], key=lambda c: c['candidate_id']):
            gallery.append((entry['observation_key']+'/'+candidate['candidate_id'],
                            {'entry': {k: v for k, v in entry.items() if k != 'candidates'}, 'candidate': candidate}))
    return entries, gallery, files.receipt()


from .bounded_feature_cache import FeatureBank


class FrozenRankers:
    def __init__(self, package_root, manifest, device):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from .model_catalogue import ProfileConfig
        from .reranker_models import RerankerModel
        from .model_identity import weights_digest
        from paper_reconstruction_contracts import verify_receipt
        require(device in ('cpu', 'cuda'), 'Unsupported inference device')
        require(device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable; select --device cpu explicitly to run on CPU')
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch = torch; self.device = device; self.models = []; self.parity = []
        for model in manifest['rankers']:
            net = RerankerModel(model['name'], ProfileConfig(), {'mode': 'full', 'bundle': 'G+P+A',
                'numerics_protocol': 'stage3.reranker-fp32-affinity.v070'})
            payload = torch.load(verify_receipt(package_root, model['checkpoint']), map_location='cpu', weights_only=True)
            require(payload['context'] == model['context'], 'Frozen ranker context mismatch')
            net.load_state_dict(payload['model'], strict=True)
            require(weights_digest(net) == model['weights_sha256'], 'Frozen ranker tensor identity mismatch')
            self.models.append((model['name'], net.eval().requires_grad_(False).to(device)))

    def score(self, bank, pairs, progress=print):
        from .model_inputs import observable_pair_evidence
        torch = self.torch; result = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 16):
                block = pairs[start:start+16]; a = [bank.features(q) for q, _ in block]; b = [bank.features(t) for _, t in block]
                # Evidence uses the same rigid partial fit as the frozen pair models.
                ev = np.stack([observable_pair_evidence({k: v for k, v in x.items() if k != 'strip'},
                    {k: v for k, v in y.items() if k != 'strip'})[0] for x, y in zip(a, b)])
                rev = np.stack([observable_pair_evidence({k: v for k, v in y.items() if k != 'strip'},
                    {k: v for k, v in x.items() if k != 'strip'})[0] for x, y in zip(a, b)])
                def stack(rows, device):
                    return {k: torch.from_numpy(np.stack([r[k] for r in rows])).to(device) for k in ('sequence', 'strip')}
                aa, bb = stack(a, self.device), stack(b, self.device)
                ee, rr = torch.from_numpy(ev).to(self.device), torch.from_numpy(rev).to(self.device)
                scores = {}
                for name, net in self.models:
                    logits = net(aa, bb, ee, rr)['compatibility']
                    require(bool(torch.isfinite(logits).all()), 'Nonfinite frozen ranker output')
                    if start == 0 and self.device == 'cuda':
                        cpu = copy.deepcopy(net).cpu()
                        reference = cpu(stack(a, 'cpu'), stack(b, 'cpu'), torch.from_numpy(ev), torch.from_numpy(rev))['compatibility']
                        error = float(torch.max(torch.abs(logits.cpu()-reference)))
                        require(torch.allclose(logits.cpu(), reference, atol=2e-4, rtol=2e-4), 'CPU/CUDA frozen ranker parity failed')
                        self.parity.append({'model': name, 'maximum_logit_difference': error, 'passed': True}); del cpu
                    scores[name] = torch.sigmoid(logits).cpu().tolist()
                result.extend({'query': q, 'target': t, 'scores': {name: values[i] for name, values in scores.items()}}
                              for i, (q, t) in enumerate(block))
                if start == 0 or (start//16) % 20 == 0 or start+16 >= len(pairs):
                    progress(f'Frozen pair scoring: {min(start+16, len(pairs))}/{len(pairs)}; zero optimizer steps')
        return result


def retrieve_parents(inputs, recipe):
    from .baselines import descriptor
    keys = [k for k, _ in inputs.gallery]; rows = []; geometry = {}
    if not keys: return rows, geometry
    a = np.stack([descriptor(inputs.features(i, {'samples': 512}), probabilities=True) for i in range(len(keys))])
    b = np.stack([descriptor(inputs.features(i, {'samples': 512}), reverse=True, probabilities=True) for i in range(len(keys))])
    pieces = [p['entry']['piece_id'] for _, p in inputs.gallery]
    for q in range(len(keys)):
        distances = np.sum((b-a[q])**2, axis=1)
        pool = sorted((t for t in range(len(keys)) if pieces[q] != pieces[t]), key=lambda t: (float(distances[t]), keys[t]))[:recipe['retrieval_pool_per_query']]
        rows.extend((q, t) for t in pool)
        geometry.update({(q, t): rank for rank, t in enumerate(pool, 1)})
    return rows, geometry


def fuse_parents(scores, geometry, gallery, recipe):
    by_query = defaultdict(list)
    for row in scores: by_query[row['query']].append(row)
    parents = []; by_pair = defaultdict(list)
    for q, values in sorted(by_query.items()):
        ranks = {}
        for model in ('pair_mlp_small', 'paired_strip_cnn_small'):
            ranks[model] = {r['target']: rank for rank, r in enumerate(sorted(values, key=lambda r: (-r['scores'][model], gallery[r['target']][0])), 1)}
        ranked = []
        for row in values:
            t = row['target']; score = 60*(.5/(60+geometry[q, t]) + .25/(60+ranks['pair_mlp_small'][t]) + .25/(60+ranks['paired_strip_cnn_small'][t]))
            ranked.append({'query': q, 'target': t, 'retrieval_score': score, 'ranker_scores': row['scores'], 'windows': [[None, None]]})
        ranked.sort(key=lambda p: (-p['retrieval_score'], gallery[p['target']][0]))
        for rank, parent in enumerate(ranked[:recipe['parents_per_query']], 1):
            parent['parent_rank'] = rank
            pair = tuple(sorted(gallery[i][1]['entry']['piece_id'] for i in (q, parent['target'])))
            by_pair[pair].append(parent)
    for pair, values in sorted(by_pair.items()):
        values.sort(key=lambda p: (-p['retrieval_score'], gallery[p['query']][0], gallery[p['target']][0]))
        parents.extend(values[:recipe['parents_per_physical_pair']])
    return parents, {'scored_directed_pairs': len(scores), 'queries': len(by_query), 'retained_parents': len(parents),
        'physical_pairs_with_parents': len(by_pair), 'resource_bounds': {k: recipe[k] for k in
        ('retrieval_pool_per_query', 'parents_per_query', 'parents_per_physical_pair')},
        'search_domain_exhausted_claimed': False, 'reference_anchors_used': False}


def attach_windows(parent, inputs, spacing, uncertainty, recipe):
    from .windows import observable_windows
    q, t = parent['query'], parent['target']
    if max(spacing[q], spacing[t]) <= recipe['local_spacing_px'] and max(uncertainty[q], uncertainty[t]) < recipe['local_uncertainty']: return
    windows = []
    for owner in (q, t):
        _, item = inputs.gallery[owner]
        windows.append(observable_windows(inputs.contours[item['entry']['observation_key']], item['candidate'], length_px=256.))
    combinations = [(x, y) for x in windows[0] for y in windows[1]]
    # Uniform deterministic support coverage; declared bounded approximation.
    if combinations:
        grid = np.unique(np.linspace(0, len(combinations)-1, min(len(combinations), recipe['local_window_budget'])).round().astype(int))
        parent['windows'].extend([[combinations[i][0]['window_id'], combinations[i][1]['window_id']] for i in grid])


def clean_pose(value):
    if isinstance(value, dict): return {k: clean_pose(v) for k, v in value.items() if not k.startswith('_')}
    if isinstance(value, (list, tuple)): return [clean_pose(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def sliding_starts(parent, feature, inputs, base, correction, policy):
    """Find observable admissible starts even when the old subarc grid misses."""
    from .pose_overlap_geometry import sliding_fits
    from .pose_inference_contracts import SEARCH
    from .pose_boundary_evidence import evidence
    from .alignment import collision_fraction
    from .pose_retention_selection import portfolio
    from .projective import warp_check
    q, t = parent['query'], parent['target']
    first, second = feature(q, 512), feature(t, 512)
    coarse, fine, audit = sliding_fits(first, second, base, SEARCH)
    overlap_trace(q, t, 'bootstrap', coarse, fine, base)
    values = [p for p in coarse+fine if p['alignment']['normalized_residual'] <= base['maximum_normalized_residual']
              and p['alignment']['inward_normal_penalty'] <= base['maximum_normal_penalty']]
    values, _ = portfolio(values, SEARCH['geometry_screen_per_parent'], second['points_xy_crop'][::64],
        score=lambda p: p['alignment']['geometry_score'], secondary=lambda p: p['alignment']['fitted_overlap_length_query_px'],
        distance=SEARCH['duplicate_px'])
    bootstrap_stage(q, t, values, 'screened')
    starts = []
    for p in values:
        p.update(query=q, target=t); h = p['alignment']['transform']
        bootstrap_stage(q, t, [p], 'warp_checked')
        if not warp_check(h, inputs.mask(t).shape, inputs.mask(q).shape, policy)['valid']: continue
        bootstrap_stage(q, t, [p], 'warp_pass')
        bootstrap_stage(q, t, [p], 'collision_checked')
        collision = collision_fraction(inputs.mask(q), inputs.mask(t), h, base['collision_grid'], (inputs.mask_samples(q), inputs.mask_samples(t)))
        if collision > base['maximum_pair_collision']: continue
        bootstrap_stage(q, t, [p], 'collision_pass')
        p['alignment']['mask_collision_proxy'] = collision
        p['_broad'] = evidence(feature(q, 128), feature(t, 128), h, p, correction)
        bootstrap_stage(q, t, [p], 'broader_support_checked')
        if not p['_broad']['credible']: continue
        bootstrap_stage(q, t, [p], 'broader_support_pass')
        starts.append(p)
        if len(starts) == base['alternatives_per_parent']: break
    return starts, {**audit, 'admissible_bootstrap_starts': len(starts), 'bounded_screen': len(values)}


def infer(package, stage1, stage2, output, *, device='cuda', puzzle_id='production', progress=print):
    from .alpha_inputs import lock, unchanged
    from .baselines import BaselineConfig, ObservableInputs
    from .observable_risk import observed_risk
    from .pose_geometry import fit_parent
    from .pose_correction_geometry import curve
    from .pose_boundary_evidence import evidence
    from .pose_visual_evidence import visual_profile
    from .visual_preparation import VisualPreparationCache
    from .exact_acceleration import VisualCache
    from .pose_inference_geometry import generate
    from .pose_inference_contracts import SEARCH
    from .alignment import apply_transform
    from paper_reconstruction_contracts import load_bundle
    started = time.perf_counter(); root, package_manifest = load_package(package, full=False)
    progress('Authenticate Stage 1/2 production handoff; no labels or training')
    entries, gallery, receipts = production_inputs(stage1, stage2)
    config, receipts = configuration(stage1, package_manifest['configuration'], receipts)
    recipe = config['recipe']; base = config['pose_configuration']
    require(len(gallery) <= recipe['maximum_candidates'], 'Alpha candidate resource limit exceeded; no silent truncation. Split the capture batch or define a new measured resource policy.')
    output = Path(output).resolve()
    protected = [Path(stage1).resolve(), Path(stage2).resolve(), root]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected), 'Inference output overlaps immutable inputs/package')
    signature = digest([ALGORITHM, package_manifest['manifest_sha256'], package_manifest['implementation_sha256'], receipts, gallery, puzzle_id, device])
    if (output/'bundle.json').exists():
        existing = load_bundle(output, full=True)
        require(existing.manifest['provenance']['inference_signature'] == signature, 'Existing inference output has different inputs')
        return {**existing.summary(), 'reused': True}
    work = output.with_name(output.name+'.work'); work.mkdir(parents=True, exist_ok=True)
    with lock(work):
        marker = work/'identity.json'; identity = {'inference_signature': signature}
        if marker.exists(): require(read_json(marker) == identity, 'Inference resume context changed')
        else: atomic_json(marker, identity)
        cfg = replace(BaselineConfig(), alignment_samples=base['alignment_samples'], collision_grid=base['collision_grid'],
                      minimum_fit_length_px=base['minimum_fit_length_px'], minimum_scale=base['minimum_scale'],
                      maximum_scale=base['maximum_scale'], feature_cache_entries=256)
        inputs = ObservableInputs(gallery, cfg); inputs.pose_windows = {}; bank = FeatureBank(inputs)
        pairs, geometry = retrieve_parents(inputs, recipe); score_path = work/'rankings.json'
        rankings_reused = score_path.exists(); resumed_pairs = 0; new_pairs = 0
        if score_path.exists():
            saved = read_json(score_path); require(saved['signature'] == signature, 'Ranking cache identity changed')
            require(saved['payload_sha256'] == digest(saved['rows']), 'Ranking cache checksum mismatch'); scores = saved['rows']; parity = saved['cpu_cuda_parity']
            require([(r['query'], r['target']) for r in scores] == pairs, 'Ranking cache pool changed')
        elif pairs:
            score_started = time.perf_counter()
            rankers = FrozenRankers(root, package_manifest, device); scores = rankers.score(bank, pairs, progress); parity = rankers.parity
            atomic_json(score_path, {'signature': signature, 'rows': scores, 'payload_sha256': digest(scores), 'cpu_cuda_parity': parity,
                                    'scoring_seconds': time.perf_counter()-score_started})
            del rankers
        else: scores = []; parity = []
        parents, retrieval_audit = fuse_parents(scores, geometry, gallery, recipe)
        chosen = selection_rows(parents, gallery)
        atomic_json(work/'parent_selection.json', {'inference_signature': signature, 'audit': retrieval_audit, 'rows': chosen, 'payload_sha256': digest(chosen)})
        spacing, uncertainty = observed_risk(inputs); grouped = defaultdict(list)
        for parent in parents:
            attach_windows(parent, inputs, spacing, uncertainty, recipe)
            pair = tuple(sorted(gallery[i][1]['entry']['piece_id'] for i in (parent['query'], parent['target'])))
            grouped[pair].append(parent)
        variants = {name: {'records': []} for name in ('alpha_rigid', 'alpha_sliding', 'alpha_rigid_refined')}; costs = []
        curves = {}
        def feature(owner, count):
            if (owner, count) not in curves:
                curves[owner, count] = curve(inputs, owner, count)
                if len(curves) > 128: curves.pop(next(iter(curves)))
            return curves[owner, count]
        for pair_number, (pair, pair_parents) in enumerate(sorted(grouped.items()), 1):
            cache = work/('pair_'+digest(pair)[:24]+'.json')
            if cache.exists():
                resumed_pairs += 1
                block = read_json(cache)
                require(block['signature'] == signature and block['pieces'] == list(pair) and block['sha256'] == digest(block['variants']), 'Pair cache changed')
            else:
                new_pairs += 1
                progress(f'Observable pose search: physical pair {pair_number}/{len(grouped)} {pair}')
                rows = []; lookup = {}; bootstrap_audit = []; pair_variants = {k: {'records': []} for k in variants}
                visual_cache = VisualCache()
                visual_preparation = VisualPreparationCache(pair)
                def record(pose, variant, parent, index):
                    q, t = pose['query'], pose['target']; h = normalized_matrix(pose['alignment']['transform'])
                    require_rigid(h)
                    canonical = require_rigid(h if gallery[q][1]['entry']['piece_id'] == pair[0] else normalized_matrix(np.linalg.inv(h)))
                    broad = pose.get('_broad') or evidence(feature(q, 128), feature(t, 128), h, pose, config['correction_configuration'])
                    visual = visual_cache.evaluate(q, t, pose, lambda: visual_profile(feature(q, 512), feature(t, 512), pose, bank.image(q), bank.image(t), config['correction_configuration'], config['policy'], visual_preparation, q))
                    weight = .55*pose['alignment']['geometry_score']+.2*parent['retrieval_score']+.25*broad['score']+visual.get('bounded_weight_adjustment', 0.)
                    value = {'record_id': variant+':'+digest([pair, index, q, t, canonical.tolist()])[:24],
                        'source_record_id': pose.get('_source_record_id'), 'piece_a': pair[0], 'piece_b': pair[1],
                        'pose': clean_pose(pose), 'canonical_transform_target_b_to_query_a': canonical.tolist(),
                        'methods': {'alpha_fused': {'weight': float(weight), 'parent_rank': parent['parent_rank'],
                            'reason': 'observable_frozen_core', 'score_scope': 'bounded_new_input_composition',
                            'retrieval_score': parent['retrieval_score'], 'pair_ranker_scores': parent['ranker_scores']}},
                        'selection_evidence': {'geometry': broad}, 'visual_evidence': visual}
                    return value
                # Canonical native-crop probes, shared across directed candidate pairs.
                b_entry = next(e for e in entries if e['piece_id'] == pair[1]); hh, ww = b_entry['size_hw']
                probes = np.array([[0, 0], [ww-1, 0], [0, hh-1], [ww-1, hh-1]], float)
                for pi, parent in enumerate(pair_parents):
                    fitted = fit_parent(parent, inputs, base)
                    starts = fitted['poses']; start_variant = 'alpha_rigid'
                    if not starts:
                        starts, detail = sliding_starts(parent, feature, inputs, base, config['correction_configuration'], config['policy'])
                        bootstrap_audit.append({'parent_index': pi, **detail}); start_variant = 'alpha_sliding'
                    for j, pose in enumerate(starts):
                        row = record(pose, start_variant, parent, ['start', pi, j]); pair_variants[start_variant]['records'].append(row)
                        qpiece = gallery[pose['query']][1]['entry']['piece_id']
                        def canonical_probes(h, reverse=qpiece != pair[0], points=probes):
                            return apply_transform(points, np.linalg.inv(h) if reverse else h)
                        rows.append({'record_id': row['record_id'], 'parent_index': pi, 'pose': pose, '_canonical_probes': canonical_probes})
                        lookup[row['record_id']] = parent
                if rows:
                    # The bounded search must cover each candidate parent before
                    # spending additional slots on its alternative transforms.
                    counts = defaultdict(int)
                    for row in rows:
                        row['_round_robin'] = counts[row['parent_index']]; counts[row['parent_index']] += 1
                    rows.sort(key=lambda r: (r['_round_robin'], r['parent_index'], r['record_id']))
                    pools, arrays, audit = generate(rows, feature, inputs.mask, inputs.mask_samples, base,
                                              config['correction_configuration'], config['policy'], SEARCH)
                    generated_trace(pair, arrays, audit)
                    for variant, source_name in (('alpha_sliding', 'sliding_joint'), ('alpha_rigid_refined', 'projective_joint')):
                        for i, pose in enumerate(pools[source_name]):
                            parent = lookup[pose['_source_record_id']]
                            pair_variants[variant]['records'].append(record(pose, variant, parent, i))
                else:
                    audit = {'reason': 'no_admissible_similarity_or_sliding_start', 'sliding_search_performed': True,
                             'resource_or_initializer_limit': True, 'search_domain_exhausted_claimed': False}
                audit['sliding_bootstrap'] = bootstrap_audit
                audit['exact_visual_cache'] = visual_cache.summary()
                audit['visual_preparation'] = visual_preparation.summary()
                block = {'signature': signature, 'pieces': list(pair), 'variants': pair_variants,
                         'sha256': digest(pair_variants), 'search_audit': clean_pose(audit)}
                atomic_json(cache, block)
            for name in variants: variants[name]['records'].extend(block['variants'][name]['records'])
            costs.append({'pieces': list(pair), 'search_audit': block['search_audit']})
        require(unchanged(receipts, True), 'Stage 1/2 inputs changed during inference')
        provenance = {'runtime_variant': 'unit-scale-v5', 'kind': 'new_input_frozen_core_composition', 'inference_signature': signature,
            'package_manifest_sha256': package_manifest['manifest_sha256'], 'recipe': recipe,
            'inputs': [{'sha256': r['sha256'], 'size': r['size']} for r in receipts],
            'retrieval': retrieval_audit, 'cpu_cuda_parity': parity, 'search': costs,
            'training_performed': False, 'optimizer_steps': 0, 'validation_truth_opened': False,
            'historical_control_exactly_reproduced': False}
        manifest = export_bundle(output, puzzle_id, entries, gallery, variants, provenance=provenance, configuration=config,
                                 maximum=recipe['working_pool_per_physical_pair'])
        result = {**load_bundle(output, full=True).summary(), 'invocation_seconds': time.perf_counter()-started,
                  'rankings_reused': rankings_reused, 'physical_pair_caches_reused': resumed_pairs, 'physical_pairs_newly_computed': new_pairs,
                  'device': device, 'reused': False, 'bundle': str(output), 'source_inputs_verified_unchanged': True}
        result['feature_image_cache'] = bank.summary()
        atomic_json(work/'completion.json', {'signature': signature, 'manifest_sha256': manifest['manifest_sha256'], 'summary': result})
        return result
