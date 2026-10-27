"""Bounded alternative core splits from observed native pose disagreement."""
from collections import defaultdict, deque
import time
import numpy as np
from paper_reconstruction_contracts import map_points
from .attachments import without_join
from .geometry import relative
from .outputs import native_probes
from .page_geometry import fingerprint
from .search.execution import clone


def path_edges(context, source, a, b):
    adjacency = defaultdict(list)
    for eid in sorted(source.selected):
        edge = context.edges[eid]
        adjacency[edge.a].append((edge.b, eid))
        adjacency[edge.b].append((edge.a, eid))
    pending = deque([(a, [])]); seen = {a}
    while pending:
        pid, path = pending.popleft()
        if pid == b:
            return path
        for other, eid in adjacency[pid]:
            if other not in seen:
                seen.add(other); pending.append((other, [*path, eid]))
    return []


def alternative_sources(context, originals, proposals, deadline):
    """Retain the original first source; split alternatives never certify truth."""
    candidates = []; evidence = []
    for source, removed, kind in originals[:2]:
        if time.perf_counter() >= deadline:
            break
        risk = defaultdict(float); conflicts = []
        for proposal in proposals:
            a, b = proposal['pieces']
            if source.groups[a] != source.groups[b] or not proposal['conditioned']:
                continue
            probes = np.asarray(native_probes(context.problem.pieces[b]))
            delta = np.linalg.norm(map_points(probes, relative(source.poses, a, b)) -
                                   map_points(probes, np.asarray(proposal['transform_b_to_a'])), axis=1)
            error = float(np.quantile(delta, .95))
            if error <= 2 * context.config.contact_tolerance_px:
                continue
            path = path_edges(context, source, a, b)
            weight = min(proposal['native_length_px'], 512.) * min(proposal['shape_spread_ratio'], .2)
            for eid in path:
                risk[eid] += weight / max(1, len(path))
            conflicts.append({'pieces': [a, b], 'proposal_id': proposal['proposal_id'],
                              'probe_disagreement_px': error, 'selected_path': path})
        joins = {j['geometry_id']: j for j in context.contacts(source)['joins']}
        # Unwitnessed selected links supply a finite fallback when no conditioned
        # alternative has survived native proposal retention.
        ordered = sorted(source.selected, key=lambda eid: (
            -risk[eid], bool(joins[eid]['independent_witness_pairs']),
            joins[eid]['native_length_px'], eid))
        seed = clone(source); released = []
        for eid in ordered:
            if len(released) >= 4 or time.perf_counter() >= deadline:
                break
            if not risk[eid] and joins[eid]['independent_witness_pairs']:
                continue
            trial = without_join(seed, context.edges, eid, canonical=True)
            if len(trial.components()) > 8:
                continue
            seed = trial; released.append(eid)
        if released and len(seed.components()) >= 2:
            candidates.append((seed, [*removed, *released], 'native_conflict_forest'))
        # Removing an incident cut can open a cyclic bad core that single-edge
        # deletion leaves connected. Never release more than four extra links.
        incident = defaultdict(list)
        for eid in source.selected:
            edge = context.edges[eid]
            incident[edge.a].append(eid); incident[edge.b].append(eid)
        ranked = sorted(incident, key=lambda p: (-sum(risk[e] for e in incident[p]), len(incident[p]), p))
        for pid in ranked:
            if not 1 <= len(incident[pid]) <= 4 or not any(risk[e] for e in incident[pid]):
                continue
            split = clone(source)
            for eid in sorted(incident[pid]):
                split = without_join(split, context.edges, eid, canonical=True)
            if 2 <= len(split.components()) <= 8:
                candidates.append((split, [*removed, *sorted(incident[pid])], 'native_conflict_piece_cut'))
                break
        evidence.append({'source_candidate_id': fingerprint(source), 'conflicts': conflicts,
                         'ranked_release_geometry_ids': ordered, 'extra_release_limit': 4})
    # One original lane is protected. Alternatives get their own reserved slots.
    ordered = [*originals[:1], *candidates[:1], *originals[1:2], *candidates[1:], *originals[2:]]
    selected = []; families = set()
    for row in ordered:
        family = context.pose_family(row[0])
        if family in families:
            continue
        families.add(family); selected.append(row)
    return selected[:context.config.repair_source_limit], {
        'policy': 'observed-native-core-alternatives.v1', 'evidence': evidence,
        'eligible_families': len(selected), 'omitted_by_limit': max(0, len(selected)-context.config.repair_source_limit),
        'maximum_components': 8, 'maximum_extra_released_joins': 4,
        'disagreement_is_proof_of_wrong_core': False, 'evaluation_truth_opened': False}
