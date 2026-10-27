"""Independent, bounded native-pose growth from separate physical pieces.

Only observed native proposals enter the graph. No old selected joins or piece
poses are inherited. Final contact extraction and independent replay are still
required before any graph completion can enter a published assembly.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import time
import numpy as np
from .geometry import regauge, relative
from paper_reconstruction_contracts import map_points
from .constraints import require_rigid, require_native_rotation
from .native_bridge import contour_arc, check_arcs
from .support import overlaps
from .outputs import digest
from .repair_search import Node
from .search.execution import clone
from .component_closure import shape_measurement
from .page_geometry import state_record


@dataclass
class Growth:
    node: Node
    anchor: str
    seed: str


def pose_identity(growth):
    state = growth.node.state
    poses = []
    for pid in sorted(state.poses):
        value = np.round(state.poses[pid], 6)
        value[value == 0.] = 0.
        poses.append([pid, state.groups[pid], value.tolist()])
    return digest(poses)


def identity(growth):
    return digest([pose_identity(growth), sorted(row['proposal_id'] for row in growth.node.contacts)])


def quality(proposal):
    return float(proposal['native_length_px'] * (.1+min(proposal['shape_spread_ratio'], .35)) /
                 (1.+proposal['native_metrics']['bidirectional_contact_p95_native_px']/4.))


def incremental_check(context, candidate, new_piece, members, physical, counters):
    """Coarse growth only: the previous placed group was already checked.

    An axis box within the area allowance is a cheap sufficient enclosing
    rectangle. Otherwise retain the exact minimum-rectangle check. Only new
    collision pairs are introduced. The complete final state is rechecked by
    the unchanged independent finalizer before acceptance.
    """
    points = np.vstack([map_points(context.hulls[p], candidate.poses[p]) for p in sorted(members)])
    upper = float(np.prod(np.ptp(points, axis=0)+1.))
    if upper <= sum(context.areas.values())*(1.+context.config.page_component_area_slack):
        counters['coarse_axis_rectangle_passes'] += 1
    elif not context.component_bounds(candidate)['valid']:
        return False, 'component_area_bound'
    require_rigid(candidate.poses[new_piece])
    for other in sorted(members-{new_piece}):
        a, b = sorted((new_piece, other))
        check = physical.pair(a, b, relative(candidate.poses, a, b), final=False)
        counters['incremental_collision_pair_checks'] += 1
        if not check['valid']:
            return False, check['reason']
    return True, 'accepted'


def offers(growth, indexed, tolerance):
    state = growth.node.state; group = state.groups[growth.anchor]
    seen = set(); rows = []
    for pid in sorted(state.poses):
        if state.groups[pid] != group:
            continue
        for proposal in indexed[pid]:
            if proposal['proposal_id'] in seen:
                continue
            seen.add(proposal['proposal_id']); a, b = proposal['pieces']
            inside_a, inside_b = state.groups[a] == group, state.groups[b] == group
            if inside_a == inside_b:
                continue
            unknown, neighbor = (b, a) if inside_a else (a, b)
            h = np.asarray(proposal['transform_b_to_a'])
            pose = state.poses[a]@h if inside_a else state.poses[b]@np.linalg.inv(h)
            rows.append({'proposal': proposal, 'piece': unknown, 'neighbor': neighbor,
                         'pose': pose, 'quality': quality(proposal)})
    by_piece = defaultdict(list)
    for row in rows:
        by_piece[row['piece']].append(row)
    for alternatives in by_piece.values():
        rotations = np.stack([r['pose'][:2, :2].ravel() for r in alternatives])
        translations = np.stack([r['pose'][:2, 2] for r in alternatives])
        compatible = ((np.max(np.abs(rotations[:, None]-rotations[None, :]), axis=2) <= 1e-7) &
                      (np.linalg.norm(translations[:, None]-translations[None, :], axis=2) <= tolerance))
        for index, row in enumerate(alternatives):
            witnesses = {}
            for other_index in np.flatnonzero(compatible[index]):
                other = alternatives[other_index]
                if other['neighbor'] == row['neighbor']:
                    continue
                witnesses[other['neighbor']] = max(witnesses.get(other['neighbor'], 0.), other['quality'])
            # Distinct neighbors can corroborate a placement. These are cheap
            # graph heuristics, not accepted native contact or confidence claims.
            row['cycle_neighbors'] = len(witnesses)
            row['priority'] = row['quality']+sum(sorted(witnesses.values(), reverse=True)[:4])+100.*min(4, len(witnesses))
    return sorted(rows, key=lambda r: (-r['cycle_neighbors'], -r['priority'], r['piece'], r['proposal']['proposal_id']))


def search(context, basis, proposals, deadline, maximum_checks, beam=24, physical=None):
    from .feasibility import Feasibility
    if basis.selected or any(len(g) != 1 for g in basis.components()) or basis.assignments or basis.active_support:
        raise ValueError('Independent native graph requires an empty singleton source')
    if any(not np.array_equal(h, np.eye(3)) for h in basis.poses.values()):
        raise ValueError('Independent native graph cannot inherit old piece poses')
    if not context.problem.capture_metadata or context.problem.capture_metadata.get('rotation_family') not in {'quarter_turn','continuous_rigid'}:
        raise ValueError('Independent native graph requires authenticated rigid capture')
    started = time.perf_counter(); physical = physical or Feasibility(context.problem, context.config)
    indexed = defaultdict(list); decoded = {}; rejected = Counter(); checks = 0
    replay_checks = 0; nodes = []; complete = []; peak = 0; dropped = 0; depth = 1
    ordered = sorted(proposals, key=lambda p: (not p['conditioned'], -quality(p), p['proposal_id']))
    for proposal in ordered:
        for pid in proposal['pieces']:
            indexed[pid].append(proposal)

    def extend(growth, offer):
        nonlocal checks, replay_checks
        checks += 1; proposal = offer['proposal']; a, b = proposal['pieces']
        state = growth.node.state; pid = offer['piece']
        key = proposal['proposal_id']
        if key not in decoded:
            try:
                h = np.asarray(proposal['transform_b_to_a'], float); require_native_rotation(h, context.problem.capture_metadata)
                arcs = tuple(contour_arc(context.problem, row) for row in proposal['arcs'])
                if [arc.piece for arc in arcs] != [a, b]:
                    raise ValueError('Proposal arc pair differs')
                replay_checks += 1
                metrics = check_arcs(context.problem, arcs, {a: np.eye(3), b: h}, context.config)
                decoded[key] = (arcs, metrics)
            except ValueError:
                decoded[key] = None
        if decoded[key] is None:
            rejected['native_proposal_replay'] += 1; return None
        arcs, metrics = decoded[key]
        used = [arc for row in growth.node.contacts for arc in row['arcs']]
        occupied_overlap = any(overlaps(a, b, context.config) for a in arcs for b in used)
        if occupied_overlap:
            # These are unaccepted pose hypotheses. Do not lock coarse arc
            # ownership before the complete-pose joint support assignment.
            rejected['deferred_occupied_support'] += 1
        candidate = clone(state); candidate.poses[pid] = offer['pose'].copy()
        members = {p for p in state.poses if state.groups[p] == state.groups[growth.anchor]} | {pid}
        for member in members:
            candidate.groups[member] = min(members)
        candidate.poses = regauge(candidate.poses, members)
        valid, reason = incremental_check(context, candidate, pid, members, physical, rejected)
        if not valid:
            rejected[reason] += 1; return None
        contact = {'pieces': [a, b], 'arcs': arcs, 'metrics': metrics,
            'proposal_id': key, 'proposed_transform_b_to_a': proposal['transform_b_to_a'],
            'source_groups': state.components(), 'merged_groups': [
                sorted(p for p in state.poses if state.groups[p] == state.groups[a]),
                sorted(p for p in state.poses if state.groups[p] == state.groups[b])],
            'role': 'merge', 'spread': proposal['shape_spread_ratio']}
        rejected['accepted'] += 1
        value = offer['priority']-(.5*quality(proposal) if occupied_overlap else 0.)
        return Growth(Node(candidate, [*growth.node.contacts, contact], growth.node.value+value), growth.anchor, growth.seed)

    # Separate well-conditioned physical pairs supply different anchored starts.
    # We retain alternative modes after physical-pair diversity has been filled.
    seeds = []; used_pairs = set(); rest = []
    seed_limit = max(2, min(12, beam//2))
    for proposal in ordered:
        pair = tuple(proposal['pieces'])
        if pair in used_pairs:
            rest.append(proposal); continue
        used_pairs.add(pair); seeds.append(proposal)
        if len(seeds) >= seed_limit:
            break
    if len(seeds) < seed_limit:
        seeds.extend(rest[:seed_limit-len(seeds)])
    attempted_seeds = []; accepted_seeds = []
    remainder = [p for p in ordered if p['proposal_id'] not in {s['proposal_id'] for s in seeds}]
    for proposal in [*seeds, *remainder]:
        if len(accepted_seeds) >= seed_limit or len(attempted_seeds) >= 96:
            break
        if checks >= maximum_checks or time.perf_counter() >= deadline:
            break
        attempted_seeds.append(proposal['proposal_id'])
        a, b = proposal['pieces']; initial = Growth(Node(clone(basis)), a, proposal['proposal_id'])
        offer = {'proposal': proposal, 'piece': b, 'pose': np.asarray(proposal['transform_b_to_a']), 'priority': quality(proposal)}
        row = extend(initial, offer)
        if row is not None:
            nodes.append(row); accepted_seeds.append(proposal['proposal_id'])
    depth = 2 if nodes else 1
    deepest = depth
    best_frontier = list(nodes)
    while nodes:
        if depth == len(basis.poses):
            for growth in nodes:
                if time.perf_counter() >= deadline:
                    break
                shape = shape_measurement(context, growth.node.state, 256)
                if shape['shape_gate_pass']:
                    complete.append((growth.node, shape))
                else:
                    rejected['complete_shape'] += 1
            break
        following = []; seen = set()
        for growth in nodes:
            candidates = offers(growth, indexed, context.config.cycle_tolerance_px)
            by_piece = defaultdict(list)
            for offer in candidates:
                by_piece[offer['piece']].append(offer)
            expanded_pieces = 0; local_checks = 0
            for choices in by_piece.values():
                accepted = 0
                for offer in choices[:8]:
                    if checks >= maximum_checks or time.perf_counter() >= deadline or local_checks >= 32:
                        break
                    local_checks += 1; child = extend(growth, offer)
                    if child is None:
                        continue
                    identifier = identity(child)
                    if identifier in seen:
                        continue
                    seen.add(identifier); following.append(child); accepted += 1
                    deepest = max(deepest, depth+1)
                    if accepted >= 2:
                        break
                if accepted:
                    expanded_pieces += 1
                if expanded_pieces >= 2 or local_checks >= 32 or checks >= maximum_checks or time.perf_counter() >= deadline:
                    break
            if checks >= maximum_checks or time.perf_counter() >= deadline:
                break
        following.sort(key=lambda g: (-g.node.value, identity(g)))
        pose_counts = Counter(); distinct = []
        for growth in following:
            key = pose_identity(growth)
            if pose_counts[key] < 2:
                distinct.append(growth); pose_counts[key] += 1
        following = distinct
        keep = []; families = set()
        for growth in following:
            if growth.seed not in families:
                families.add(growth.seed); keep.append(growth)
            if len(keep) >= beam//2:
                break
        used = {identity(g) for g in keep}
        for growth in following:
            if len(keep) >= beam:
                break
            identifier = identity(growth)
            if identifier not in used:
                keep.append(growth); used.add(identifier)
        peak = max(peak, len(following)); dropped += max(0, len(following)-len(keep))
        nodes = keep; depth += 1
        if nodes:
            best_frontier = list(nodes)
        if (checks >= maximum_checks or time.perf_counter() >= deadline) and depth != len(basis.poses):
            break
    complete.sort(key=lambda row: (-row[1]['score'], -row[0].value))
    return complete, {'policy': 'independent-native-anchor-growth.v1',
        'source_groups': basis.components(), 'legacy_selected_joins_inherited': 0,
        'old_piece_poses_inherited': False, 'merge_checks': checks, 'maximum_merge_checks': maximum_checks,
        'coarse_support_assignment_deferred': True, 'next_piece_families_per_node': 2,
        'coarse_collision_policy': 'new-piece-pairs.v1', 'full_final_feasibility_unchanged': True,
        'placement_modes_per_piece': 2, 'placement_checks_per_node': 32,
        'native_proposal_replay_checks': replay_checks, 'completed_depth': deepest,
        'required_piece_depth': len(basis.poses), 'complete_shape_candidates': len(complete),
        'rejections': dict(rejected), 'pruning': {'beam_nodes_dropped': dropped, 'peak_successors': peak},
        'seed_proposal_ids': attempted_seeds, 'seed_count_started': len(attempted_seeds),
        'accepted_seed_proposal_ids': accepted_seeds,
        'observed_growth_states': [{'state': state_record(g.node.state), 'seed_proposal_id': g.seed,
            'value': g.node.value, 'placement_proposal_ids': [r['proposal_id'] for r in g.node.contacts],
            'component_sizes': sorted(map(len, g.node.state.components()), reverse=True),
            'accepted_assembly': False} for g in best_frontier[:3]],
        'stop_reason': ('time_limit' if time.perf_counter() >= deadline else 'check_limit' if checks >= maximum_checks
                        else 'complete_growth_examined' if deepest == len(basis.poses) else 'bounded_frontier_exhausted'),
        'growth_choices_exhaustive': False,
        'elapsed_seconds': time.perf_counter()-started,
        'budget_abstained': checks >= maximum_checks or time.perf_counter() >= deadline,
        'full_page_and_native_verification_required': True, 'evaluation_truth_opened': False}
