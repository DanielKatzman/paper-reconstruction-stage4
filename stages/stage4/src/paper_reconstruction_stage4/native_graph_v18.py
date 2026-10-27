"""Native graph search with immediate validation and deferred backtracking.

The fast beam is an ordering pass, not an exhaustion claim: discarded branches
remain reachable in the following fair depth-first sessions. Only a verified
construction, deadline, or exhaustion of this finite proposal graph ends search.
"""
from collections import Counter, defaultdict, OrderedDict, deque
from dataclasses import dataclass
import time
import numpy as np
from .native_graph_v17 import Growth, offers, quality, identity, pose_identity, incremental_check
from .geometry import regauge
from .constraints import require_rigid
from .native_bridge import contour_arc, check_arcs
from .support import overlaps
from .repair_search import Node
from .search.execution import clone
from .component_closure import shape_measurement
from .page_geometry import state_record


@dataclass
class Frame:
    growth: Growth
    choices: object


def search(context, basis, proposals, deadline, on_complete, physical=None, fast_only=False, on_idle=None):
    from .feasibility import Feasibility
    if basis.selected or any(len(g) != 1 for g in basis.components()) or basis.assignments or basis.active_support:
        raise ValueError('Independent native graph requires an empty singleton source')
    if any(not np.array_equal(h, np.eye(3)) for h in basis.poses.values()):
        raise ValueError('Independent native graph cannot inherit old piece poses')
    if not context.problem.capture_metadata or context.problem.capture_metadata.get('rotation_family') != 'quarter_turn':
        raise ValueError('Independent native graph requires declared quarter-turn capture')
    started = time.perf_counter(); physical = physical or Feasibility(context.problem, context.config)
    indexed = defaultdict(list); primary_index = defaultdict(list); decoded = {}; rejected = Counter(); checks = 0
    replay_checks = 0
    ordered = sorted(proposals, key=lambda p: (not p['conditioned'], -quality(p), p['proposal_id']))
    for proposal in ordered:
        for pid in proposal['pieces']:
            indexed[pid].append(proposal)

    # A small front-of-queue view prevents weak alternative modes from
    # fabricating corroboration for one another. Every other fitted mode stays
    # in the full index and in the parallel expanded backtracking sessions.
    from .seam_poses_v15 import retain
    per_pair = defaultdict(list)
    for proposal in ordered:
        per_pair[tuple(proposal['pieces'])].append(proposal)
    for values in per_pair.values():
        for proposal in retain(values):
            for pid in proposal['pieces']:
                primary_index[pid].append(proposal)

    def placement_choices(growth, primary_only=False):
        primary = offers(growth, primary_index, context.config.cycle_tolerance_px)
        yield from primary
        if not primary_only:
            used = {r['proposal']['proposal_id'] for r in primary}
            yield from (r for r in offers(growth, indexed, context.config.cycle_tolerance_px)
                        if r['proposal']['proposal_id'] not in used)

    def extend(growth, offer):
        nonlocal checks, replay_checks
        checks += 1; proposal = offer['proposal']; a, b = proposal['pieces']
        state = growth.node.state; pid = offer['piece']
        key = proposal['proposal_id']
        if key not in decoded:
            try:
                h = np.asarray(proposal['transform_b_to_a'], float); require_rigid(h)
                if not np.allclose(h[:2, :2], np.rint(h[:2, :2]), atol=1e-7, rtol=0.):
                    raise ValueError('Undeclared graph rotation')
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

    best = []; deepest = 1; finals = 0; coarse = 0; backtracks = 0
    memo = OrderedDict(); final_memo = OrderedDict(); answer = None
    deadline_hit = False

    def remember(cache, key):
        if key in cache:
            cache.move_to_end(key)
            return False
        cache[key] = True
        if len(cache) > context.config.maximum_cache_entries:
            cache.popitem(last=False)  # Eviction repeats work; it never drops an unexplored branch.
        return True

    def consider(growth):
        nonlocal deepest, best, finals, coarse, answer
        state = growth.node.state
        depth = sum(state.groups[p] == state.groups[growth.anchor] for p in state.poses)
        if depth > deepest:
            deepest = depth; best = [growth]
        elif depth == deepest and len(best) < 3:
            best.append(growth)
        if depth != len(state.poses):
            return False
        coarse += 1
        shape = shape_measurement(context, state, 256)
        if not shape['shape_gate_pass']:
            rejected['complete_shape'] += 1
            return True
        if remember(final_memo, identity(growth)) and time.perf_counter() < deadline:
            finals += 1
            # Includes fine geometry, physical checks, joint native support
            # selection and independent serialized-evidence replay.
            answer = on_complete(growth.node, shape)
        return True

    # A fast diverse ordering pass preserves the old positive routes. Its
    # branch limits only choose ordering; the backtracking pass reopens them.
    seeds = []; rest = []; pairs = set()
    for proposal in ordered:
        pair = tuple(proposal['pieces'])
        (rest if pair in pairs else seeds).append(proposal)
        pairs.add(pair)
    seed_order = [*seeds, *rest]

    def seed(proposal):
        a, b = proposal['pieces']
        initial = Growth(Node(clone(basis)), a, proposal['proposal_id'])
        return extend(initial, {'proposal': proposal, 'piece': b,
            'pose': np.asarray(proposal['transform_b_to_a']), 'priority': quality(proposal)})

    nodes = []
    for proposal in seed_order:
        if time.perf_counter() >= deadline or len(nodes) >= context.config.repair_beam//2:
            break
        growth = seed(proposal)
        if growth is not None:
            if consider(growth) and answer is not None:
                break
            nodes.append(growth)
    while nodes and answer is None and time.perf_counter() < deadline:
        following = []
        for growth in nodes:
            by_piece = defaultdict(list)
            for offer in placement_choices(growth, primary_only=True):
                by_piece[offer['piece']].append(offer)
            expanded = 0
            for choices in by_piece.values():
                accepted = 0
                for offer in choices:
                    if time.perf_counter() >= deadline:
                        break
                    child = extend(growth, offer)
                    if child is None:
                        continue
                    if consider(child):
                        if answer is not None:
                            break
                    else:
                        following.append(child)
                    accepted += 1
                    if accepted == 2:
                        break
                if answer is not None:
                    break
                if accepted:
                    expanded += 1
                if expanded == 2:
                    break
            if answer is not None or time.perf_counter() >= deadline:
                break
        following.sort(key=lambda g: (-g.node.value, identity(g)))
        keep = []; used = set(); families = set()
        for growth in following:
            if growth.seed not in families and len(keep) < context.config.repair_beam//2:
                keep.append(growth); used.add(identity(growth)); families.add(growth.seed)
        for growth in following:
            key = identity(growth)
            if key not in used:
                keep.append(growth); used.add(key)
            if len(keep) >= context.config.repair_beam:
                break
        nodes = keep
        if answer is None and on_idle is not None and time.perf_counter() < deadline:
            answer = on_idle()

    # Sessions retain their entire alternative-offer stack. Round-robin
    # advances each live seed, so one misleading good-looking seed cannot
    # consume the entire deadline. The window bounds memory, not total seeds.
    sessions = deque(); pending = iter((p, primary) for p in seed_order for primary in (True, False)); seeds_exhausted = False
    seeds_started = 0

    def frame(growth, primary_only):
        return Frame(growth, iter(placement_choices(growth, primary_only)))

    while not fast_only and answer is None and time.perf_counter() < deadline:
        while not seeds_exhausted and len(sessions) < context.config.repair_beam:
            seed_item = next(pending, None)
            if seed_item is None:
                seeds_exhausted = True
                break
            proposal, primary_only = seed_item
            seeds_started += 1
            growth = seed(proposal)
            if growth is not None:
                if consider(growth):
                    if answer is not None:
                        break
                else:
                    sessions.append((primary_only, [frame(growth, primary_only)]))
            if time.perf_counter() >= deadline:
                break
        if answer is not None or not sessions:
            break
        primary_only, stack = sessions.popleft(); top = stack[-1]
        offer = next(top.choices, None)
        if offer is None:
            stack.pop(); backtracks += 1
        else:
            child = extend(top.growth, offer)
            if child is not None and remember(memo, (primary_only, identity(child))):
                if not consider(child):
                    stack.append(frame(child, primary_only))
        if stack:
            sessions.append((primary_only, stack))
        if answer is None and on_idle is not None and time.perf_counter() < deadline:
            answer = on_idle()
    deadline_hit = time.perf_counter() >= deadline and answer is None
    audit = {'policy': 'geometry-checked-native-backtracking.v1',
        'source_groups': basis.components(), 'legacy_selected_joins_inherited': 0,
        'old_piece_poses_inherited': False, 'evaluation_truth_opened': False,
        'merge_checks': checks, 'maximum_merge_checks': None, 'proposal_count_is_telemetry': True,
        'final_candidates_checked': finals, 'final_candidate_count_limit': None,
        'complete_shape_candidates': coarse, 'completed_depth': deepest,
        'required_piece_depth': len(basis.poses), 'native_proposal_replay_checks': replay_checks,
        'backtracks': backtracks, 'backtracking_seed_sessions_started': seeds_started,
        'rejections': dict(rejected), 'budget_abstained': deadline_hit,
        'stop_reason': 'verified_construction' if answer is not None else 'overall_deadline' if deadline_hit else 'fast_pass_yielded' if fast_only else 'configured_native_graph_exhausted',
        'growth_choices_exhaustive': False, 'proposal_graph_exhausted': bool(answer is None and not deadline_hit and seeds_exhausted and not sessions),
        'fast_pass_pruning_is_reopened': True, 'primary_and_expanded_sessions_interleaved': True, 'lower_modes_permanently_discarded': False, 'elapsed_seconds': time.perf_counter()-started,
        'observed_growth_states': [{'state': state_record(g.node.state), 'seed_proposal_id': g.seed,
            'value': g.node.value, 'placement_proposal_ids': [r['proposal_id'] for r in g.node.contacts],
            'component_sizes': sorted(map(len, g.node.state.components()), reverse=True),
            'accepted_assembly': False} for g in best]}
    return answer, audit

