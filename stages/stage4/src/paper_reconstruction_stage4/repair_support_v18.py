"""Native supports trimmed around competing contacts at the same final pose."""
from collections import defaultdict
import time
import numpy as np
from .geometry import relative
from .native_bridge import pair_options, check_arcs, arc_value
from .seam_poses import spread
from .support import overlaps
from .outputs import digest


def reachable(pieces, pairs, start, excluded=None):
    adjacency = defaultdict(set)
    for index, (a, b) in enumerate(pairs):
        if index == excluded:
            continue
        adjacency[a].add(b); adjacency[b].add(a)
    seen = set(); pending = [start]
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid); pending.extend(adjacency[pid] - seen)
    return seen


def unsupported(context, source, contacts):
    fixed = [(context.edges[e].a, context.edges[e].b) for e in source.selected]
    pairs = [*fixed, *[row['pieces'] for row in contacts]]
    result = []
    for i, row in enumerate(contacts):
        length = min(a.length for a in row['arcs']); condition = spread(row['arcs'])
        if length >= 96. and condition >= .02:
            continue
        a, b = row['pieces']; cut = reachable(source.poses, pairs, a, len(fixed)+i)
        if b not in cut:
            result.append({'pieces': [a, b], 'proposal_id': row['proposal_id'],
                           'native_length_px': length, 'shape_spread_ratio': condition,
                           'cut': sorted(cut), 'other_side': sorted(set(source.poses)-cut),
                           'arcs': [arc_value(v) for v in row['arcs']]})
    return result


def pool(context, source, node, deadline):
    used = [arc for eid in source.selected for arc in context.support.arcs_for(source, eid)]
    mandatory = {}; cycles = []; ids = sorted(node.state.poses)
    original = {tuple(row['pieces']): row for row in node.contacts if row['role'] == 'merge'}
    legacy_occupied = [*used, *[arc for row in node.contacts for arc in row['arcs']]]
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if time.perf_counter() >= deadline:
                return None, None, used
            if source.groups[a] == source.groups[b] or node.state.groups[a] != node.state.groups[b]:
                continue
            pair = (a, b); choices = []
            base = original.get(pair)
            if base is not None:
                try:
                    check_arcs(context.problem, base['arcs'], node.state.poses, context.config, used)
                    choices.append(base)
                except ValueError:
                    pass
            h = relative(node.state.poses, a, b); proposed = h.copy()
            if context.problem.capture_metadata['rotation_family']=='quarter_turn':
                proposed[:2, :2] = np.rint(h[:2, :2])
            options = pair_options(context.problem, a, b, node.state.poses, context.config, used)
            if base is not None:
                other_contacts = [*used, *[arc for row in node.contacts if tuple(row['pieces']) != pair for arc in row['arcs']]]
                options += pair_options(context.problem, a, b, node.state.poses, context.config, other_contacts)
            if base is None:
                # Preserve contacts trimmed around the original merge supports,
                # as measured by V14/V15's positive completion route.
                options += pair_options(context.problem, a, b, node.state.poses, context.config, legacy_occupied)
            for arcs, metrics in options:
                if base is None and min(v.length for v in arcs) < 96.:
                    continue
                choices.append({'pieces': [a, b], 'arcs': arcs, 'metrics': metrics,
                    'role': 'merge' if base is not None else 'cycle',
                    'proposal_id': base['proposal_id'] if base else digest(['native-contact-at-final-pose.v1', a, b, h.tolist()]),
                    'proposed_transform_b_to_a': base['proposed_transform_b_to_a'] if base else proposed.tolist(),
                    'source_groups': base['source_groups'] if base else source.components(),
                    'merged_groups': base['merged_groups'] if base else [
                        sorted(p for p in source.poses if source.groups[p] == source.groups[a]),
                        sorted(p for p in source.poses if source.groups[p] == source.groups[b])]})
            unique = {digest([arc_value(v) for v in row['arcs']]): row for row in choices}
            choices = sorted(unique.values(), key=lambda r: (-min(v.length for v in r['arcs']), r['proposal_id']))
            if base is not None:
                mandatory[pair] = choices
            else:
                cycles.extend(choices)
    return mandatory, cycles, used

