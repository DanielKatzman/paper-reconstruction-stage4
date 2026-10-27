"""Bounded triangle/four-cycle seeds; main search retains the complete pool."""
from collections import defaultdict
from itertools import product
from ..model import singleton_state


def loop_seeds(search, budget):
    config = search.config
    pairs, adjacency = defaultdict(list), defaultdict(set)
    for edge in search.problem.edges:
        pairs[edge.a, edge.b].append(edge)
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)
    for pair in pairs:
        pairs[pair] = pairs[pair][:config.loop_alternatives_per_pair]
    checked, seeds, seen = 0, [], set()
    pieces = sorted(search.problem.pieces)
    stop = False
    for a in pieces:
        if stop:
            break
        # Canonical cycles: smallest vertex first, second < last.
        paths = ((a, b, c) for b in sorted(adjacency[a]) for c in sorted(adjacency[b])
                 if a < b < c and a in adjacency[c])
        four = ((a, b, c, d) for b in sorted(adjacency[a]) for c in sorted(adjacency[b])
                if c > a and c != b for d in sorted(adjacency[c])
                if d > b and d != a and a in adjacency[d])
        from itertools import chain
        for vertices in chain(paths, four):
            links = [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
            alternatives = [pairs[tuple(sorted(pair))] for pair in links]
            for selected in product(*alternatives):
                if checked >= config.max_loop_checks or len(seeds) >= config.max_loop_seeds:
                    stop = True
                    break
                checked += 1
                key = tuple(sorted(e.id for e in selected))
                if key in seen:
                    continue
                seen.add(key)
                state = singleton_state(search.problem)
                for edge in selected:
                    if not budget.take():
                        stop = True
                        state = None
                        break
                    state = search.attempt(state, edge)
                    if state is None:
                        break
                if state is not None:
                    seeds.append(state)
                if stop:
                    break
            if stop:
                break
    return seeds, {"enabled": True, "checked_combinations": checked, "accepted_seeds": len(seeds),
                   "alternatives_per_pair_for_seeds": config.loop_alternatives_per_pair,
                   "seed_search_truncated": stop, "working_pool_unchanged": True}
