"""Bounded remove/replace and component rebuilding from observable edges."""
from .common import retain


def repair(search, beam, budget):
    checks = 0
    exhausted = False
    for _ in range(search.config.repair_rounds):
        proposals = list(beam)
        for state in beam[:search.config.portfolio_size]:
            for removed in sorted(state.selected):
                kept = tuple(eid for eid in state.selected if eid != removed)
                # Rebuilding can split a component and frees its old boundary.
                rebuilt = search.rebuild(kept, budget)
                if rebuilt is None:
                    if budget.reason != "exhausted_candidates":
                        exhausted = True
                        break
                    continue
                proposals.append(rebuilt)
                for edge in search.problem.edges:
                    if checks >= search.config.max_repair_checks or not budget.take():
                        exhausted = True
                        break
                    checks += 1
                    candidate = search.attempt(rebuilt, edge)
                    if candidate is not None:
                        proposals.append(candidate)
                if exhausted:
                    break
            if exhausted:
                break
        updated = retain(proposals, search.config.beam_width)
        if exhausted or updated[0].score <= beam[0].score + 1e-12:
            beam = updated
            break
        beam = updated
    return beam, {"enabled": True, "checks": checks, "truncated": exhausted,
                  "moves": "remove one selected join, rebuild components, try each replacement"}
