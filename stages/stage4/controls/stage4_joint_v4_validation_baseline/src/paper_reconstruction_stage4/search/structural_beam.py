"""Pair-wise beam, observable proposal ordering and coordinated fair repair."""
from collections import defaultdict
from itertools import combinations
import time
import numpy as np
from ..geometry import normalize, regauge
from ..model import State, singleton_state
from .partial_beam import PartialSearch
from .common import Budget, SearchResult, identity, objective


def retain(states, width):
    unique = {}
    for state in states:
        key = identity(state)
        if key not in unique or state.score > unique[key].score:
            unique[key] = state
    return [s for _, s in sorted(unique.items(), key=lambda kv: (-kv[1].score, kv[0]))[:width]]


class StructuralSearch(PartialSearch):
    def __init__(self, problem, config, feasibility, support):
        super().__init__(problem, config, feasibility, support)
        self.phase_seconds = {}
        self.trace = []
        self.priorities = {}
        pairs = defaultdict(list)
        for edge in problem.edges:
            try:
                state = singleton_state(problem)
                state.poses[edge.b] = normalize(edge.transform)
                state.groups[edge.b] = edge.a
                # Proposal ordering uses the same observable component terms;
                # support/contact assignment remains mandatory in attempt().
                state.selected = (edge.id,)
                self.priorities[edge.id] = problem.structural.evaluate(state)["score"]
                pairs[edge.a, edge.b].append(edge)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.rejections["invalid_proposal_ordering_warp"] += 1
        self.pairs = {pair: sorted(values, key=lambda e: (-self.priorities[e.id], e.id))[:config.structural_alternatives_per_pair]
                      for pair, values in pairs.items()}
        self.ordered_pairs = sorted(self.pairs, key=lambda p: (-self.priorities[self.pairs[p][0].id], p))

    def candidates_for(self, state):
        selected_pairs = {(self.edges[e].a, self.edges[e].b) for e in state.selected}
        pairs = [pair for pair in self.ordered_pairs if pair not in selected_pairs and state.groups[pair[0]] != state.groups[pair[1]]]
        return [self.pairs[p][depth] for depth in range(self.config.structural_alternatives_per_pair)
                for p in pairs if depth < len(self.pairs[p])]

    def coordinated_repair(self, beam, budget):
        config = self.config
        trace_start = len(self.trace)
        tasks = []
        # Every portfolio state and every cut receive a quota. Two-edge cuts
        # can undo mutually supporting bad choices without requiring a score
        # improvement after the first removal or replacement.
        for index, state in enumerate(beam[:config.portfolio_size]):
            cuts = [(eid,) for eid in state.selected]
            if config.structural_repair_removals > 1:
                pairs = list(combinations(state.selected, 2))
                pairs.sort(key=lambda pair: (sum(self.priorities.get(e, 0.) for e in pair), pair))
                cuts.extend(pairs[:max(1, len(state.selected))])
            for removed in cuts:
                tasks.append((index, state, removed))
        tasks.sort(key=lambda t: (len(t[2]), t[0], sum(self.priorities.get(e, 0.) for e in t[2]), t[2]))
        quota = max(1, config.max_repair_checks//max(1, sum(len(t[2]) for t in tasks)))
        checks, proposals = 0, list(beam)
        for index, source, removed in tasks:
            if checks >= config.max_repair_checks or not budget.take():
                break
            before = dict(self.rejections)
            rebuilt = self.rebuild(tuple(e for e in source.selected if e not in removed), budget)
            trace = {"portfolio_state": index, "removed_geometry_ids": list(removed), "score_before": source.score,
                     "replacement_geometry_ids_tried": [], "best_score_after": None, "rejection_counts": {}}
            if rebuilt is None:
                trace["status"] = "rebuild_rejected"
            else:
                frontier = [rebuilt]
                proposals.append(rebuilt)
                for depth in range(len(removed)):
                    next_frontier = []
                    # Round-robin over frontier states prevents the first
                    # branch from consuming the entire task allocation.
                    options = [self.candidates_for(s) for s in frontier]
                    consumed = 0
                    for step in range(quota):
                        if checks >= config.max_repair_checks:
                            break
                        for state, candidates in zip(frontier, options):
                            if step >= len(candidates) or consumed >= quota:
                                continue
                            if not budget.take():
                                break
                            edge = candidates[step]
                            checks += 1
                            consumed += 1
                            trace["replacement_geometry_ids_tried"].append(edge.id)
                            candidate = self.attempt(state, edge)
                            if candidate is not None:
                                next_frontier.append(candidate)
                        if consumed >= quota or budget.reason != "exhausted_candidates":
                            break
                    if not next_frontier:
                        break
                    frontier = retain(next_frontier, config.structural_repair_width)
                    proposals.extend(frontier)
                trace.update(status="checked", best_score_after=max(s.score for s in frontier))
            trace["rejection_counts"] = {k: self.rejections[k]-before.get(k, 0) for k in self.rejections if self.rejections[k] != before.get(k, 0)}
            self.trace.append(trace)
        return retain(proposals, config.beam_width), {"enabled": True, "checks": checks, "tasks_planned": len(tasks),
             "tasks_visited": len(self.trace)-trace_start, "per_depth_task_quota": quota, "maximum_removed_joins": config.structural_repair_removals,
             "truncated": len(self.trace)-trace_start < len(tasks) or checks >= config.max_repair_checks or budget.reason != "exhausted_candidates",
             "moves": "remove one/two joins, rebuild, bounded coordinated reconnect", "trace": self.trace}

    def solve(self):
        budget = Budget(self.config)
        beam = [singleton_state(self.problem)]
        accepted = processed = 0
        repair_info = {"enabled": False, "checks": 0}
        started = time.perf_counter()
        try:
            stopped = False
            for pair in self.ordered_pairs:
                alternatives = list(beam)
                for state in beam:
                    for edge in self.pairs[pair]:
                        if not budget.take():
                            stopped = True
                            break
                        candidate = self.attempt(state, edge)
                        if candidate is not None:
                            alternatives.append(candidate)
                            accepted += 1
                    if stopped:
                        break
                beam = retain(alternatives, self.config.beam_width)
                processed += 1
                if stopped:
                    break
            self.phase_seconds["pair_beam"] = time.perf_counter()-started
            repair_started = time.perf_counter()
            if self.config.repair_rounds and not stopped:
                for _ in range(self.config.repair_rounds):
                    beam, repair_info = self.coordinated_repair(beam, budget)
                    if budget.reason != "exhausted_candidates":
                        break
            self.phase_seconds["coordinated_repair"] = time.perf_counter()-repair_started
        except KeyboardInterrupt:
            budget.reason = "interrupted"
            repair_info["interrupted"] = True
        candidates = retain([*beam, singleton_state(self.problem)], self.config.beam_width+1)
        return SearchResult(candidates, {"proposals": budget.proposals, "accepted_proposals": accepted,
            "processed_physical_pairs": processed, "candidate_edges": len(self.problem.edges),
            "searched_edges": sum(map(len, self.pairs.values())), "final_search_states": len(beam),
            "assignment_nodes": self.assignment_nodes, "invalid_support_occurrences": len(self.support.invalid),
            "duplicate_support_options_skipped": self.support.duplicate_options_skipped,
            "loop_search": {"enabled": False, "reason": "pair-wise beam with soft unselected-pair corroboration; no exhaustive loop seed enumeration"},
            "repair": repair_info, "phase_seconds": self.phase_seconds,
            "proposal_shortlist_policy": "observable structural score, bounded alternatives per physical pair",
            "shortlisted_geometry_ids": {" / ".join(p): [e.id for e in v] for p, v in self.pairs.items()}}, budget.reason, dict(self.rejections))
