"""Bounded diverse constructions and ambiguity-directed multi-join rebuilds.

All rankings use native observations/incoming evidence. No reference-picked
geometry, physical neighbor list or exact main-section constraint is accepted.
"""
from collections import Counter, defaultdict
import time
import numpy as np
from ..geometry import normalize
from ..model import State, singleton_state
from .common import Budget, SearchResult, identity
from .partial_beam import PartialSearch


def topology(state, edges):
    return tuple(sorted((edges[e].a, edges[e].b) for e in state.tree))


def diverse_retain(states, width, edges):
    unique = {}
    for state in states:
        key = identity(state)
        if key not in unique or state.score > unique[key].score:
            unique[key] = state
    ranked = sorted(unique.values(), key=lambda s: (-s.score, identity(s)))
    if len(ranked) <= width:
        return ranked
    # Preserve the objective winner, distinct backbones and complete competitors.
    selected, used, buckets = [], set(), defaultdict(list)
    for s in ranked:
        buckets[(len(s.components()), topology(s, edges))].append(s)
    for s in ranked[:max(1, width//3)]:
        selected.append(s); used.add(identity(s))
    complete = [s for s in ranked if len(s.components()) == 1]
    for s in complete[:max(1, width//4)]:
        if identity(s) not in used and len(selected) < width:
            selected.append(s); used.add(identity(s))
    depth = 0
    ordered = sorted(buckets, key=lambda k: (-buckets[k][0].score, k))
    while len(selected) < width:
        added = False
        for key in ordered:
            values = buckets[key]
            if depth < len(values) and identity(values[depth]) not in used:
                selected.append(values[depth]); used.add(identity(values[depth])); added = True
                if len(selected) == width:
                    break
        depth += 1
        if not added and all(depth >= len(v) for v in buckets.values()):
            break
    return sorted(selected, key=lambda s: (-s.score, identity(s)))


class JointSearch(PartialSearch):
    def __init__(self, problem, config, feasibility, support):
        super().__init__(problem, config, feasibility, support)
        self.rng = np.random.default_rng(config.seed)
        self.priorities, pairs, self.strata = {}, defaultdict(list), {}
        self.phase_seconds, self.trace, self.completed = {}, [], []
        started = time.perf_counter()
        for edge in problem.edges:
            try:
                proto = State({edge.a: np.eye(3), edge.b: normalize(edge.transform)}, {edge.a: edge.a, edge.b: edge.a}, (edge.id,))
                self.priorities[edge.id] = problem.structural.evaluate(proto)["score"]
                pairs[edge.a, edge.b].append(edge)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.rejections["invalid_proposal_ordering_warp"] += 1
        self.pairs = {}
        self.shortlist_audit = {}
        for pair, values in sorted(pairs.items()):
            ranked = sorted(values, key=lambda e: (-self.priorities[e.id], e.id))
            for edge in ranked:
                h = edge.transform
                diagonal = max(1., np.linalg.norm(problem.pieces[edge.a].shape))
                pose = (int(np.floor(np.arctan2(h[1, 0], h[0, 0])/np.deg2rad(15.))),
                        int(np.floor(np.log(max(np.linalg.norm(h[:2, 0]), 1e-9))/.1)),
                        *np.floor(h[:2, 2]/(.05*diagonal)).astype(int).tolist())
                supports = tuple(sorted({tuple((a.piece, tuple((round(x/64), round(y/64)) for x, y in a.intervals))
                                               for a in support.arcs[o["occurrence_id"]])
                                         for o in edge.occurrences if o["occurrence_id"] in support.arcs}))
                self.strata[edge.id] = (pose, supports)
            chosen = ranked[:min(4, config.joint_alternatives_per_pair)]
            remaining = [e for e in ranked if e not in chosen]
            while remaining and len(chosen) < config.joint_alternatives_per_pair:
                poses, regions = {self.strata[e.id][0] for e in chosen}, {self.strata[e.id][1] for e in chosen}
                candidate = max(remaining, key=lambda e: (int(self.strata[e.id][0] not in poses)+int(self.strata[e.id][1] not in regions),
                                                         self.priorities[e.id], e.id))
                chosen.append(candidate); remaining.remove(candidate)
            self.pairs[pair] = chosen
            self.shortlist_audit[" / ".join(pair)] = {"retained_geometry_ids": [e.id for e in chosen],
                "discarded_geometry_ids": [e.id for e in remaining], "available": len(ranked),
                "pose_strata_retained": len({self.strata[e.id][0] for e in chosen}),
                "support_strata_retained": len({self.strata[e.id][1] for e in chosen})}
        self.candidates = [e for values in self.pairs.values() for e in values]
        self.phase_seconds["stratification"] = time.perf_counter()-started

    def remember(self, state, origin):
        if len(state.components()) == 1:
            self.completed.append({"origin": origin, "score": state.score, "selected_geometry_ids": list(state.selected),
                "occurrence_ids": dict(state.assignments), "active_support": state.active_support,
                "native_to_component": {p: h.tolist() for p, h in state.poses.items()}})
            # A bounded audit, not an unbounded archive of all serving work.
            if len(self.completed) > 128:
                self.completed.sort(key=lambda v: (-v["score"], v["selected_geometry_ids"]))
                del self.completed[128:]

    def ordered_candidates(self, state, noise=0.):
        values = [e for e in self.candidates if state.groups[e.a] != state.groups[e.b]]
        perturbed = {e.id: self.priorities[e.id]+noise*self.rng.gumbel() for e in values}
        return sorted(values, key=lambda e: (-perturbed[e.id], e.id))

    def regions(self, state):
        selected = list(state.tree)
        if not selected:
            return []
        ordered = sorted(selected, key=lambda e: (self.priorities[e], e))
        regions = []
        # Grow ambiguous neighborhoods across both endpoints of a weak join.
        for seed in ordered:
            removed, pieces = [seed], {self.edges[seed].a, self.edges[seed].b}
            while len(removed) < min(self.config.joint_region_joins, len(selected)):
                eligible = [e for e in ordered if e not in removed and ({self.edges[e].a, self.edges[e].b} & pieces)]
                if not eligible:
                    break
                next_edge = eligible[0]
                removed.append(next_edge); pieces.update((self.edges[next_edge].a, self.edges[next_edge].b))
            key = tuple(sorted(removed))
            if key not in regions:
                regions.append(key)
            if len(regions) >= 4:
                break
        return regions

    def rebuild_regions(self, pool, budget):
        tasks = [(s, region) for s in pool[:self.config.portfolio_size] for region in self.regions(s)]
        quota = max(1, self.config.max_repair_checks//max(1, len(tasks)))
        checks, visited = 0, 0
        for source, removed in tasks:
            if budget.reason != "exhausted_candidates" or checks >= self.config.max_repair_checks:
                break
            visited += 1
            record = {"removed_geometry_ids": list(removed), "source_score": source.score,
                      "depth_counts": [], "complete_candidates": 0, "rejections": {}}
            before = Counter(self.rejections)
            seed = self.rebuild(tuple(e for e in source.selected if e not in removed), budget)
            if seed is None:
                record["status"] = "base_rebuild_rejected"; self.trace.append(record); continue
            frontier, consumed = [seed], 0
            for depth in range(len(removed)):
                next_frontier = []
                # Allocate the same bounded amount at each depth, across states.
                per_depth = max(1, quota//len(removed))
                options = [self.ordered_candidates(s, .03 if depth else 0.) for s in frontier]
                tested = 0
                step = 0
                while tested < per_depth and any(step < len(v) for v in options):
                    for state, candidates in zip(frontier, options, strict=True):
                        if tested >= per_depth or step >= len(candidates):
                            continue
                        if not budget.take():
                            break
                        candidate = self.attempt(state, candidates[step]); tested += 1; checks += 1; consumed += 1
                        if candidate is not None:
                            next_frontier.append(candidate)
                            if len(candidate.components()) == 1:
                                self.remember(candidate, "region_rebuild")
                                record["complete_candidates"] += 1
                    if budget.reason != "exhausted_candidates":
                        break
                    step += 1
                record["depth_counts"].append({"depth": depth+1, "tested": tested, "valid": len(next_frontier)})
                if not next_frontier:
                    break
                frontier = diverse_retain(next_frontier, self.config.joint_rebuild_width, self.edges)
                pool = diverse_retain([*pool, *frontier], self.config.beam_width, self.edges)
                if budget.reason != "exhausted_candidates":
                    break
            record.update(status="checked", checks=consumed, best_score=max(s.score for s in frontier),
                          rejections=dict(self.rejections-before))
            self.trace.append(record)
        return pool, {"tasks_planned": len(tasks), "tasks_visited": visited, "checks": checks, "quota_per_task": quota,
                      "maximum_removed_joins": self.config.joint_region_joins, "trace": self.trace,
                      "truncated": visited < len(tasks) or budget.reason != "exhausted_candidates"}

    def solve(self):
        budget = Budget(self.config)
        if self.problem.deadline is not None:
            # Include proposal setup/loading in the enclosing consumer budget.
            reserve = self.config.joint_finalize_seconds+(self.config.refine_seconds if self.config.refine else 0.)
            budget.start = min(budget.start, self.problem.deadline-reserve-self.config.max_seconds)
        pool, controls, accepted, trials = [singleton_state(self.problem)], [], 0, 0
        for source in self.problem.initial_states:
            assigned, reason, nodes = self.support.assign(source.selected, source.poses, self.edges)
            self.assignment_nodes += nodes
            if assigned is not None and self.feasibility.state(source, final=True)["valid"]:
                source.assignments = assigned
                if hasattr(self.support, "bind"):
                    self.support.bind(source)
                source.score = self.problem.structural.evaluate(source)["score"]
                controls.append(source)
                pool.append(source); self.remember(source, "authenticated_control")
            else:
                self.rejections["initial_state_rejected_"+reason] += 1
        started = time.perf_counter()
        repair = {"enabled": False}
        try:
            for trial in range(self.config.joint_trials):
                if budget.reason != "exhausted_candidates":
                    break
                state = singleton_state(self.problem)
                # Independent complete constructions allow arbitrary coordinated
                # changes; no requirement to improve after each intermediate join.
                order = self.ordered_candidates(state, 0. if trial == 0 else .04+.025*(trial%8))
                for edge in order:
                    if state.groups[edge.a] == state.groups[edge.b]:
                        continue
                    if not budget.take():
                        break
                    candidate = self.attempt(state, edge)
                    if candidate is not None:
                        state = candidate; accepted += 1
                        if len(state.components()) == 1:
                            break
                trials += 1
                self.remember(state, "construction_"+str(trial))
                pool = diverse_retain([*pool, state], self.config.beam_width, self.edges)
            self.phase_seconds["diverse_constructions"] = time.perf_counter()-started
            started = time.perf_counter()
            if self.config.repair_rounds and budget.reason == "exhausted_candidates":
                pool, repair = self.rebuild_regions(pool, budget)
            self.phase_seconds["region_rebuilds"] = time.perf_counter()-started
        except KeyboardInterrupt:
            budget.reason = "interrupted"
        pool = diverse_retain(pool, self.config.beam_width, self.edges)
        counts = Counter(topology(s, self.edges) for s in pool)
        # Coarse candidates may displace a valid control from the beam and then
        # fail final checks. Keep authenticated controls outside that pruning.
        finalists = {identity(s): s for s in [*pool, *controls, singleton_state(self.problem)]}
        finalists = sorted(finalists.values(), key=lambda s: (-s.score, identity(s)))
        return SearchResult(finalists, {"proposals": budget.proposals, "accepted_proposals": accepted, "construction_trials": trials,
            "candidate_edges": len(self.problem.edges), "searched_edges": len(self.candidates), "assignment_nodes": self.assignment_nodes,
            "invalid_support_occurrences": len(self.support.invalid), "proposal_strata": self.shortlist_audit,
            "phase_seconds": self.phase_seconds, "repair": repair, "retained_backbones": len(counts),
            "complete_candidate_records": self.completed, "complete_candidate_record_limit": 128,
            "retained_complete_layouts": sum(len(s.components()) == 1 for s in pool),
            "protected_feasible_control_states": len(controls),
            "pruning": {"beam_width": self.config.beam_width, "selection": "objective winner, complete competitors and backbone diversity",
                        "exhaustion_means": "declared bounded constructions/regions; not exhaustive global search"},
            "evaluation_truth_opened": False}, budget.reason, dict(self.rejections))
