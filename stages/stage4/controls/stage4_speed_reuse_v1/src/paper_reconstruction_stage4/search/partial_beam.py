from collections import Counter
import numpy as np
from ..geometry import normalize, pose_error, regauge, relative
from ..model import State, singleton_state
from .common import Budget, SearchResult, identity, objective, retain


class PartialSearch:
    def __init__(self, problem, config, feasibility, support):
        self.problem, self.config = problem, config
        self.feasibility, self.support = feasibility, support
        self.edges = {e.id: e for e in problem.edges}
        self.rejections = Counter()
        self.assignment_nodes = 0

    def attempt(self, state, edge):
        execution = getattr(self,'execution',None)
        if edge.id in state.selected:
            return None
        if any((self.edges[e].a, self.edges[e].b) == (edge.a, edge.b) for e in state.selected):
            self.rejections["competing_pair_geometry"] += 1
            return None
        a, b = edge.a, edge.b
        ga, gb = state.groups[a], state.groups[b]
        poses, groups = dict(state.poses), dict(state.groups)
        joining = ga != gb
        try:
            if joining:
                first = [p for p in poses if groups[p] == ga]
                second = [p for p in poses if groups[p] == gb]
                move = poses[a] @ edge.transform @ np.linalg.inv(poses[b])
                poses.update({p: normalize(move @ poses[p]) for p in second})
                poses = regauge(poses, set(first + second))
                root = min(first + second)
                groups.update({p: root for p in first + second})
                for x in first:
                    for y in second:
                        h = relative(poses, x, y)
                        report = (execution.call('physical_checks',self.feasibility.pair,x,y,h)
                            if execution is not None else self.feasibility.pair(x,y,h))
                        if not report["valid"]:
                            self.rejections[report["reason"]] += 1
                            return None
            else:
                forward = pose_error(relative(poses, a, b), edge.transform, self.problem.pieces[b].shape)
                backward = pose_error(relative(poses, b, a), np.linalg.inv(edge.transform), self.problem.pieces[a].shape)
                if max(forward, backward) > self.config.cycle_tolerance_px:
                    self.rejections["selected_cycle_inconsistent"] += 1
                    return None
            selected = tuple(sorted((*state.selected, edge.id)))
            assignments, reason, nodes = (execution.call('support_assignment',self.support.assign,selected,poses,self.edges)
                if execution is not None else self.support.assign(selected, poses, self.edges))
            self.assignment_nodes += nodes
            if assignments is None:
                self.rejections[reason] += 1
                return None
            candidate = State(poses, groups, selected, (*state.tree, edge.id) if joining else state.tree, assignments)
            if hasattr(self.support, "bind"):
                self.support.bind(candidate)
            candidate.score = (execution.call('scoring',objective,candidate,self.problem,self.edges,self.config)
                if execution is not None else objective(candidate, self.problem, self.edges, self.config))
            return candidate
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            self.rejections["invalid_composed_transform"] += 1
            return None

    def rebuild(self, edge_ids, budget=None):
        state = singleton_state(self.problem)
        # A deterministic spanning forest is reconstructed, then cycle supports.
        remaining = sorted(edge_ids, key=lambda eid: (-self.edges[eid].weight, eid))
        for eid in remaining:
            if budget is not None and not budget.take():
                return None
            state = self.attempt(state, self.edges[eid])
            if state is None:
                return None
        return state

    def solve(self):
        budget = Budget(self.config)
        beam = [singleton_state(self.problem)]
        loop_info = {"enabled": self.config.loop_proposals}
        accepted = processed = 0
        try:
            if self.config.loop_proposals:
                from .loops import loop_seeds
                seeds, loop_info = loop_seeds(self, budget)
                beam = retain([*beam, *seeds], self.config.beam_width)
            stopped = False
            for edge in self.problem.edges:
                alternatives = list(beam)
                for state in beam:
                    if not budget.take():
                        stopped = True
                        break
                    candidate = self.attempt(state, edge)
                    if candidate is not None:
                        accepted += 1
                        alternatives.append(candidate)
                beam = retain(alternatives, self.config.beam_width)
                processed += 1
                if stopped:
                    break
            if self.config.repair_rounds and not stopped:
                from .repair import repair
                beam, repair_info = repair(self, beam, budget)
            else:
                repair_info = {"enabled": False, "checks": 0}
        except KeyboardInterrupt:
            budget.reason = "interrupted"
            repair_info = {"enabled": self.config.repair_rounds > 0, "interrupted": True}
        # The always-feasible singleton candidate remains available if final
        # high-resolution checks reject every coarse-search finalist.
        candidates = retain([*beam, singleton_state(self.problem)], self.config.beam_width + 1)
        return SearchResult(candidates, {"proposals": budget.proposals, "accepted_proposals": accepted,
                           "processed_edges": processed, "candidate_edges": len(self.problem.edges),
                           "final_search_states": len(beam), "assignment_nodes": self.assignment_nodes,
                           "invalid_support_occurrences": len(self.support.invalid), "loop_search": loop_info,
                           "duplicate_support_options_skipped": self.support.duplicate_options_skipped,
                           "repair": repair_info, "deterministic_order": "descending weight, physical pair, geometry ID"},
                            budget.reason, dict(self.rejections))
