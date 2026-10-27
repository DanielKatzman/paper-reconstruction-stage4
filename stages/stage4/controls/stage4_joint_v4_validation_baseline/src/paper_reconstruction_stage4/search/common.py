from dataclasses import dataclass, field
import time
import numpy as np
from ..geometry import normalize, pose_error, relative


class Budget:
    def __init__(self, config):
        self.config = config
        self.start = time.perf_counter()
        self.proposals = 0
        self.reason = "exhausted_candidates"

    def take(self):
        if self.proposals >= self.config.max_proposals:
            self.reason = "proposal_budget"
            return False
        if time.perf_counter() - self.start >= self.config.max_seconds:
            self.reason = "time_budget"
            return False
        self.proposals += 1
        return True


@dataclass
class SearchResult:
    states: list
    diagnostics: dict
    termination: str
    rejections: dict = field(default_factory=dict)


def identity(state):
    value = (tuple((p, state.groups[p], tuple(np.round(normalize(state.poses[p]).ravel(), 6))) for p in sorted(state.poses)),
             tuple(sorted(state.selected)), tuple(sorted(state.assignments.items())))
    return (*value, tuple(sorted((e, a["support_id"]) for e, a in state.active_support.items()))) if state.active_support else value


def objective(state, problem, edges, config):
    if config.objective_version in {"observable-structure.v2", "observable-joint.v3", "observable-joint.v4"}:
        if problem.structural is None:
            raise ValueError("Structural objective has not been initialized")
        return problem.structural.evaluate(state)["score"]
    gain = sum(float(np.clip(edges[eid].weight, -1., 1.)) - config.join_cost for eid in state.selected)
    residual = 0.
    for eid in state.selected:
        edge = edges[eid]
        err = pose_error(relative(state.poses, edge.a, edge.b), edge.transform, problem.pieces[edge.b].shape)
        residual += min((err / max(config.cycle_tolerance_px, 1e-9)) ** 2, 4.)
    return gain - .05 * residual


def retain(states, width):
    unique = {}
    for state in states:
        key = identity(state)
        if key not in unique or state.score > unique[key].score:
            unique[key] = state
    return sorted(unique.values(), key=lambda state: (-state.score, identity(state)))[:width]
