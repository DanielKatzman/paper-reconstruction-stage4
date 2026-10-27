"""Bounded exact proposal reuse; never caches an objective across policies."""
from collections import Counter, OrderedDict
import copy
import json
import time
import numpy as np
from ..model import State
from .common import objective


def clone(state):
    return State({p:h.copy() for p,h in state.poses.items()}, dict(state.groups),
        tuple(state.selected), tuple(state.tree), dict(state.assignments), state.score,
        copy.deepcopy(state.active_support))


def state_key(state):
    return (tuple((p,state.groups[p],h.dtype.str,h.shape,h.tobytes()) for p,h in sorted(state.poses.items())),
        tuple(state.selected),tuple(state.tree),tuple(sorted(state.assignments.items())),
        json.dumps(state.active_support,sort_keys=True,separators=(',',':'),allow_nan=False) if state.active_support else '')


class ExactReuse:
    def __init__(self, config):
        self.capacity = min(config.maximum_cache_entries,4096)
        self.cache = OrderedDict()
        self.strata = {}
        self.counts = Counter()
        self.seconds = Counter()

    def call(self, phase, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args,**kwargs)
        finally:
            self.seconds[phase] += time.perf_counter()-started
            self.counts[phase+'_calls'] += 1

    def attempt(self, search, state, edge, uncached):
        started = time.perf_counter()
        # The shared context belongs to one solve and one immutable physical /
        # support index. Cycle tolerance is read directly by PartialSearch.
        key = (id(search.feasibility),id(search.support),search.config.cycle_tolerance_px,
               edge.id,state_key(state))
        self.seconds['cache_key'] += time.perf_counter()-started
        self.counts['logical_attempts'] += 1
        if key in self.cache:
            candidate,rejections,nodes = self.cache[key]
            self.cache.move_to_end(key)
            self.counts['cache_hits'] += 1
            search.rejections.update(rejections)
            search.assignment_nodes += nodes
            self.counts['assignment_nodes_reused'] += nodes
            if candidate is None:
                self.counts['rejected_attempt_hits'] += 1
                return None
            result = clone(candidate)
            # Geometry/support decisions can be shared by v4 and v6. Scores
            # cannot: always evaluate under the requesting arm's objective.
            try:
                result.score = self.call('score_replay',objective,result,search.problem,search.edges,search.config)
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                # Match PartialSearch's failure behavior. A scoring exception
                # in one policy must not invalidate geometry in another.
                search.rejections['invalid_composed_transform'] += 1
                return None
            return result
        self.counts['cache_misses'] += 1
        previous = Counter(search.rejections)
        nodes = search.assignment_nodes
        candidate = self.call('uncached_attempt',uncached,state,edge)
        rejected = search.rejections-previous
        if candidate is None and rejected.get('invalid_composed_transform'):
            self.counts['exception_results_not_cached'] += 1
            return None
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.counts['evictions'] += 1
        self.cache[key] = (None if candidate is None else clone(candidate),rejected,search.assignment_nodes-nodes)
        self.counts['peak_entries'] = max(self.counts['peak_entries'],len(self.cache))
        return candidate

    def report(self):
        return {'policy':'exact-reuse.v1','cache_capacity':self.capacity,
            'counts':dict(self.counts),'phase_seconds':dict(self.seconds),
            'pose_keys':'exact bytes; groups, selections, tree, assignments and full active records',
            'objective_scores_recomputed_on_hit':True,'proposal_budget_counts_cache_hits':True,
            'initial_result_seeds':False,'evaluation_truth_opened':False,
            'timing_scope':'Nested cumulative phases; not additive wall-time partitions',
            'support_statistics_scope':'Actual support computations; logical assignment nodes include replay'}
