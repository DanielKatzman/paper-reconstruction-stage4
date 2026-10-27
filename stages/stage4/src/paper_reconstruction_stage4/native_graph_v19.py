"""Try nonstraight native growth before reopening the complete pose bank.

Almost perfectly straight contours often fit exterior page borders with very
small residuals. That is ambiguous placement evidence, not a confident seam.
This route defers them during one fast ordering pass; it never removes them
from native fitting, final support assignment, or subsequent search.
"""
from collections import Counter
import math
import time
from .native_graph_v18 import search as base_search

MINIMUM_SPREAD = .001


def informative(proposal):
    spread = float(proposal.get('shape_spread_ratio', 0.))
    return math.isfinite(spread) and spread >= MINIMUM_SPREAD


def search(context, basis, proposals, deadline, on_complete, physical=None, fast_only=False, on_idle=None):
    started = time.perf_counter()
    distinct = [p for p in proposals if informative(p)]
    lanes = []
    # A finite fast pass chooses ordering; the common overall deadline remains
    # the only time ceiling. The all-mode route can still use real straight cuts.
    if distinct and len(distinct) < len(proposals) and time.perf_counter() < deadline:
        candidate, trace = base_search(context, basis, distinct, deadline, on_complete,
            physical, fast_only=True, on_idle=on_idle)
        trace['growth_lane'] = 'nonstraight_first'
        lanes.append(trace)
    else:
        candidate = None
    if candidate is None:
        candidate, trace = base_search(context, basis, proposals, deadline, on_complete,
            physical, fast_only=fast_only, on_idle=on_idle)
        trace['growth_lane'] = 'all_modes_fallback'
        lanes.append(trace)
    # Preserve the old top-level audit shape for existing renderers and coverage
    # readers; nested lanes retain every unaccepted growth trace separately.
    audit = dict(lanes[-1]); rejected = Counter()
    for lane in lanes:
        rejected.update(lane['rejections'])
    for field in ('merge_checks', 'native_proposal_replay_checks', 'final_candidates_checked',
                  'complete_shape_candidates', 'backtracks', 'backtracking_seed_sessions_started'):
        audit[field] = sum(lane[field] for lane in lanes)
    deepest = max(lane['completed_depth'] for lane in lanes)
    audit.update(policy='geometry-checked-straight-deferred.v1',
        growth_policy=context.config.native_growth_policy, minimum_growth_spread=MINIMUM_SPREAD,
        full_proposal_count=len(proposals), nonstraight_proposal_count=len(distinct),
        deferred_straight_proposal_count=len(proposals)-len(distinct),
        straight_modes_permanently_discarded=False, final_support_pool_filtered=False,
        final_validation_changed=False, growth_lanes=lanes, rejections=dict(rejected),
        completed_depth=deepest, elapsed_seconds=time.perf_counter()-started,
        observed_growth_states=[r for lane in lanes for r in lane['observed_growth_states']])
    return candidate, audit
