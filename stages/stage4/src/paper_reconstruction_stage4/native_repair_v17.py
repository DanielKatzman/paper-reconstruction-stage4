"""Adaptive stopping, legacy general ordering and an independent native route."""
import time
from collections import defaultdict, OrderedDict
import numpy as np
from paper_reconstruction_contracts import map_points
from .page import enclosing_rectangle
from .native_repair_v16 import RepairContext as PreviousRepairContext
from .component_closure import ClosureContext, shape_measurement
from .page_geometry import fingerprint, state_record
from .model import singleton_state
from .outputs import serialize_layout
from .active_support import verify_result_support
from .feasibility import Feasibility
from .seam_poses_v15 import generate
from .repair_sources_v16 import alternative_sources
from .repair_search_v17 import search as component_search
from .native_graph_v17 import search as native_search


def diverse_completions(context, complete):
    """Check different poses first, then a second support tree per failed pose."""
    families = defaultdict(list)
    for row in complete:
        family = context.pose_family(row[0].state)
        if len(families[family]) < 2:
            families[family].append(row)
    ordered = [row for layer in range(2) for rows in families.values() for row in rows[layer:layer+1]]
    return ordered, {'available': len(complete), 'pose_families': len(families),
                     'maximum_support_variants_per_pose': 2, 'ordered_checks': len(ordered)}


def satisfactory(candidate, verdict):
    # A stricter stopping heuristic only: acceptance still uses the established
    # page/native gates. A lower-scoring verified page remains in the pool.
    shape = verdict.get('geometry', {})
    return bool(candidate is not None and len(candidate.components()) == 1 and
                verdict.get('disposition') == 'verified_completion' and
                shape.get('shape_gate_pass') is True and shape.get('score', 0.) >= .98)


class RepairContext(PreviousRepairContext):
    def __init__(self, problem, config, support):
        super().__init__(problem, config, support)
        self.component_rectangle_cache = OrderedDict()
        self.repair_audit.update(component_bound_policy='adaptive-native-graph-portfolio.v1',
            general_ordering='legacy-score-order-16-successors.v1',
            independent_native_route=True, source_pose_records_saved=True,
            early_stop_policy='independently-verified-page-score-0.98.v1',
            early_stopped=False, route_stop_reason=None)

    def component_bounds(self, state):
        """Reuse exact immutable component rectangles, including singletons."""
        started = time.perf_counter()
        key = self.pose_key(state)
        if key in self.bound_cache:
            self.counts['component_bound_cache_hits'] += 1
            self.bound_cache.move_to_end(key)
            self.ambiguity_seconds['component_bounds_calls'] += time.perf_counter()-started
            return self.bound_cache[key]
        total = sum(self.areas.values()); limit = 1.+self.config.page_component_area_slack; rows = []
        for group in state.components():
            component_key = tuple((p, state.poses[p].dtype.str, state.poses[p].shape, state.poses[p].tobytes()) for p in group)
            if component_key in self.component_rectangle_cache:
                area = self.component_rectangle_cache[component_key]
                self.component_rectangle_cache.move_to_end(component_key)
                self.counts['component_rectangle_cache_hits'] += 1
            else:
                points = np.vstack([map_points(self.hulls[p], state.poses[p]) for p in group])
                area = enclosing_rectangle(points)[0]
                self.component_rectangle_cache[component_key] = area
                self.counts['component_rectangle_extractions'] += 1
                if len(self.component_rectangle_cache) > 4096:
                    self.component_rectangle_cache.popitem(last=False)
            ratio = float(area/total)
            rows.append({'pieces': group, 'minimum_rectangle_area_px2': area, 'ratio_to_total_observed_area': ratio,
                         'valid': ratio <= limit})
        value = {'valid': all(row['valid'] for row in rows), 'components': rows,
            'total_observed_area_px2': total, 'maximum_area_ratio': limit, 'orientation_freely_fitted': True,
            'partial_fill_required': False, 'independent_component_rectangles_summed': False}
        self.bound_cache[key] = value
        if len(self.bound_cache) > 1024:
            self.bound_cache.popitem(last=False)
        self.counts['component_bound_extractions'] += 1
        self.ambiguity_seconds['component_bounds_calls'] += time.perf_counter()-started
        return value

    def source_record(self, source, removed, kind):
        return {'candidate_id': fingerprint(source), 'groups': source.components(),
            'released_geometry_ids': removed, 'kind': kind, 'state': state_record(source),
            'selected_native_bridge_records': [self.edges[e].raw['native_bridge'] for e in source.selected
                                               if 'native_bridge' in self.edges[e].raw]}

    def augment_finalists(self, states, deadline):
        states = ClosureContext.augment_finalists(self, states, deadline)
        started = time.perf_counter(); stop = min(started+self.config.repair_seconds, deadline-8.)
        physical = Feasibility(self.problem, self.config)
        for state in states:
            if len(state.components()) != 1 or not shape_measurement(self, state, 256)['shape_gate_pass'] or self.closure_priority(state, self.features(state)) <= 0:
                continue
            if not physical.state(state, final=True)['valid']:
                continue
            try:
                verify_result_support(self.problem, {'layouts': [serialize_layout(self.problem, state, 0, {'valid': True}, {'attempted': False})]}, self.config)
            except ValueError:
                continue
            self.repair_audit.update(protected_supported_completion=True, route_stop_reason='protected_verified_completion')
            return self.finish(states, started, stop)
        if not self.problem.capture_metadata or self.problem.capture_metadata.get('rotation_family') != 'quarter_turn':
            self.repair_audit['unavailable_reason'] = 'capture_does_not_declare_quarter_turns'
            return self.finish(states, started, stop)
        # Build a fresh identity-gauge source even when every legacy source is
        # excluded. It carries no old poses, assignments, selected joins or arcs.
        basis = singleton_state(self.problem)
        from .native_repair import RepairContext as LegacyContext
        originals, omitted = LegacyContext.sources(self, states)
        allowance = max(0, getattr(self, 'finalization_proposal_allowance', 0)-len(self.deletion_audit['checks'])-self.closure_audit['generated_proposals'])
        if allowance == 0 or time.perf_counter() >= stop:
            self.repair_audit.update(budget_abstained=True, route_stop_reason='no_remaining_allowance')
            return self.finish(states, started, stop)
        frontiers = [s.components() for s, _, _ in originals] or [basis.components()]
        proposals, generation = generate(self.problem, self.config, min(stop, started+self.config.repair_seconds*.25),
            maximum_checks=min(allowance, self.config.repair_native_checks), frontiers=frontiers)
        allowance -= generation['native_checks']
        self.repair_audit.update(native_generation=generation, generated_pose_proposals=proposals,
            shared_native_generation_passes=1, proposal_checks=generation['native_checks'], sources_omitted_by_limit=omitted)
        sources = [*originals[:1], (basis, [], 'independent_native_singletons')]
        original_index = 0 if originals else None; basis_index = 1 if originals else 0
        if originals and self.config.repair_source_limit > 2 and time.perf_counter() < stop:
            alternatives, conditioning = alternative_sources(self, originals, proposals, min(stop, time.perf_counter()+1.))
            self.repair_audit['core_conditioning'] = conditioning
            for alternative in alternatives:
                if alternative[2].startswith('native_conflict_'):
                    sources.append(alternative); break
        sources = sources[:self.config.repair_source_limit]
        self.repair_audit.update(sources=[self.source_record(*row) for row in sources], scheduled_source_count=len(sources),
                                maximum_components=len(basis.poses), maximum_components_scope='independent route only; general source eligibility unchanged')
        jobs = []
        if original_index is not None:
            jobs.append((original_index, 'conditional_extent'))
        main = [(original_index, 'observed_area')] if original_index is not None else []
        native = [(basis_index, 'independent_native_graph')]
        jobs.extend([*native, *main] if len(basis.poses) > 12 else [*main, *native])
        if len(sources) > basis_index+1:
            jobs.append((basis_index+1, 'split_area_fallback'))
        # Two principal routes have separate work, while a hard total and the
        # outer proposal/time limits bound the whole recovery procedure.
        merge_left = min(allowance, 2*self.config.repair_merge_checks)
        initial_merge = merge_left; extra = []; checked = 0; attempted = set(); stopped = False
        successful_families = set()
        for index, lane in jobs:
            if stopped:
                self.repair_audit['jobs'].append({'source_index': index, 'lane': lane, 'status': 'skipped_verified_page'})
                continue
            now = time.perf_counter()
            if now >= stop or merge_left <= 0:
                self.repair_audit['jobs'].append({'source_index': index, 'lane': lane, 'status': 'unattempted_budget'})
                continue
            source = sources[index][0]; remaining = stop-now
            if lane == 'conditional_extent':
                seconds = min(16., remaining*.3); checks = min(7500, self.config.repair_merge_checks, merge_left)
            elif lane == 'observed_area' and len(basis.poses) <= 12:
                seconds = min(40., max(0., remaining-10.)) if remaining > 15. else remaining*.7
                checks = min(self.config.repair_merge_checks, max(1, merge_left-2000))
            elif lane == 'independent_native_graph' and len(basis.poses) > 12 and main:
                seconds = remaining*.75; checks = min(self.config.repair_merge_checks, max(1, int(merge_left*.75)))
            else:
                seconds = remaining*.85 if lane == 'independent_native_graph' and len(sources) > 2 else remaining
                checks = min(self.config.repair_merge_checks, merge_left)
            checks = min(checks, merge_left)
            job_stop = min(stop, now+seconds); search_stop = max(now, job_stop-min(3., seconds*.2))
            attempted.add(index)
            if lane == 'independent_native_graph':
                complete, trace = native_search(self, source, proposals, search_stop, checks, self.config.repair_beam, physical)
            else:
                complete, trace = component_search(self, source, proposals, search_stop, checks,
                    self.config.repair_beam, area_bound=lane != 'conditional_extent', physical=physical)
            trace.update(source_candidate_id=fingerprint(source), lane=lane, reserved_seconds=seconds,
                         finalization_reserved_seconds=job_stop-search_stop)
            from .native_bridge import arc_value
            trace['coarse_complete_records'] = [{'state': state_record(node.state), 'geometry': shape, 'candidate_id': fingerprint(node.state),
                'contacts': [{**{k: v for k, v in row.items() if k != 'arcs'},
                              'arcs': [arc_value(a) for a in row['arcs']]} for row in node.contacts],
                'accepted_assembly': False} for node, shape in complete[:self.config.page_top_k]]
            trace['coarse_records_omitted_by_limit'] = max(0, len(complete)-self.config.page_top_k)
            self.repair_audit['merge_searches'].append(trace)
            merge_left -= trace['merge_checks']; self.repair_audit['proposal_checks'] += trace['merge_checks']
            self.repair_deadline = job_stop
            ordered, diversity = diverse_completions(self, complete)
            final_quota = 4 if lane == 'conditional_extent' else 8
            attempted_finals = 0; satisfactory_found = False
            for node, shape in ordered:
                family = self.pose_family(node.state)
                if family in successful_families:
                    continue
                if checked >= self.config.repair_complete_checks or attempted_finals >= final_quota or time.perf_counter() >= job_stop:
                    break
                checked += 1; attempted_finals += 1
                try:
                    candidate, verdict = self.finalize(source, node, physical)
                except ValueError as exc:
                    candidate = None; verdict = {'disposition': 'rejected_native_support', 'reason': str(exc),
                        'diagnostic_state': state_record(node.state), 'candidate_id': fingerprint(node.state),
                        'source_candidate_id': fingerprint(source)}
                verdict.update(lane=lane, source_index=index, coarse_candidate_id=fingerprint(node.state))
                self.repair_audit['final_checks'].append(verdict)
                if candidate is not None:
                    extra.append(candidate); successful_families.add(family)
                    satisfactory_found |= satisfactory(candidate, verdict)
                elif 'diagnostic_state' in verdict:
                    self.repair_audit['rejected_complete_candidates'].append(verdict)
                # A few different poses can be assessed within the successful
                # job; no later, unrelated route consumes the remaining budget.
                if satisfactory_found and attempted_finals >= 4:
                    break
            trace.update(complete_candidates_unchecked=len(complete)-attempted_finals, completion_diversity=diversity)
            self.repair_audit['jobs'].append({'source_index': index, 'lane': lane, 'status': 'attempted',
                'merge_checks': trace['merge_checks'], 'complete_shapes': len(complete), 'checked_finals': attempted_finals,
                'satisfactory_verified_page': satisfactory_found})
            if satisfactory_found:
                stopped = True; self.repair_audit.update(early_stopped=True, route_stop_reason='satisfactory_verified_page')
        self.repair_audit.update(verified_completions=len(extra), scheduled_sources_not_attempted=len(sources)-len(attempted),
            merge_checks_total=initial_merge-merge_left, maximum_merge_checks_total=initial_merge,
            per_job_merge_check_limit=self.config.repair_merge_checks,
            final_candidates_checked=checked, final_candidates_limit=self.config.repair_complete_checks)
        if not stopped:
            self.repair_audit['route_stop_reason'] = 'bounded_routes_finished'
        return self.finish([*states, *extra], started, stop)
