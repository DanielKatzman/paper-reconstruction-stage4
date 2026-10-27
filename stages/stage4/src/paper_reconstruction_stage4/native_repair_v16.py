"""Shared native proposals, reserved recovery lanes and inspectable final checks."""
from dataclasses import replace
import time
import numpy as np
from .native_repair import RepairContext as PreviousRepairContext
from .component_closure import ClosureContext, shape_measurement
from .seam_poses_v15 import generate
from .page_geometry import fingerprint, state_record
from .outputs import serialize_layout
from .native_bridge import check_arcs, edge_from_record
from .native_bridge_v2 import make_record
from .active_support import verify_result_support
from .feasibility import Feasibility
from .search.execution import clone
from .repair_sources_v16 import alternative_sources
from .repair_support_v16 import pool, select_witnesses, unsupported
from .repair_search_v16 import search


class RepairContext(PreviousRepairContext):
    def __init__(self, problem, config, support):
        super().__init__(problem, config, support)
        self.repair_audit.update(component_bound_policy='bounded-extent-and-area-portfolio.v1',
            generation_schedule='pair-rotation-strata.v1', source_selection_changed=True,
            fit_measurement_changed=False, page_format_or_dpi_prior_used=False,
            maximum_components=8, maximum_released_joins=8, jobs=[],
            shared_native_generation_passes=0, rejected_complete_candidates=[])

    def evaluate_contacts(self, source, node, contacts, physical):
        state = clone(node.state)
        used = [arc for eid in source.selected for arc in self.support.arcs_for(source, eid)]
        edges = []; records = []
        for contact in contacts:
            try:
                check_arcs(self.problem, contact['arcs'], state.poses, self.config, used)
            except ValueError as exc:
                from .native_bridge import arc_value
                return None, {'disposition': 'rejected_native_support', 'reason': str(exc),
                    'failed_contact': {'pieces': contact['pieces'], 'proposal_id': contact['proposal_id'],
                                       'arcs': [arc_value(a) for a in contact['arcs']]}}
            used.extend(contact['arcs'])
            record = make_record(self.problem, state, contact, fingerprint(source))
            edge, arcs = edge_from_record(self.problem, record, self.config)
            state.selected = (*state.selected, edge.id)
            state.assignments[edge.id] = edge.occurrences[0]['occurrence_id']
            if contact['role'] == 'merge':
                state.tree = (*state.tree, edge.id)
            edges.append((edge, arcs)); records.append(record)
        shape = shape_measurement(self, state, 768)
        missing = unsupported(self, source, contacts)
        for row in missing:
            row['bridge_ids'] = [r['bridge_id'] for r in records if r['placement_proposal_id'] == row['proposal_id'] and r['pieces'] == row['pieces']]
        audit = {'geometry': shape, 'unsupported_bridges': missing, 'native_bridge_records': records}
        if not shape['shape_gate_pass']:
            return None, {**audit, 'disposition': 'rejected_geometry'}
        check = physical.state(state, final=True)
        if not check['valid']:
            return None, {**audit, 'disposition': 'rejected_native', 'reason': check['reason']}
        temporary = replace(self.problem, edges=[*self.problem.edges, *[e for e, _ in edges if e.id not in self.edges]])
        try:
            verify_result_support(temporary, {'layouts': [serialize_layout(temporary, state, 0, check, {'attempted': False})]}, self.config)
        except ValueError as exc:
            return None, {**audit, 'disposition': 'rejected_native_support', 'reason': str(exc)}
        # Only independently verified contacts enter the serving problem.
        for edge, arcs in edges:
            if edge.id not in self.edges:
                self.problem.edges.append(edge); self.edges[edge.id] = edge
            self.support.arcs[edge.occurrences[0]['occurrence_id']] = arcs
        self.cache.clear(); self.ambiguity_cache.clear()
        scorer = self.scorer(); scorer.edges.update(self.edges)
        state.score = float(scorer.evaluate(state)['score'])
        return state, {**audit, 'disposition': 'verified_completion', 'candidate_id': fingerprint(state),
                       'native_bridges': len(edges), 'new_pairs': [r['pieces'] for r in contacts]}

    def finalize(self, source, node, physical):
        deadline = getattr(self, 'repair_deadline', float('inf'))
        mandatory, cycles, used = pool(self, source, node, deadline)
        if mandatory is None:
            return None, {'disposition': 'unverified_budget', 'reason': 'native_support_pool_deadline',
                          'diagnostic_state': state_record(node.state), 'candidate_id': fingerprint(node.state),
                          'source_candidate_id': fingerprint(source)}
        # First preserve the previous support choice and greedy cycle ordering.
        contacts = list(node.contacts); occupied = [*used, *[a for r in contacts for a in r['arcs']]]
        pairs = {tuple(r['pieces']) for r in contacts}
        from .support import overlaps
        for row in sorted(cycles, key=lambda r: (-min(a.length for a in r['arcs']), r['pieces'], r['proposal_id'])):
            pair = tuple(row['pieces'])
            if pair in pairs or any(overlaps(a, b, self.config) for a in row['arcs'] for b in occupied):
                continue
            contacts.append(row); occupied.extend(row['arcs']); pairs.add(pair)
        candidate, first = self.evaluate_contacts(source, node, contacts, physical)
        if candidate is not None:
            return candidate, {**first, 'support_selection': {'policy': 'preserved-greedy', 'attempted': False}}
        chosen, selection = select_witnesses(self, source, mandatory, cycles, used, deadline)
        if chosen is not None:
            candidate, verdict = self.evaluate_contacts(source, node, chosen, physical)
        else:
            verdict = first
        verdict = {**verdict, 'initial_support_failure': first,
                   'support_selection': {'policy': 'bounded-disjoint-cut-witness.v1', **selection}}
        if candidate is None:
            # Diagnostic records are never passed back as solver seeds.
            verdict['diagnostic_state'] = state_record(node.state)
            verdict['candidate_id'] = fingerprint(node.state)
            verdict['source_candidate_id'] = fingerprint(source)
            verdict['occupied_source_arcs'] = [
                {'piece': a.piece, 'intervals': [list(v) for v in a.intervals], 'length': a.length} for a in used]
        return candidate, verdict

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
            self.repair_audit['protected_supported_completion'] = True
            return self.finish(states, started, stop)
        if not self.problem.capture_metadata or self.problem.capture_metadata.get('rotation_family') != 'quarter_turn':
            self.repair_audit['unavailable_reason'] = 'capture_does_not_declare_quarter_turns'
            return self.finish(states, started, stop)
        originals, omitted = super().sources(states)
        self.repair_audit['sources_omitted_by_limit'] = omitted
        allowance = max(0, getattr(self, 'finalization_proposal_allowance', 0)-len(self.deletion_audit['checks'])-self.closure_audit['generated_proposals'])
        if not originals or allowance == 0 or time.perf_counter() >= stop:
            self.repair_audit['budget_abstained'] = True
            return self.finish(states, started, stop)
        proposals, generation = generate(self.problem, self.config, min(stop, started+self.config.repair_seconds*.25),
            maximum_checks=min(allowance, self.config.repair_native_checks),
            frontiers=[s.components() for s, _, _ in originals])
        self.repair_audit.update(native_generation=generation, generated_pose_proposals=proposals,
                                 shared_native_generation_passes=1, proposal_checks=generation['native_checks'])
        allowance -= generation['native_checks']
        sources, conditioning = alternative_sources(self, originals, proposals, min(stop, time.perf_counter()+2.))
        self.repair_audit.update(core_conditioning=conditioning, scheduled_source_count=len(sources))
        self.repair_audit['sources'] = [{'candidate_id': fingerprint(s), 'groups': s.components(),
            'released_geometry_ids': removed, 'kind': kind} for s, removed, kind in sources]
        if not sources:
            return self.finish(states, started, stop)
        # Reserve both a legacy-positive lane and general source alternatives.
        # Unused quotas flow to later jobs, but no first source owns all time.
        jobs = [(0, 'conditional_extent'), *[(i, 'observed_area') for i in range(len(sources))]]
        merge_left = min(allowance, self.config.repair_merge_checks)
        initial_merge = merge_left; extra = []; checked = 0; attempted_sources = set()
        for j, (index, lane) in enumerate(jobs):
            remaining_jobs = len(jobs)-j
            now = time.perf_counter()
            if now >= stop or merge_left <= 0:
                self.repair_audit['jobs'].append({'source_index': index, 'lane': lane, 'status': 'unattempted_budget'})
                continue
            # The constrained lane needs its original check allowance; the area
            # lane is guaranteed the remaining share by a separate reserve.
            checks = min(merge_left, max(1, int(initial_merge*.75))) if j == 0 else (
                max(1, int(merge_left*.75)) if j == 1 and remaining_jobs > 1 else max(1, merge_left//remaining_jobs))
            seconds = min(16., (stop-now)*.36) if j == 0 else (
                (stop-now)*.65 if j == 1 and remaining_jobs > 1 else (stop-now)/remaining_jobs)
            job_stop = min(stop, now+seconds); search_stop = max(now, job_stop-min(2., seconds*.2))
            source = sources[index][0]; attempted_sources.add(index)
            complete, trace = search(self, source, proposals, search_stop, maximum_checks=checks,
                beam=self.config.repair_beam, area_bound=lane == 'observed_area', physical=physical)
            trace.update(source_candidate_id=fingerprint(source), lane=lane,
                         reserved_seconds=seconds, finalization_reserved_seconds=job_stop-search_stop)
            self.repair_audit['merge_searches'].append(trace)
            self.repair_audit['jobs'].append({'source_index': index, 'lane': lane, 'status': 'attempted',
                                            'merge_checks': trace['merge_checks'], 'complete_shapes': len(complete)})
            merge_left -= trace['merge_checks']; self.repair_audit['proposal_checks'] += trace['merge_checks']
            # Reserve at least one final check per later lane, when available.
            quota = min(self.config.repair_complete_checks-checked,
                        max(1, (self.config.repair_complete_checks-checked)//remaining_jobs))
            self.repair_deadline = job_stop
            attempted_finals = 0
            for node, shape in complete[:quota]:
                if time.perf_counter() >= job_stop or checked >= self.config.repair_complete_checks:
                    break
                checked += 1; attempted_finals += 1
                try:
                    candidate, verdict = self.finalize(source, node, physical)
                except ValueError as exc:
                    candidate = None; verdict = {'disposition': 'rejected_native_support', 'reason': str(exc),
                        'diagnostic_state': state_record(node.state), 'candidate_id': fingerprint(node.state),
                        'source_candidate_id': fingerprint(source)}
                verdict.update(lane=lane, source_index=index)
                self.repair_audit['final_checks'].append(verdict)
                if candidate is not None:
                    extra.append(candidate)
                elif 'diagnostic_state' in verdict:
                    self.repair_audit['rejected_complete_candidates'].append(verdict)
            trace['complete_candidates_unchecked'] = len(complete)-attempted_finals
            # A valid completion is retained, never replaced by a later failure.
        self.repair_audit.update(verified_completions=len(extra),
            scheduled_sources_not_attempted=len(sources)-len(attempted_sources),
            merge_checks_total=initial_merge-merge_left, maximum_merge_checks_total=initial_merge,
            final_candidates_checked=checked, final_candidates_limit=self.config.repair_complete_checks)
        return self.finish([*states, *extra], started, stop)

    def finish(self, states, started, stop):
        generation = self.repair_audit.get('native_generation') or {}
        self.repair_audit.update(elapsed_seconds=time.perf_counter()-started,
            generation_budget_abstained=bool(generation.get('budget_abstained', False)),
            merge_budget_abstained=any(r['budget_abstained'] for r in self.repair_audit['merge_searches']),
            route_budget_abstained=bool(self.repair_audit['budget_abstained'] or time.perf_counter() >= stop or
                                       any(j['status'] == 'unattempted_budget' for j in self.repair_audit['jobs'])))
        self.repair_audit['budget_abstained'] = any(self.repair_audit[k] for k in (
            'generation_budget_abstained', 'merge_budget_abstained', 'route_budget_abstained'))
        return states
