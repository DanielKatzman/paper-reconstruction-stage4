"""Geometry-driven V18 recovery; all accepted evidence is replayed independently."""
from dataclasses import replace
from collections import Counter, OrderedDict
import time
import numpy as np
from .native_repair_v17 import RepairContext as PreviousRepairContext
from .component_closure import shape_measurement
from .page_geometry import fingerprint, state_record
from .outputs import serialize_layout
from .native_bridge import check_arcs, edge_from_record
from .native_bridge_v2 import make_record
from .native_bridge_v3 import make_record as make_boundary_record
from .page_boundary_v18 import measure as measure_boundary
from .active_support import verify_result_support
from .repair_support_v18 import pool, unsupported
from .support import overlaps
from .search.execution import clone


class RepairContext(PreviousRepairContext):
    def __init__(self, problem, config, support):
        super().__init__(problem, config, support)
        self.geometry_finished = False
        self.augmentation_done = False
        self.expanded_generation_done = False
        self.support_jobs = OrderedDict()
        self.generated = []; self.completed_verdicts = {}; self.fast_states = []
        self.repair_audit.update(stopping_policy='verified-construction-or-deadline.v1',
            proposal_count_is_telemetry=True, proposal_count_limit=None,
            final_candidate_count_limit=None, page_boundary_witness_enabled=True,
            legacy_limits_not_used_to_stop_geometry=['max_proposals', 'repair_merge_checks', 'repair_complete_checks', 'repair_seconds'],
            native_checks_are_initial_batch_size=True,
            search_is_exhaustive=False, native_first=True)

    def run_geometry(self, deadline, fast_only=False):
        from .model import singleton_state
        from .feasibility import Feasibility
        from .config import STRAIGHT_DEFERRED_GROWTH
        if self.config.native_growth_policy == STRAIGHT_DEFERRED_GROWTH:
            from .native_graph_v19 import search
        else:
            from .native_graph_v18 import search
        basis = singleton_state(self.problem)
        physical = Feasibility(self.problem, self.config)
        self.repair_deadline = deadline

        def complete(node, shape):
            # A failed tree at one pose does not suppress different support
            # histories. Identical replays need not consume more final time.
            from .outputs import digest
            key = digest([state_record(node.state), [r['proposal_id'] for r in node.contacts]])
            if key in self.completed_verdicts:
                return None
            candidate, verdict = self.finalize(basis, node, physical)
            if verdict.get('disposition') != 'deferred_native_support':
                self.completed_verdicts[key] = verdict
            verdict.update(coarse_candidate_id=fingerprint(node.state), lane='native_first' if fast_only else 'native_backtracking')
            self.repair_audit['final_checks'].append(verdict)
            if candidate is None and 'diagnostic_state' in verdict:
                self.repair_audit['rejected_complete_candidates'].append(verdict)
            if candidate is not None:
                self.geometry_finished = True
                self.repair_audit.update(verified_completions=1,
                    route_stop_reason='verified_construction',
                    acceptance_requires_pixel_perfection=False)
            return candidate

        last_resume = time.perf_counter(); resume_cost = 0.

        def resume_pending(force=False):
            nonlocal last_resume, resume_cost
            if not force and time.perf_counter()-last_resume < resume_cost:
                return None
            pending = next(((key, job) for key, job in self.support_jobs.items() if not job['exhausted']), None)
            if pending is None:
                return None
            key, job = pending; started = time.perf_counter()
            candidate, verdict = self.finalize(job['source'], job['node'], physical)
            self.support_jobs.move_to_end(key)
            verdict.update(lane='deferred_native_support', continuation_resume=True)
            self.repair_audit['final_checks'].append(verdict)
            resume_cost = time.perf_counter()-started; last_resume = time.perf_counter()
            if candidate is not None:
                self.geometry_finished = True
                self.repair_audit.update(verified_completions=1, route_stop_reason='verified_construction')
            return candidate

        candidate, trace = search(self, basis, self.generated, deadline, complete, physical, fast_only,
            on_idle=None if fast_only else resume_pending)
        if not fast_only:
            while candidate is None and time.perf_counter() < deadline and any(not j['exhausted'] for j in self.support_jobs.values()):
                candidate = resume_pending(force=True)
        if candidate is not None:
            trace['stop_reason'] = 'verified_construction'
        trace['lane'] = 'native_first_fast_ordering' if fast_only else 'native_backtracking'
        self.repair_audit['merge_searches'].append(trace)
        self.repair_audit['proposal_checks'] += trace['merge_checks']
        self.repair_audit['final_candidates_checked'] = len(self.repair_audit['final_checks'])
        self.repair_audit['route_stop_reason'] = trace['stop_reason']
        self.repair_audit['pending_support_searches'] = sum(not j['exhausted'] for j in self.support_jobs.values())
        return candidate

    def solve_geometry(self, feasibility):
        """Called before legacy cold search; no count allowance is consumed."""
        from .seam_poses_v18 import generate
        from .search.common import SearchResult
        started = time.perf_counter()
        if not self.problem.capture_metadata or self.problem.capture_metadata.get('rotation_family') not in {'quarter_turn','continuous_rigid'}:
            self.repair_audit['unavailable_reason'] = 'capture_does_not_declare_quarter_turns'
            return None
        deadline = self.problem.deadline-self.config.joint_finalize_seconds
        proposals, generation = generate(self.problem, self.config, deadline,
            maximum_checks=self.config.repair_native_checks)
        self.generated = proposals
        self.repair_audit.update(native_generation=generation, generated_pose_proposals=proposals,
            shared_native_generation_passes=1, proposal_checks=generation['native_checks'])
        candidate = self.run_geometry(deadline, fast_only=True)
        incumbent = candidate
        shape = shape_measurement(self, incumbent, 768) if incumbent is not None and self.config.native_expand_marginal_geometry else None
        # This is a quality-triggered expansion, not a stronger acceptance gate.
        # Keep a verified incumbent if more fitting/search times out or worsens
        # the page. The target scales with raster measurement uncertainty;
        # no reference error or fixed 0.98 geometry score is required.
        marginal = shape is not None and shape['unfilled_fraction'] > 3.*shape['raster_fraction_uncertainty_estimate']
        if marginal:
            self.repair_audit['geometry_expansion'] = {
                'policy': 'visible-gap-expansion.v1', 'trigger': 'unfilled_exceeds_three_raster_uncertainty_estimates',
                'initial_candidate_id': fingerprint(incumbent), 'initial_geometry': shape,
                'incumbent_preserved': True, 'acceptance_thresholds_changed': False,
                'reference_geometry_used': False}
        if (candidate is None or marginal) and time.perf_counter() < deadline:
            proposals, expanded = generate(self.problem, self.config, deadline, maximum_checks=None, peaks=8)
            self.generated = list({r['proposal_id']: r for r in [*self.generated, *proposals]}.values())
            self.expanded_generation_done = True
            self.repair_audit.update(expanded_native_generation=expanded, generated_pose_proposals=self.generated,
                shared_native_generation_passes=2,
                expansion_trigger='visible_page_gaps' if marginal else 'no_verified_initial_construction')
            self.repair_audit['proposal_checks'] += expanded['native_checks']
            candidate = self.run_geometry(deadline, fast_only=True)
        if marginal:
            alternative_shape = shape_measurement(self, candidate, 768) if candidate is not None else None
            improved = alternative_shape is not None and alternative_shape['score'] > shape['score']
            if not improved:
                candidate = incumbent
            self.geometry_finished = True
            self.repair_audit['route_stop_reason'] = 'verified_construction'
            self.repair_audit['geometry_expansion'].update(
                expanded_search_attempted=self.expanded_generation_done, alternative_geometry=alternative_shape,
                alternative_improved=bool(improved), selected_candidate_id=fingerprint(candidate),
                selected_geometry=alternative_shape if improved else shape)
        self.repair_audit['native_first_seconds'] = time.perf_counter()-started
        if candidate is None:
            return None
        return SearchResult([candidate], {'proposals': 0, 'native_first_completion': True}, 'verified_construction')

    def augment_finalists(self, states, deadline):
        if self.geometry_finished or self.augmentation_done:
            return states
        self.augmentation_done = True
        from .feasibility import Feasibility
        physical = Feasibility(self.problem, self.config)
        for state in states:
            if len(state.components()) != 1 or not shape_measurement(self, state, 768)['shape_gate_pass']:
                continue
            check = physical.state(state, final=True)
            if not check['valid']:
                continue
            try:
                verify_result_support(self.problem, {'layouts': [serialize_layout(self.problem, state, 0, check, {'attempted': False})]}, self.config)
            except ValueError:
                continue
            self.geometry_finished = True
            self.repair_audit['route_stop_reason'] = 'protected_verified_construction'
            return states
        if self.repair_audit.get('unavailable_reason'):
            return states
        # Preserve the established positive component routes as ordering
        # passes. Their finite beams are followed by native backtracking;
        # neither a route count nor the old global 40k allowance ends V18.
        from .repair_sources_v16 import alternative_sources
        from .repair_search_v18 import search as component_search
        originals, _ = self.sources(states)
        sources, conditioning = alternative_sources(self, originals, self.generated,
            min(deadline, time.perf_counter()+2.))
        self.repair_audit['core_conditioning'] = conditioning
        self.repair_audit['sources'] = [self.source_record(s, removed, kind) for s, removed, kind in sources]
        for index, (source, _, _) in enumerate(sources):
            for area_bound in (False, True):
                if time.perf_counter() >= deadline:
                    break
                # A scheduling slice leaves later routes time. It never means
                # that a rejected/deferred branch is globally impossible.
                lane_deadline = min(deadline, time.perf_counter()+max(4., (deadline-time.perf_counter())*.12))
                self.repair_deadline = lane_deadline

                def complete(node, shape):
                    candidate, verdict = self.finalize(source, node, physical)
                    verdict.update(lane='component_area' if area_bound else 'component_extent', source_index=index)
                    self.repair_audit['final_checks'].append(verdict)
                    if candidate is None and 'diagnostic_state' in verdict:
                        self.repair_audit['rejected_complete_candidates'].append(verdict)
                    return candidate

                candidate, trace = component_search(self, source, self.generated, lane_deadline,
                    maximum_checks=None, beam=self.config.repair_beam, area_bound=area_bound,
                    physical=physical, on_complete=complete)
                trace.update(lane='component_area' if area_bound else 'component_extent', source_index=index)
                self.repair_audit['merge_searches'].append(trace)
                self.repair_audit['proposal_checks'] += trace['merge_checks']
                if not isinstance(candidate, list) and candidate is not None:
                    self.geometry_finished = True
                    self.repair_audit.update(verified_completions=1, route_stop_reason='verified_construction')
                    return [candidate, *states]
        # Broaden observed native modes after the quick route fails. This is
        # finite configured fit work; its count is not a recovery stop limit.
        from .seam_poses_v18 import generate
        if not self.expanded_generation_done:
            proposals, generation = generate(self.problem, self.config, deadline, maximum_checks=None, peaks=8)
            by_id = {r['proposal_id']: r for r in [*self.generated, *proposals]}
            self.generated = list(by_id.values())
            self.repair_audit.update(expanded_native_generation=generation,
                generated_pose_proposals=self.generated, shared_native_generation_passes=2)
            self.repair_audit['proposal_checks'] += generation['native_checks']
        candidate = self.run_geometry(deadline)
        if candidate is not None:
            return [candidate, *states]
        self.repair_audit.update(deadline_reached=time.perf_counter() >= deadline,
            budget_abstained=time.perf_counter() >= deadline,
            unresolved_is_not_impossibility_proof=True)
        return states

    def prioritize_finalists(self, problem, config, states, feasibility):
        # Recovery can use the entire remaining solve deadline, rather than a
        # fixed finalization share or the legacy max_proposals allowance.
        from .page_geometry import GeometryContext
        states = self.augment_finalists(states, problem.deadline)
        # Parent calls augment again; the completed/attempted flag suppresses it.
        return GeometryContext.prioritize_finalists(self, problem, config, states, feasibility)

    def finalize(self, source, node, physical):
        """Advance a resumable support search, yielding time to other poses.

        Scheduling quanta defer work without rejecting it or forgetting its
        DFS stack. They do not cap the number of alternatives or final pages.
        """
        from .outputs import digest
        from .native_bridge import arc_value
        deadline = getattr(self, 'repair_deadline', float('inf'))
        key = digest([fingerprint(source), state_record(node.state),
            [[r['pieces'], r['proposal_id'], [arc_value(a) for a in r['arcs']]] for r in node.contacts]])
        if key not in self.support_jobs:
            mandatory, cycles, occupied = pool(self, source, node, deadline)
            if mandatory is None:
                return None, {'disposition': 'unverified_deadline', 'reason': 'support_measurement_deadline'}
            audit = {'policy': 'resumable-disjoint-cut-witness.v3', 'nodes': 0,
                'support_alternatives': sum(map(len, mandatory.values())),
                'cycle_options': len(cycles), 'maximum_nodes': None, 'resumptions': 0}
            pairs = sorted(mandatory, key=lambda pair: (len(mandatory[pair]), pair))
            pause = object()

            def visit(chosen, used, depth):
                if time.perf_counter() >= self.support_turn_deadline:
                    yield pause
                audit['nodes'] += 1
                if depth < len(pairs):
                    options = mandatory[pairs[depth]]
                else:
                    greedy = list(chosen); busy = list(used)
                    selected = {tuple(r['pieces']) for r in chosen}
                    for row in cycles:
                        pair = tuple(row['pieces'])
                        if pair not in selected and not any(overlaps(a, b, self.config) for a in row['arcs'] for b in busy):
                            greedy.append(row); busy.extend(row['arcs']); selected.add(pair)
                    yield greedy
                    yield chosen
                    missing = unsupported(self, source, chosen)
                    if not missing:
                        return
                    cut = set(min(missing, key=lambda r: (min(len(r['cut']), len(r['other_side'])), r['pieces']))['cut'])
                    selected = {tuple(r['pieces']) for r in chosen}
                    options = [r for r in cycles if tuple(r['pieces']) not in selected and
                               ((r['pieces'][0] in cut) != (r['pieces'][1] in cut))]
                for row in options:
                    if time.perf_counter() >= self.support_turn_deadline:
                        yield pause
                    if any(overlaps(a, b, self.config) for a in row['arcs'] for b in used):
                        continue
                    yield from visit([*chosen, row], [*used, *row['arcs']], depth+1)

            self.support_jobs[key] = {'source': source, 'node': node, 'iterator': visit([], list(occupied), 0),
                'audit': audit, 'tested': set(), 'last': {}, 'exhausted': False, 'pause': pause}
        job = self.support_jobs[key]; audit = job['audit']; audit['resumptions'] += 1
        # Fair scheduling only: retain the iterator when its time share ends.
        remaining = max(0., deadline-time.perf_counter())
        quantum = max(.25, min(2., remaining/max(2, len(node.state.poses)**2)))
        turn_deadline = min(deadline, time.perf_counter()+quantum)
        self.support_turn_deadline = turn_deadline
        candidate = None
        while not job['exhausted'] and time.perf_counter() < turn_deadline:
            chosen = next(job['iterator'], None)
            if chosen is None:
                job['exhausted'] = True
                break
            if chosen is job['pause']:
                break
            support_key = digest([[r['pieces'], [arc_value(a) for a in r['arcs']]] for r in chosen])
            if support_key in job['tested']:
                continue
            job['tested'].add(support_key)
            candidate, job['last'] = self.evaluate_contacts(source, node, chosen, physical)
            if candidate is not None:
                job['exhausted'] = True
                break
        audit.update(selected=candidate is not None, alternatives_exhausted=job['exhausted'],
            continuation_pending=not job['exhausted'], overall_deadline_reached=time.perf_counter() >= deadline)
        verdict = {**job['last'], 'support_selection': dict(audit)}
        if candidate is None:
            verdict.update(diagnostic_state=state_record(node.state), candidate_id=fingerprint(node.state),
                source_candidate_id=fingerprint(source),
                disposition=('deferred_native_support' if not job['exhausted'] else job['last'].get('disposition', 'rejected_native_support')))
        return candidate, verdict

    def evaluate_contacts(self, source, node, contacts, physical):
        state = clone(node.state)
        used = [arc for eid in source.selected for arc in self.support.arcs_for(source, eid)]
        edges = []; records = []
        shape = shape_measurement(self, state, 768)
        if not shape['shape_gate_pass']:
            return None, {'geometry': shape, 'disposition': 'rejected_geometry'}
        missing = unsupported(self, source, contacts)
        all_occupied = [*used, *[a for row in contacts for a in row['arcs']]]
        witnesses = {}; witness_failures = []
        for row in missing:
            try:
                witnesses[(tuple(row['pieces']), row['proposal_id'])] = measure_boundary(
                    self.problem, state.poses, row['cut'], all_occupied, self.config)
            except ValueError as exc:
                witness_failures.append({'pieces': row['pieces'], 'reason': str(exc)})
        for contact in contacts:
            try:
                check_arcs(self.problem, contact['arcs'], state.poses, self.config, used)
            except ValueError as exc:
                from .native_bridge import arc_value
                return None, {'disposition': 'rejected_native_support', 'reason': str(exc),
                    'failed_contact': {'pieces': contact['pieces'], 'proposal_id': contact['proposal_id'],
                                       'arcs': [arc_value(a) for a in contact['arcs']]}}
            used.extend(contact['arcs'])
            witness = witnesses.get((tuple(contact['pieces']), contact['proposal_id']))
            record = (make_boundary_record(self.problem, state, contact, fingerprint(source), witness)
                      if witness is not None else make_record(self.problem, state, contact, fingerprint(source)))
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
        audit = {'page_boundary_witnesses': len(witnesses), 'page_boundary_failures': witness_failures, 'geometry': shape, 'unsupported_bridges': missing, 'native_bridge_records': records}
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

