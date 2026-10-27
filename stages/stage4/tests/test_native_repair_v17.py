from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from paper_reconstruction_stage4.config import CLOSURE_POLICY, REPAIR_V17_POLICY
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.native_bridge import pair_options, arc_value
from paper_reconstruction_stage4.seam_poses import spread
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.repair_search import Node
from paper_reconstruction_stage4.repair_search import search as legacy_search
from paper_reconstruction_stage4.repair_search_v17 import search as general_search
from paper_reconstruction_stage4.native_graph_v17 import search as graph_search
from paper_reconstruction_stage4.native_repair_v17 import satisfactory, diverse_completions
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import serialize_layout, digest, build_result, request_identity, validate_result
from paper_reconstruction_stage4.search.common import SearchResult
from tests.test_native_repair_v14 import upgraded


def context(folder):
    p, c, _, source, full = upgraded(folder)
    c = replace(c, native_repair_policy=REPAIR_V17_POLICY, repair_source_limit=4).validate()
    ctx = make_page_context(p, c, ActiveSupportIndex(p, c)); p.page_context = ctx; p.structural = ctx.scorer()
    return p, c, ctx, source, full


def native_proposals(p, c, full):
    rows = []
    for a, b in combinations(sorted(p.pieces), 2):
        for arcs, metrics in pair_options(p, a, b, full.poses, c, []):
            h = np.linalg.inv(full.poses[a])@full.poses[b]
            row = {'pieces': [a, b], 'arcs': [arc_value(v) for v in arcs], 'native_length_px': min(v.length for v in arcs),
                'shape_spread_ratio': spread(arcs), 'native_metrics': metrics, 'transform_b_to_a': h.tolist(),
                'conditioned': spread(arcs) >= .02, 'score': 1.}
            rows.append({**row, 'proposal_id': digest(row)})
    return rows


def grid_context(folder, rows=4, cols=4):
    from tests.test_ambiguity_v12 import upgraded as grid
    p, c, _, full = grid(folder, rows, cols)
    c = replace(c, component_closure_policy=CLOSURE_POLICY, native_repair_policy=REPAIR_V17_POLICY,
        repair_source_limit=4, joint_finalize_seconds=30., joint_wall_seconds=50.).validate()
    for piece in p.pieces.values():
        piece.mask = np.zeros((128, 128), bool); piece.mask[8:117, 8:117] = True
        piece.contour['points_xy'] *= 4.
        with Image.open(piece.image_path) as image:
            image.resize((128, 128)).save(Path(folder)/(piece.id+'.png'))
        piece.image_path = Path(folder)/(piece.id+'.png')
    for pose in full.poses.values():
        pose[:2, 2] *= 4.
    p.edges = []; full.selected = (); full.tree = (); full.assignments = {}
    p.capture_metadata = {'rotation_family': 'quarter_turn', 'capture_profile_sha256': p.constraint_profile['capture_profile_sha256']}
    p.structural = None
    ctx = make_page_context(p, c, ActiveSupportIndex(p, c)); p.page_context = ctx; p.structural = ctx.scorer()
    return p, c, ctx, full


class NativeRecoveryV17Tests(unittest.TestCase):
    def test_legacy_general_order_and_cap_match_v15_search(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            proposal = {'pieces': ['piece_00', 'piece_02'], 'score': 1., 'proposal_id': 'fixture',
                        'transform_b_to_a': (np.linalg.inv(full.poses['piece_00'])@full.poses['piece_02']).tolist()}
            old, before = legacy_search(ctx, source, [proposal], time.perf_counter()+5., maximum_checks=32,
                                       component_bound=lambda state: ctx.component_bounds(state)['valid'])
            new, after = general_search(ctx, source, [proposal], time.perf_counter()+5., maximum_checks=32, area_bound=True)
            self.assertEqual(before['rejections'], after['rejections']); self.assertEqual(len(old), len(new))
            self.assertEqual(after['successor_limit'], 16)

    def test_component_cache_matches_original_bounds_and_reuses_unchanged_pieces(self):
        from paper_reconstruction_stage4.page_geometry import GeometryContext
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            basis = singleton_state(p); changed = deepcopy(basis); changed.poses['piece_03'][0, 2] += .125
            for state in [basis, changed, source, full]:
                ctx.bound_cache.clear(); expected = GeometryContext.component_bounds(ctx, state)
                ctx.bound_cache.clear(); actual = ctx.component_bounds(state)
                self.assertEqual(expected, actual)
            ctx.component_rectangle_cache.clear(); ctx.bound_cache.clear()
            ctx.component_bounds(basis); before = ctx.counts['component_rectangle_extractions']
            ctx.component_bounds(changed)
            self.assertEqual(ctx.counts['component_rectangle_extractions']-before, 1)
            self.assertGreaterEqual(ctx.counts['component_rectangle_cache_hits'], 3)

    def test_graph_priority_uses_distinct_correlated_neighbors(self):
        from collections import defaultdict
        from paper_reconstruction_stage4.native_graph_v17 import Growth, offers
        from paper_reconstruction_stage4.model import State
        poses = {pid: np.eye(3) for pid in 'abcd'}; poses['b'][0, 2] = 20.
        state = State(poses, {'a': 'a', 'b': 'a', 'c': 'c', 'd': 'd'})
        indexed = defaultdict(list)
        for a, b, tx in [('a', 'c', 10.), ('b', 'c', -10.), ('a', 'd', 70.)]:
            h = np.eye(3); h[0, 2] = tx
            row = {'pieces': [a, b], 'transform_b_to_a': h.tolist(), 'native_length_px': 100.,
                'shape_spread_ratio': .05, 'native_metrics': {'bidirectional_contact_p95_native_px': 1.},
                'proposal_id': a+b}
            indexed[a].append(row); indexed[b].append(row)
        rows = offers(Growth(Node(state), 'a', 'seed'), indexed, 8.)
        self.assertEqual(rows[0]['piece'], 'c'); self.assertEqual(rows[0]['cycle_neighbors'], 1)
        self.assertEqual(next(r for r in rows if r['piece'] == 'd')['cycle_neighbors'], 0)

    def test_independent_graph_reconstructs_sixteen_piece_grid_without_old_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = grid_context(folder)
            rows = native_proposals(p, c, full); self.assertEqual(len(p.edges), 0)
            basis = singleton_state(p)
            completed, audit = graph_search(ctx, basis, rows, time.perf_counter()+15., 4000, beam=12)
            self.assertTrue(completed, audit); self.assertEqual(audit['completed_depth'], 16)
            self.assertEqual(audit['legacy_selected_joins_inherited'], 0)
            candidate, verdict = ctx.finalize(basis, completed[0][0], Feasibility(p, c))
            self.assertIsNotNone(candidate, verdict)
            layout = serialize_layout(p, candidate, 0, {'valid': True}, {'attempted': False})
            self.assertTrue(verify_result_support(p, {'layouts': [layout]}, c)['valid'])
            result = build_result(p, c, SearchResult([candidate], {**ctx.report(), 'final_feasibility': [{'valid': True}],
                'refinement': [{'attempted': False}], 'constraint_profile': p.constraint_profile}, 'exhausted_candidates'), request_identity(p, c))
            validate_result(result)

    def test_independent_graph_does_not_accept_old_locks_or_translated_singletons(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            with self.assertRaisesRegex(ValueError, 'singleton'):
                graph_search(ctx, source, [], time.perf_counter()+1., 10)
            basis = singleton_state(p); basis.poses[min(p.pieces)][0, 2] = 20.
            with self.assertRaisesRegex(ValueError, 'old piece poses'):
                graph_search(ctx, basis, [], time.perf_counter()+1., 10)

    def test_exhausted_growth_preserves_its_deepest_partial_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder); rows = native_proposals(p, c, full)[:1]
            completed, audit = graph_search(ctx, singleton_state(p), rows, time.perf_counter()+2., 32)
            self.assertFalse(completed); self.assertEqual(audit['stop_reason'], 'bounded_frontier_exhausted')
            self.assertEqual(audit['observed_growth_states'][0]['component_sizes'], [2, 1, 1])

    def test_graph_budget_deadline_and_rotation_contract_abstain(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder); rows = native_proposals(p, c, full); basis = singleton_state(p)
            completed, audit = graph_search(ctx, basis, rows, time.perf_counter()-1., 10)
            self.assertFalse(completed); self.assertEqual(audit['merge_checks'], 0); self.assertTrue(audit['budget_abstained'])
            completed, audit = graph_search(ctx, basis, rows, time.perf_counter()+2., 1)
            self.assertFalse(completed); self.assertEqual(audit['merge_checks'], 1)
            p.capture_metadata = None
            with self.assertRaisesRegex(ValueError, 'quarter-turn'):
                graph_search(ctx, basis, rows, time.perf_counter()+1., 10)

    def test_incremental_checks_match_full_checks_when_previous_group_is_valid(self):
        from collections import Counter
        from paper_reconstruction_stage4.native_graph_v17 import incremental_check
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder); physical = Feasibility(p, c)
            for shift in (0., -108., 1000.):
                trial = deepcopy(full); trial.poses['piece_03'][0, 2] += shift
                counters = Counter()
                actual, reason = incremental_check(ctx, trial, 'piece_03', set(trial.poses), physical, counters)
                self.assertEqual(actual, physical.state(trial, final=False)['valid'])
                self.assertLessEqual(counters['incremental_collision_pair_checks'], len(p.pieces)-1)

    def test_undeclared_rotation_is_rejected_during_native_proposal_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder); rows = native_proposals(p, c, full)
            bad = deepcopy(rows[0]); angle = .3
            bad['transform_b_to_a'][0][:2] = [np.cos(angle), -np.sin(angle)]
            bad['transform_b_to_a'][1][:2] = [np.sin(angle), np.cos(angle)]
            completed, audit = graph_search(ctx, singleton_state(p), [bad], time.perf_counter()+2., 10)
            self.assertFalse(completed)
            self.assertEqual(audit['rejections']['native_proposal_replay'], 1)

    def test_early_stop_requires_independent_success_and_good_fine_page(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, source, full = context(folder)
            verdict = {'disposition': 'verified_completion', 'geometry': {'shape_gate_pass': True, 'score': .99}}
            self.assertTrue(satisfactory(full, verdict)); self.assertFalse(satisfactory(source, verdict))
            self.assertFalse(satisfactory(None, verdict))
            self.assertFalse(satisfactory(full, {**verdict, 'disposition': 'rejected_native_support'}))
            self.assertFalse(satisfactory(full, {**verdict, 'geometry': {'shape_gate_pass': True, 'score': .94}}))

    def test_final_checks_are_pose_diverse_with_second_support_option(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            first = Node(full); same = Node(deepcopy(full)); other = Node(deepcopy(full))
            other.state.poses['piece_03'][0, 2] += 30.
            ordered, audit = diverse_completions(ctx, [(first, {}), (same, {}), (other, {}), (same, {})])
            self.assertIs(ordered[0][0], first); self.assertIs(ordered[1][0], other)
            self.assertEqual(audit['pose_families'], 2); self.assertEqual(len(ordered), 3)

    def test_verified_page_stops_later_lanes_but_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder); ctx.finalization_proposal_allowance = 100
            from paper_reconstruction_stage4.component_closure import ClosureContext
            from paper_reconstruction_stage4.native_repair import RepairContext as Legacy
            trace = {'merge_checks': 1, 'budget_abstained': False, 'complete_shape_candidates': 1,
                     'rejections': {}, 'source_groups': source.components()}
            with patch.object(ClosureContext, 'augment_finalists', return_value=[source]), \
                 patch.object(Legacy, 'sources', return_value=([(source, [], 'original')], 0)), \
                 patch('paper_reconstruction_stage4.native_repair_v17.generate', return_value=([], {'native_checks': 1, 'budget_abstained': False})) as generation, \
                 patch('paper_reconstruction_stage4.native_repair_v17.alternative_sources', return_value=([(source, [], 'original')], {})), \
                 patch('paper_reconstruction_stage4.native_repair_v17.component_search', return_value=([(Node(full), {})], trace)) as search, \
                 patch('paper_reconstruction_stage4.native_repair_v17.native_search') as independent, \
                 patch.object(ctx, 'finalize', return_value=(full, {'disposition': 'verified_completion', 'geometry': {'shape_gate_pass': True, 'score': .99}})):
                states = ctx.augment_finalists([source], time.perf_counter()+15.)
            generation.assert_called_once(); search.assert_called_once(); independent.assert_not_called()
            self.assertTrue(any(state is full for state in states)); self.assertTrue(ctx.repair_audit['early_stopped'])
            self.assertEqual(ctx.repair_audit['jobs'][-1]['status'], 'skipped_verified_page')
            self.assertIn('state', ctx.repair_audit['sources'][0])
            self.assertFalse(ctx.repair_audit['sources'][1]['state']['selected_geometry_ids'])


if __name__ == '__main__':
    unittest.main()
