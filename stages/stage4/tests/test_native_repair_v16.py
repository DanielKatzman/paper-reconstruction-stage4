from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from paper_reconstruction_stage4.config import SolverConfig, REPAIR_V16_POLICY
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
from paper_reconstruction_stage4.native_bridge import pair_options
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import serialize_layout, digest
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.support import Arc
from paper_reconstruction_stage4.repair_search import Node, search as previous_search
from paper_reconstruction_stage4.repair_search_v16 import search, fair_order
from paper_reconstruction_stage4.repair_sources_v16 import alternative_sources
from paper_reconstruction_stage4.repair_support_v16 import select_witnesses, unsupported
from tests.test_native_repair_v14 import upgraded


def context(folder):
    p, c, _, source, full = upgraded(folder)
    c = replace(c, native_repair_policy=REPAIR_V16_POLICY, repair_source_limit=4).validate()
    ctx = make_page_context(p, c, ActiveSupportIndex(p, c))
    p.page_context = ctx; p.structural = ctx.scorer()
    return p, c, ctx, source, full


def contact(a, b, first, second, name):
    points = np.column_stack((np.arange(8.), np.zeros(8)))
    return {'pieces': [a, b], 'arcs': [Arc(a, ((first, first+100.),), points, 100.),
                                      Arc(b, ((second, second+100.),), points, 100.)],
            'proposal_id': name, 'role': 'merge'}


class NativeRecoveryV16Tests(unittest.TestCase):
    def test_conditional_lane_matches_legacy_and_general_lane_admits_nonspanning_page(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            h = np.linalg.inv(full.poses['piece_00']) @ full.poses['piece_02']
            proposal = {'pieces': ['piece_00', 'piece_02'], 'score': 1., 'conditioned': False,
                        'proposal_id': 'fixture', 'transform_b_to_a': h.tolist()}
            old, trace = previous_search(ctx, source, [proposal], time.perf_counter()+10., maximum_checks=16)
            strict, audit = search(ctx, source, [proposal], time.perf_counter()+10., maximum_checks=16)
            self.assertEqual(len(old), len(strict)); self.assertEqual(trace['rejections'], audit['rejections'])
            new, audit = search(ctx, source, [proposal], time.perf_counter()+10., maximum_checks=16, area_bound=True)
            self.assertTrue(new); self.assertFalse(audit['provisional_extent_enforced'])
            candidate, verdict = ctx.finalize(source, new[0][0], Feasibility(p, c))
            self.assertIsNotNone(candidate, verdict)
            layout = serialize_layout(p, candidate, 0, {'valid': True}, {'attempted': False})
            self.assertTrue(verify_result_support(p, {'layouts': [layout]}, c)['valid'])
            from paper_reconstruction_stage4.outputs import build_result, request_identity, validate_result
            from paper_reconstruction_stage4.search.common import SearchResult
            diagnostics = {**ctx.report(), 'final_feasibility': [{'valid': True}],
                           'refinement': [{'attempted': False}], 'constraint_profile': p.constraint_profile}
            result = build_result(p, c, SearchResult([candidate], diagnostics, 'exhausted_candidates'), request_identity(p, c))
            validate_result(result)
            self.assertEqual(result['schema_version'], 'stage4.assembly-result.v4')

    def test_joint_support_backtracks_from_an_occupied_cycle_witness(self):
        source = State({p: np.eye(3) for p in 'abc'}, {p: p for p in 'abc'})
        ctx = SimpleNamespace(config=SolverConfig(), edges={})
        ab = contact('a', 'b', 0., 0., 'ab')
        bc_bad = contact('b', 'c', 200., 0., 'bc_bad')
        bc_good = contact('b', 'c', 200., 200., 'bc_good')
        ac = contact('a', 'c', 200., 0., 'ac'); ac['role'] = 'cycle'
        chosen, audit = select_witnesses(ctx, source, {('a', 'b'): [ab], ('b', 'c'): [bc_bad, bc_good]},
                                        [ac], [], time.perf_counter()+3.)
        self.assertIsNotNone(chosen); self.assertEqual({r['proposal_id'] for r in chosen}, {'ab', 'bc_good', 'ac'})
        self.assertFalse(unsupported(ctx, source, chosen))
        self.assertGreater(audit['nodes'], 3)
        missing = unsupported(ctx, source, [ab, bc_bad])
        self.assertEqual(len(missing), 2); self.assertIn('cut', missing[0]); self.assertIn('arcs', missing[0])

    def test_witness_limits_cannot_accept_an_unsupported_bridge(self):
        source = State({p: np.eye(3) for p in 'ab'}, {p: p for p in 'ab'})
        ctx = SimpleNamespace(config=SolverConfig(), edges={}); ab = contact('a', 'b', 0., 0., 'ab')
        chosen, audit = select_witnesses(ctx, source, {('a', 'b'): [ab]}, [], [], time.perf_counter()+3.)
        self.assertIsNone(chosen); self.assertFalse(audit['selected'])
        chosen, audit = select_witnesses(ctx, source, {('a', 'b'): [ab]}, [], [], time.perf_counter()-1.)
        self.assertIsNone(chosen); self.assertTrue(audit['deadline_reached'])
        chosen, audit = select_witnesses(ctx, source, {('a', 'b'): [ab]}, [], [], time.perf_counter()+3., maximum_nodes=1)
        self.assertIsNone(chosen); self.assertTrue(audit['node_limit_reached'])

    def test_core_alternatives_preserve_original_and_do_not_mutate_locked_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            before = {pid: h.copy() for pid, h in source.poses.items()}
            sources, audit = alternative_sources(ctx, [(source, [], 'original')], [], time.perf_counter()+3.)
            self.assertIs(sources[0][0], source); self.assertLessEqual(len(sources), c.repair_source_limit)
            self.assertTrue(any(kind == 'native_conflict_forest' for _, _, kind in sources))
            self.assertTrue(all(len(s.components()) <= 8 and len(removed) <= 4 for s, removed, _ in sources))
            for pid, h in before.items():
                np.testing.assert_array_equal(source.poses[pid], h)
            self.assertFalse(audit['evaluation_truth_opened'])

    def test_failure_records_full_poses_and_exact_cut_without_installing_bridges(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            arcs, metrics = pair_options(p, 'piece_00', 'piece_02', full.poses, c, [])[0]
            row = {'pieces': ['piece_00', 'piece_02'], 'arcs': arcs, 'metrics': metrics, 'role': 'merge',
                   'proposal_id': 'fixture', 'proposed_transform_b_to_a': (np.linalg.inv(full.poses['piece_00'])@full.poses['piece_02']).tolist(),
                   'source_groups': source.components(), 'merged_groups': source.components()}
            state = deepcopy(full); state.selected = source.selected; state.tree = source.tree
            state.assignments = dict(source.assignments)
            node = Node(state, [row]); edges = len(p.edges)
            with patch('paper_reconstruction_stage4.native_repair_v16.pool', return_value=({tuple(row['pieces']): [row]}, [], [])):
                candidate, verdict = ctx.finalize(source, node, Feasibility(p, c))
            self.assertIsNone(candidate); self.assertEqual(len(p.edges), edges)
            self.assertEqual(set(verdict['diagnostic_state']['native_to_component']), set(p.pieces))
            self.assertTrue(verdict['unsupported_bridges'][0]['bridge_ids'])
            self.assertEqual(verdict['disposition'], 'rejected_native_support')

    def test_shared_generation_reserved_jobs_keep_a_verified_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            ctx.finalization_proposal_allowance = 100
            from paper_reconstruction_stage4.component_closure import ClosureContext
            from paper_reconstruction_stage4.native_repair import RepairContext as Legacy
            def bounded(*args, **kwargs):
                return [(Node(full, []), {})], {'merge_checks': 1, 'budget_abstained': False, 'rejections': {},
                                               'source_groups': source.components(), 'complete_shape_candidates': 1}
            with patch.object(ClosureContext, 'augment_finalists', return_value=[source]), \
                 patch.object(Legacy, 'sources', return_value=([(source, [], 'original')], 0)), \
                 patch('paper_reconstruction_stage4.native_repair_v16.generate', return_value=([], {'native_checks': 1, 'budget_abstained': False})) as generation, \
                 patch('paper_reconstruction_stage4.native_repair_v16.alternative_sources', return_value=([(source, [], 'original')], {})), \
                 patch('paper_reconstruction_stage4.native_repair_v16.search', side_effect=bounded), \
                 patch.object(ctx, 'finalize', side_effect=[(full, {'disposition': 'verified_completion'}), (None, {'disposition': 'rejected_native_support'})]):
                states = ctx.augment_finalists([source], time.perf_counter()+10.)
            generation.assert_called_once(); self.assertTrue(any(s is full for s in states))
            self.assertEqual(ctx.repair_audit['verified_completions'], 1)
            self.assertEqual([j['lane'] for j in ctx.repair_audit['jobs']], ['conditional_extent', 'observed_area'])
            self.assertLessEqual(ctx.repair_audit['merge_checks_total'], c.repair_merge_checks)

    def test_unknown_capture_and_support_pool_deadline_are_disclosed(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, source, full = context(folder)
            p.capture_metadata = None
            from paper_reconstruction_stage4.component_closure import ClosureContext
            with patch.object(ClosureContext, 'augment_finalists', return_value=[source]), \
                 patch('paper_reconstruction_stage4.native_repair_v16.generate') as generation:
                ctx.augment_finalists([source], time.perf_counter()+10.)
            generation.assert_not_called(); self.assertIn('unavailable_reason', ctx.repair_audit)
            ctx.repair_deadline = time.perf_counter()-1.
            candidate, verdict = ctx.finalize(source, Node(full, []), Feasibility(p, c))
            self.assertIsNone(candidate); self.assertEqual(verdict['disposition'], 'unverified_budget')
            self.assertIn('diagnostic_state', verdict)


if __name__ == '__main__':
    unittest.main()
