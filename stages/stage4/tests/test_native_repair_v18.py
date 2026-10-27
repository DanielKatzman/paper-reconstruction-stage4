from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from tests.test_native_repair_v17 import grid_context, native_proposals
from paper_reconstruction_stage4.config import REPAIR_V18_POLICY, SolverConfig
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.native_graph_v18 import search
from paper_reconstruction_stage4.page_boundary_v18 import measure, verify
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import serialize_layout, digest


def fixture(folder, rows=2, cols=2):
    p, c, _, full = grid_context(folder, rows, cols)
    c = replace(c, native_repair_policy=REPAIR_V18_POLICY, max_proposals=1).validate()
    ctx = make_page_context(p, c, ActiveSupportIndex(p, c))
    p.page_context = ctx; p.structural = ctx.scorer()
    return p, c, ctx, full


class GeometryRecoveryV18Tests(unittest.TestCase):
    def test_profile_has_configurable_time_ceiling_and_explicit_new_policy(self):
        import json
        c = SolverConfig.from_dict(json.loads(Path('configs/joint_v18_geometry_development.json').read_text()))
        self.assertEqual(c.native_repair_policy, REPAIR_V18_POLICY)
        self.assertEqual(c.joint_wall_seconds, 600.)
        replace(c, joint_wall_seconds=900.).validate()

    def test_independent_corner_boundary_replays_and_does_not_use_page_format(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            cut = [max(p.pieces)]
            witness = measure(p, full.poses, cut, [], c)
            self.assertTrue(verify(p, full.poses, cut, [], c, witness))
            self.assertFalse(witness['page_format_or_dpi_used'])
            self.assertEqual({r['side'] for r in witness['observed_external_arcs'] if not r['cut_side']}, set(range(4)))

    def test_boundary_rejects_incomplete_inventory_and_wrong_corner_position(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            cut = [max(p.pieces)]; poses = deepcopy(full.poses)
            poses[cut[0]][0, 2] += 40.
            with self.assertRaises(ValueError):
                measure(p, poses, cut, [], c)
            del poses[cut[0]]
            with self.assertRaises(ValueError):
                measure(p, poses, cut, [], c)

    def test_boundary_rejects_rehashed_fabricated_lines_and_nonorthogonal_axes(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder); cut = [max(p.pieces)]
            witness = measure(p, full.poses, cut, [], c)
            changed = deepcopy(witness); changed['body_line_low'][0] -= 30.
            changed['witness_id'] = digest({k: v for k, v in changed.items() if k != 'witness_id'})
            with self.assertRaises(ValueError):
                verify(p, full.poses, cut, [], c, changed)
            with self.assertRaises(ValueError):
                measure(p, full.poses, cut, [], c, [[1., 1.], [0., 1.]])

    def test_geometry_can_complete_beyond_old_proposal_budget_and_stop_immediately(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder); basis = singleton_state(p)
            ctx.repair_deadline = time.perf_counter()+10.; calls = []
            physical = Feasibility(p, c)
            def complete(node, shape):
                candidate, verdict = ctx.finalize(basis, node, physical)
                calls.append(verdict)
                return candidate
            candidate, audit = search(ctx, basis, native_proposals(p, c, full), ctx.repair_deadline, complete)
            self.assertIsNotNone(candidate, audit)
            self.assertGreater(audit['merge_checks'], c.max_proposals)
            self.assertIsNone(audit['maximum_merge_checks'])
            self.assertEqual(audit['stop_reason'], 'verified_construction')
            self.assertEqual(len(calls), 1)
            layout = serialize_layout(p, candidate, 0, physical.state(candidate, final=True), {'attempted': False})
            verify_result_support(p, {'layouts': [layout]}, c)

    def test_solver_native_first_path_publishes_valid_result_with_tiny_legacy_cap(self):
        from paper_reconstruction_stage4.solver import solve
        from paper_reconstruction_stage4.outputs import build_result, request_identity, validate_result
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            proposals = native_proposals(p, c, full)
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate', return_value=(proposals, {'native_checks': 0})), \
                 patch('paper_reconstruction_stage4.search.joint_beam.JointSearch.solve', side_effect=AssertionError('Unnecessary cold search')):
                result = solve(p, c)
            self.assertEqual(len(result.states[0].components()), 1)
            self.assertTrue(result.diagnostics['native_first_completion'])
            published = build_result(p, c, result, request_identity(p, c))
            validate_result(published); verify_result_support(p, published, c)

    def test_expired_deadline_does_not_claim_graph_impossibility(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            candidate, audit = search(ctx, singleton_state(p), native_proposals(p, c, full), time.perf_counter()-1., lambda *args: self.fail())
            self.assertIsNone(candidate)
            self.assertEqual(audit['stop_reason'], 'overall_deadline')
            self.assertFalse(audit['proposal_graph_exhausted'])

    def test_page_corroborated_bridge_serialization_replays_and_old_guard_remains(self):
        from paper_reconstruction_stage4.native_bridge import contour_arc, check_arcs
        from paper_reconstruction_stage4.native_bridge_v2 import POLICY as OLD_POLICY
        from paper_reconstruction_stage4.repair_search import Node
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder, 3, 3); basis = singleton_state(p)
            corner = min(p.pieces); contacts = []; corner_count = 0
            for row in native_proposals(p, c, full):
                if corner in row['pieces']:
                    corner_count += 1
                    if corner_count > 1:
                        continue
                a, b = row['pieces']; arcs = tuple(contour_arc(p, v) for v in row['arcs'])
                contacts.append({'pieces': row['pieces'], 'arcs': arcs, 'metrics': check_arcs(p, arcs, full.poses, c),
                    'proposal_id': row['proposal_id'], 'role': 'merge', 'source_groups': basis.components(),
                    'merged_groups': [[a], [b]], 'proposed_transform_b_to_a': row['transform_b_to_a']})
            candidate, verdict = ctx.evaluate_contacts(basis, Node(full, contacts), contacts, Feasibility(p, c))
            self.assertIsNotNone(candidate, verdict)
            self.assertEqual(verdict['page_boundary_witnesses'], 1)
            layout = serialize_layout(p, candidate, 0, {'valid': True}, {'attempted': False})
            fresh = replace(p, edges=[], page_context=None, structural=None)
            verify_result_support(fresh, {'layouts': [layout]}, c)
            changed = deepcopy(layout)
            record = next(row['native_bridge'] for row in changed['selected_joins'] if row.get('native_bridge', {}).get('policy', '').endswith('.v3'))
            record['page_boundary_witness']['body_line_low'][0] -= 10.
            with self.assertRaises(ValueError):
                verify_result_support(replace(p, edges=[], page_context=None), {'layouts': [changed]}, c)
            # Relabeling as the old policy does not disable its independent
            # cycle requirement: it still rejects a sliding graph bridge.
            from paper_reconstruction_stage4.native_bridge import edge_from_record
            from paper_reconstruction_stage4.native_bridge_v2 import verify_layout
            old = deepcopy(layout)
            join = next(row for row in old['selected_joins'] if row.get('native_bridge', {}).get('policy', '').endswith('.v3'))
            record = join['native_bridge']; record.pop('page_boundary_witness'); record['policy'] = OLD_POLICY
            record['bridge_id'] = digest({k: v for k, v in record.items() if k != 'bridge_id'})
            edge, _ = edge_from_record(p, record, c); join['geometry_id'] = edge.id
            with self.assertRaisesRegex(ValueError, 'selected-cycle'):
                verify_layout(p, old, c)

    def test_failed_fast_pass_reopens_alternatives_without_node_count_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            candidate, audit = search(ctx, singleton_state(p), native_proposals(p, c, full), time.perf_counter()+1., lambda *args: None)
            self.assertIsNone(candidate)
            self.assertGreater(audit['backtracking_seed_sessions_started'], 0)
            self.assertGreater(audit['backtracks'], 0)
            self.assertTrue(audit['fast_pass_pruning_is_reopened'])

    def test_support_search_defers_and_resumes_the_same_alternative_stack(self):
        from paper_reconstruction_stage4.native_bridge import contour_arc
        from paper_reconstruction_stage4.repair_search import Node
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            basis = singleton_state(p); proposals = native_proposals(p, c, full)
            contacts = []
            for row in proposals[:3]:
                a, b = row['pieces']
                contacts.append({'pieces': row['pieces'], 'arcs': tuple(contour_arc(p, v) for v in row['arcs']),
                    'proposal_id': row['proposal_id'], 'role': 'merge', 'source_groups': basis.components(),
                    'merged_groups': [[a], [b]], 'proposed_transform_b_to_a': row['transform_b_to_a']})
            clock = [0.]; ctx.repair_deadline = 1000.
            def failed(*args):
                clock[0] += 10.
                return None, {'disposition': 'rejected_native_support', 'reason': 'try another support assignment'}
            with patch('paper_reconstruction_stage4.native_repair_v18.time.perf_counter', side_effect=lambda: clock[0]), \
                 patch.object(ctx, 'evaluate_contacts', side_effect=failed):
                first, audit1 = ctx.finalize(basis, Node(full, contacts), Feasibility(p, c))
                self.assertEqual(audit1['disposition'], 'deferred_native_support')
                job = next(iter(ctx.support_jobs.values())); iterator = job['iterator']; tested = len(job['tested'])
                second, audit2 = ctx.finalize(basis, Node(full, contacts), Feasibility(p, c))
            self.assertIs(job['iterator'], iterator)
            self.assertGreater(len(job['tested']), tested)
            self.assertEqual(audit2['support_selection']['resumptions'], 2)
            self.assertIsNone(audit2['support_selection']['maximum_nodes'])


if __name__ == '__main__':
    unittest.main()
