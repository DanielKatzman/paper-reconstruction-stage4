from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from tests.test_native_repair_v18 import fixture
from tests.test_native_repair_v17 import native_proposals
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.native_graph_v18 import search
from paper_reconstruction_stage4.outputs import digest


class GeometryRevisionTwoTests(unittest.TestCase):
    def test_expansion_is_triggered_by_missing_verified_construction_before_cold_search(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            p.deadline = time.perf_counter()+100.
            initial = native_proposals(p, c, full)[:1]; expanded = native_proposals(p, c, full)
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate', side_effect=[
                (initial, {'native_checks': 1}), (expanded, {'native_checks': 12})]) as fits, \
                 patch.object(ctx, 'run_geometry', side_effect=[None, full]) as geometry:
                result = ctx.solve_geometry(None)
            self.assertIsNotNone(result)
            self.assertEqual(fits.call_count, 2)
            self.assertIsNone(fits.call_args_list[1].kwargs['maximum_checks'])
            self.assertEqual(fits.call_args_list[1].kwargs['peaks'], 8)
            self.assertTrue(ctx.expanded_generation_done)
            self.assertEqual(geometry.call_count, 2)

    def test_a_correct_lower_ranked_mode_is_still_reachable(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder, 1, 2)
            correct = native_proposals(p, c, full)[0]; rows = []
            for i in range(4):
                bad = deepcopy(correct)
                bad['transform_b_to_a'][0][2] += 40.*(i+1)
                bad.update(score=1000.+i, native_length_px=1000.+i, conditioned=True)
                bad['proposal_id'] = digest(bad); rows.append(bad)
            candidate, audit = search(ctx, singleton_state(p), [*rows, correct], time.perf_counter()+5.,
                lambda node, shape: node.state)
            self.assertIsNotNone(candidate)
            self.assertGreaterEqual(audit['native_proposal_replay_checks'], 5)
            self.assertFalse(audit['lower_modes_permanently_discarded'])

    def test_unresolved_overall_deadline_does_not_report_old_batch_exhaustion(self):
        from paper_reconstruction_stage4.solver import solve
        from paper_reconstruction_stage4.search.common import SearchResult
        from paper_reconstruction_stage4.native_repair_v18 import RepairContext
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            p.deadline = time.perf_counter()-1.
            basis = singleton_state(p)
            with patch.object(RepairContext, 'solve_geometry', return_value=SearchResult([basis], {'proposals': 1}, 'proposal_budget')), \
                 patch.object(RepairContext, 'prioritize_finalists', return_value=([basis], {})):
                result = solve(p, c)
            self.assertEqual(result.termination, 'time_budget')
            self.assertEqual(result.diagnostics['legacy_ordering_batch_termination'], 'proposal_budget')

    def test_new_runner_preserves_the_bound_first_revision_controllers(self):
        from benchmark.common import read_json, file_hash, WORKSPACE
        from benchmark.recovery_experiment_v18_r2 import SNAPSHOT, command_for
        self.assertTrue(str(SNAPSHOT).endswith('stage4_v18_geometry_development_r2'))
        request = read_json(WORKSPACE/'outputs/v18_geometry_full/request.json')
        for path, sha in request['controllers'].items():
            self.assertEqual(file_hash(path), sha)
        argv = list(map(str, command_for(WORKSPACE/'outputs/new', request, request['cases'][0], 'adaptive')))
        self.assertNotIn('--initial-result', argv)


if __name__ == '__main__':
    unittest.main()

