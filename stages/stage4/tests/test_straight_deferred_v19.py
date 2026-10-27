from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.test_native_repair_v18 import fixture
from tests.test_native_repair_v17 import native_proposals
from paper_reconstruction_stage4.config import SolverConfig, STRAIGHT_DEFERRED_GROWTH
from paper_reconstruction_stage4.native_graph_v19 import informative, search
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import serialize_layout, build_result, validate_result, request_identity


class StraightDeferredTests(unittest.TestCase):
    def test_opt_in_preserves_historical_normalized_configuration(self):
        import json
        old = json.loads(Path('configs/joint_v18_geometry_development.json').read_text())
        new = json.loads(Path('configs/joint_v19_straight_deferred_development.json').read_text())
        self.assertEqual(SolverConfig.from_dict(old).to_dict(), old)
        self.assertEqual({k: v for k, v in new.items() if k != 'native_growth_policy'}, old)
        self.assertEqual(SolverConfig.from_dict(new).native_growth_policy, STRAIGHT_DEFERRED_GROWTH)
        with self.assertRaises(ValueError):
            replace(SolverConfig(), native_growth_policy=STRAIGHT_DEFERRED_GROWTH).validate()

    def test_slightly_curved_true_seams_are_kept_and_unknown_spread_is_not_confidence(self):
        self.assertTrue(informative({'shape_spread_ratio': .0039}))
        for row in ({}, {'shape_spread_ratio': 0.}, {'shape_spread_ratio': float('nan')}):
            self.assertFalse(informative(row))

    def test_genuine_straight_cuts_still_assemble_and_replay(self):
        from paper_reconstruction_stage4.solver import solve
        with tempfile.TemporaryDirectory() as folder:
            p, c, _, full = fixture(folder)
            c = replace(c, native_growth_policy=STRAIGHT_DEFERRED_GROWTH)
            rows = native_proposals(p, c, full)
            self.assertTrue(rows)
            self.assertTrue(all(not informative(r) for r in rows))
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate', return_value=(rows, {'native_checks': 0})), \
                 patch('paper_reconstruction_stage4.search.joint_beam.JointSearch.solve', side_effect=AssertionError('Unneeded legacy search')):
                result = solve(p, c)
            self.assertEqual(result.termination, 'verified_construction')
            self.assertEqual(len(result.states[0].components()), 1)
            published = build_result(p, c, result, request_identity(p, c))
            validate_result(published)
            self.assertTrue(verify_result_support(p, published, c)['valid'])
            audit = result.diagnostics['native_repair']['merge_searches'][0]
            self.assertEqual(audit['growth_lanes'][0]['growth_lane'], 'all_modes_fallback')
            self.assertFalse(audit['straight_modes_permanently_discarded'])

    def test_incomplete_nonstraight_graph_reopens_straight_connecting_seams(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, _, full = fixture(folder)
            c = replace(c, native_growth_policy=STRAIGHT_DEFERRED_GROWTH)
            ctx = make_page_context(p, c, ActiveSupportIndex(p, c))
            p.page_context = ctx; p.structural = ctx.scorer()
            rows = native_proposals(p, c, full)
            # Force only the seed into the ordering view. It cannot span the
            # inventory: all true straight connecting seams must be reopened.
            with patch('paper_reconstruction_stage4.native_graph_v19.informative', side_effect=lambda row: row is rows[0]):
                physical = Feasibility(p, c); basis = singleton_state(p)
                ctx.repair_deadline = time.perf_counter()+10.
                candidate, audit = search(ctx, basis, rows, ctx.repair_deadline,
                    lambda node, shape: ctx.finalize(basis, node, physical)[0], physical)
            self.assertIsNotNone(candidate, audit)
            self.assertEqual([r['growth_lane'] for r in audit['growth_lanes']], ['nonstraight_first', 'all_modes_fallback'])
            self.assertLess(audit['growth_lanes'][0]['completed_depth'], len(p.pieces))
            self.assertEqual(audit['completed_depth'], len(p.pieces))
            layout = serialize_layout(p, candidate, 0, physical.state(candidate, final=True), {'attempted': False})
            self.assertTrue(verify_result_support(p, {'layouts': [layout]}, c)['valid'])

    def test_deferred_route_does_not_make_invalid_native_poses_acceptable(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder, 1, 2)
            rows = deepcopy(native_proposals(p, c, full))
            for row in rows:
                row['transform_b_to_a'][0][2] += 1000.
            candidate, audit = search(ctx, singleton_state(p), rows, time.perf_counter()+2.,
                lambda *args: self.fail('Invalid seam reached final acceptance'))
            self.assertIsNone(candidate)
            self.assertGreater(audit['rejections'].get('native_proposal_replay', 0), 0)

    def test_frozen_v18_controllers_and_predictions_are_untouched(self):
        from scripts.repair_recovery_v18_r2_report import bootstrap
        from benchmark.common import read_json, WORKSPACE
        from benchmark.recovery_experiment_v19 import ARMS, command_for
        request, _ = bootstrap(WORKSPACE/'outputs/v18_geometry_full_r2')
        self.assertNotEqual(ARMS['adaptive'].name, Path(request['configs']['adaptive']).name)
        argv = list(map(str, command_for(WORKSPACE/'outputs/new', request, request['cases'][0], 'adaptive')))
        self.assertNotIn('--initial-result', argv)
        self.assertNotIn(request['cases'][0]['evaluation_reference'], argv)

    def test_expired_deadline_abstains_without_claiming_the_page_is_impossible(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, ctx, full = fixture(folder)
            candidate, audit = search(ctx, singleton_state(p), native_proposals(p, c, full),
                time.perf_counter()-1., lambda *args: self.fail('Deadline already expired'))
            self.assertIsNone(candidate)
            self.assertEqual(audit['merge_checks'], 0)
            self.assertEqual(audit['stop_reason'], 'overall_deadline')
            self.assertFalse(audit['proposal_graph_exhausted'])

    def test_visible_gaps_expand_observed_modes_and_select_improved_verified_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, _, full = fixture(folder)
            c = replace(c, native_growth_policy=STRAIGHT_DEFERRED_GROWTH, native_expand_marginal_geometry=True)
            ctx = make_page_context(p, c, ActiveSupportIndex(p, c)); p.deadline = time.perf_counter()+c.joint_finalize_seconds+20.
            alternative = deepcopy(full); alternative.poses[min(alternative.poses)][0, 2] += 1.
            before = {'unfilled_fraction': .06, 'raster_fraction_uncertainty_estimate': .006, 'score': .96}
            after = {**before, 'unfilled_fraction': .015, 'score': .99}
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate', return_value=([], {'native_checks': 0})) as fit, \
                 patch.object(ctx, 'run_geometry', side_effect=[full, alternative]), \
                 patch('paper_reconstruction_stage4.native_repair_v18.shape_measurement', side_effect=[before, after]):
                result = ctx.solve_geometry(None)
            self.assertIs(result.states[0], alternative)
            self.assertEqual(fit.call_count, 2)
            self.assertTrue(ctx.repair_audit['geometry_expansion']['alternative_improved'])
            self.assertFalse(ctx.repair_audit['geometry_expansion']['reference_geometry_used'])

    def test_worse_alternative_or_expired_deadline_preserves_verified_incumbent(self):
        for expired, alternative_exists in ((False, True), (False, False), (True, False)):
            with self.subTest(expired=expired, alternative_exists=alternative_exists), tempfile.TemporaryDirectory() as folder:
                p, c, _, full = fixture(folder)
                c = replace(c, native_growth_policy=STRAIGHT_DEFERRED_GROWTH, native_expand_marginal_geometry=True)
                ctx = make_page_context(p, c, ActiveSupportIndex(p, c))
                p.deadline = time.perf_counter()+c.joint_finalize_seconds+(-1. if expired else 20.)
                before = {'unfilled_fraction': .06, 'raster_fraction_uncertainty_estimate': .006, 'score': .96}
                after = {**before, 'score': .95}
                states = [full] if expired else [full, deepcopy(full) if alternative_exists else None]
                shapes = [before, before] if expired else [before, after] if alternative_exists else [before]
                with patch('paper_reconstruction_stage4.seam_poses_v18.generate', return_value=([], {'native_checks': 0})), \
                     patch.object(ctx, 'run_geometry', side_effect=states), \
                     patch('paper_reconstruction_stage4.native_repair_v18.shape_measurement', side_effect=shapes):
                    result = ctx.solve_geometry(None)
                self.assertIs(result.states[0], full)
                self.assertFalse(ctx.repair_audit['geometry_expansion']['alternative_improved'])

    def test_geometry_close_to_measurement_floor_stops_without_extra_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            p, c, _, full = fixture(folder)
            c = replace(c, native_growth_policy=STRAIGHT_DEFERRED_GROWTH, native_expand_marginal_geometry=True)
            ctx = make_page_context(p, c, ActiveSupportIndex(p, c)); p.deadline = time.perf_counter()+c.joint_finalize_seconds+20.
            shape = {'unfilled_fraction': .01, 'raster_fraction_uncertainty_estimate': .006, 'score': .99}
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate', return_value=([], {'native_checks': 0})) as fit, \
                 patch.object(ctx, 'run_geometry', return_value=full), \
                 patch('paper_reconstruction_stage4.native_repair_v18.shape_measurement', return_value=shape):
                result = ctx.solve_geometry(None)
            self.assertIs(result.states[0], full)
            self.assertEqual(fit.call_count, 1)


if __name__ == '__main__':
    unittest.main()
