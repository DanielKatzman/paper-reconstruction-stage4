from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark.scale_cohort_v20 import select, validate
from benchmark.scale_extension_v20 import upstream_commands, stage4_command
from benchmark.scale_extension_diagnosis_v20 import requires_probe, main as diagnose
from benchmark.constructibility_v20 import filter_bank


class ScaleExtensionTests(unittest.TestCase):
    def cohort(self):
        return {'cases': [{'case_id': f'p{n}_{i}', 'pieces': n, 'source_page_id': f's{n}_{i}'}
            for n in (20, 24, 30) for i in range(2)], 'capture_profile': {'shared_native_pixel_scale': True,
                'per_piece_resize': False, 'perspective': False, 'rotation_family': 'quarter_turn',
                'stage1_masks': 'generator_exact_bypass'}}

    def test_exact_six_cases_and_shared_scale_are_required(self):
        validate(self.cohort())
        for changed in ('size', 'page', 'resize', 'rotation'):
            c = deepcopy(self.cohort())
            if changed == 'size': c['cases'][-1]['pieces'] = 28
            if changed == 'page': c['cases'][-1]['source_page_id'] = c['cases'][0]['source_page_id']
            if changed == 'resize': c['capture_profile']['per_piece_resize'] = True
            if changed == 'rotation': c['capture_profile']['rotation_family'] = 'arbitrary'
            with self.assertRaises(ValueError): validate(c)

    def test_selection_excludes_control_pages_and_ignores_outcomes(self):
        rows = [{'puzzle_id': f'p{n}_{i}', 'source_id': f's{n}_{i}', 'actual_piece_count': n,
            'difficulty_tier': 'simple' if i%2 == 0 else 'moderate', 'source_family': str(i%2),
            'split': 'train', 'solver_would_pass': i%2 == 0} for n in (20, 24, 30) for i in range(5)]
        first = select(rows, {'s20_0', 's30_1'})
        for row in rows: row['solver_would_pass'] = not row['solver_would_pass']
        second = select(list(reversed(rows)), {'s20_0', 's30_1'})
        self.assertEqual([r['puzzle_id'] for r in first], [r['puzzle_id'] for r in second])
        self.assertFalse({r['source_id'] for r in first}&{'s20_0', 's30_1'})
        self.assertEqual(len({r['source_id'] for r in first}), 6)

    def test_upstream_command_never_runs_stage4_or_passes_reference_paths(self):
        request = {'cohort_root': 'C:/captures', 'stage2_cpu_threads': 4, 'snapshot': 'C:/fixed',
            'config': 'C:/fixed/config.json', 'profile': 'C:/fixed/profile.json'}
        commands = upstream_commands({'case_id': 'p', 'path': 'cases/p'}, Path('C:/stage4/outputs/up'), request)
        self.assertEqual([stage for stage, _ in commands], ['stage2', 'stage2_verify', 'stage3_native', 'stage3_pool'])
        self.assertFalse(any('evaluation_only' in str(arg) for _, command in commands for arg in command))

    def test_cold_consumer_uses_frozen_profile_without_saved_pose_seeds(self):
        request = {'snapshot': 'C:/frozen', 'config': 'C:/frozen/config.json', 'profile': 'C:/frozen/profile.json'}
        case = {'case_id': 'p', 'bundle': 'C:/up/p/stage3', 'capture_profile': 'C:/captures/p/capture_profile.json'}
        argv = list(map(str, stage4_command(Path('C:/stage4/outputs/down'), request, case)))
        self.assertIn('C:/frozen/config.json', argv)
        self.assertNotIn('--initial-result', argv)
        self.assertFalse(any('evaluation_only' in arg for arg in argv))

    def test_reference_measurement_cannot_start_before_prediction_seal(self):
        with patch('benchmark.scale_extension_diagnosis_v20.read_json', side_effect=[{}, FileNotFoundError('seal')]), \
             patch('benchmark.scale_extension_diagnosis_v20.checked', return_value={}), \
             patch('benchmark.scale_extension_diagnosis_v20.case_review') as review:
            with self.assertRaises(FileNotFoundError): diagnose(['--run', 'unused'])
            review.assert_not_called()

    def test_probe_trigger_distinguishes_partial_false_and_approximately_correct_results(self):
        q = {'all_pieces_connected': True, 'common_gauge': {'non_anchor_probe_errors_reference_px': {'p95': 20.}}}
        e = {'false_neighbor_joins': []}; a = {'status': 'supported'}
        self.assertFalse(requires_probe(q, e, a))
        self.assertTrue(requires_probe({**q, 'all_pieces_connected': False}, e, a))
        self.assertTrue(requires_probe(q, {'false_neighbor_joins': [['a', 'b']]}, a))
        self.assertTrue(requires_probe(q, e, {'status': 'unresolved'}))

    def test_accurate_pair_connectivity_is_reported_without_claiming_construction(self):
        import numpy as np
        ref = {'pieces': {p: {} for p in 'abc'}, 'contacts': [{'piece_a': a, 'piece_b': b,
            'points_a': [[0, 0], [1, 1]], 'points_b': [[0, 0], [1, 1]]} for a, b in [('a', 'b'), ('b', 'c')]]}
        proposals = [{'pieces': [c['piece_a'], c['piece_b']], 'transform_b_to_a': np.eye(3).tolist()} for c in ref['contacts']]
        filtered, graph = filter_bank(proposals, ref, 8.)
        self.assertEqual(len(filtered), 2)
        self.assertTrue(graph['connected'])
        self.assertNotIn('compatible_native_assembly_demonstrated', graph)
        wrong = deepcopy(proposals)
        for p in wrong: p['transform_b_to_a'][0][2] = 100.
        filtered, graph = filter_bank(wrong, ref, 8.)
        self.assertEqual(filtered, [])
        self.assertEqual(graph['component_sizes'], [1, 1, 1])

    def test_frozen_solver_and_past_confirmation_controllers_remain_unchanged(self):
        from benchmark.common import WORKSPACE, read_json, file_hash
        from benchmark.snapshots import verify_snapshot
        from paper_reconstruction_stage4.outputs import implementation_identity
        request = read_json(WORKSPACE/'outputs/v19_baseline_confirmation/request.json')
        self.assertEqual(verify_snapshot(request['snapshot'])['implementation'], request['implementation'])
        for path, sha in request['controllers'].items(): self.assertEqual(file_hash(path), sha)

    def test_wrong_piece_inventory_cannot_be_published_as_a_larger_case(self):
        from benchmark.common import WORKSPACE, read_json
        from benchmark.scale_extension_v20 import create_stage4
        source = WORKSPACE/'outputs/v14_full_pipeline_scale_2each'
        old = read_json(source/'request.json'); baseline = read_json(WORKSPACE/'outputs/v19_baseline_confirmation/request.json')
        case = {**old['cases'][0], 'pieces': 20}
        req = {'cases': [case], 'cohort_root': old['cohort_root'], 'config': baseline['configs']['adaptive'],
            'profile': baseline['profile'], 'configuration': baseline['configurations']['adaptive']}
        with patch('benchmark.scale_extension_v20.upstream_binding', return_value=(req, {})):
            with self.assertRaisesRegex(ValueError, 'physical inventory'):
                create_stage4(WORKSPACE/'outputs/unused-test', source)


if __name__ == '__main__':
    unittest.main()
