from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.common import digest, atomic_json, read_json, WORKSPACE
from benchmark.recovery_experiment_v17 import ARMS, command_for
from benchmark.recovery_diagnosis_v17 import main as diagnose, write_report
from benchmark.recovery_gallery_v17 import unaccepted_gallery
from tests.test_native_repair_v17 import context
from paper_reconstruction_stage4.page_geometry import state_record
from paper_reconstruction_stage4.config import SolverConfig, REPAIR_V17_POLICY


class RecoveryExperimentV17Tests(unittest.TestCase):
    def test_explicit_profile_preserves_native_measurements_and_outer_envelope(self):
        old = read_json(WORKSPACE/'configs/joint_v16_recovery_portfolio_development.json')
        new = read_json(ARMS['adaptive'])
        self.assertEqual({key for key in old if old[key] != new[key]}, {'native_repair_policy', 'repair_seconds'})
        self.assertEqual(SolverConfig.from_dict(new).native_repair_policy, REPAIR_V17_POLICY)
        self.assertEqual(new['joint_wall_seconds'], 240.)
        self.assertEqual(new['repair_native_checks'], 768)
        self.assertEqual(new['repair_seconds'], 75.)

    def test_worker_command_has_no_comparison_layouts_or_references(self):
        request = {'snapshot': 'C:/frozen/v17', 'configs': {'adaptive': 'C:/frozen/config.json'},
                   'profile': 'C:/frozen/profile.json', 'comparison_v16': {'root': 'C:/old-v16'}}
        case = {'case_id': 'case', 'bundle': 'C:/sealed/stage3', 'capture_profile': 'C:/capture.json',
                'baseline': 'C:/old-solution', 'evaluation_reference': 'C:/closed-reference.json'}
        argv = [str(v) for v in command_for(Path('C:/new'), request, case, 'adaptive')]
        self.assertIn('solve', argv); self.assertIn(case['bundle'], argv)
        self.assertFalse(any(v in argv for v in [case['baseline'], case['evaluation_reference'], 'C:/old-v16']))

    def test_evaluation_cannot_start_before_all_predictions_are_sealed(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('benchmark.recovery_diagnosis_v17.read_json', side_effect=[{}, FileNotFoundError('missing barrier')]), \
             patch('benchmark.recovery_diagnosis_v17.checked', return_value={}), \
             patch('benchmark.recovery_diagnosis_v17.measure') as measure:
            with self.assertRaises(FileNotFoundError):
                diagnose(['--run', folder])
            measure.assert_not_called()

    def test_gallery_distinguishes_full_poses_from_partial_growth_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p, c, ctx, source, full = context(folder)
            reference = root/'reference.json'
            atomic_json(reference, {'pieces': {pid: {'native_to_reference': h.tolist()} for pid, h in full.poses.items()}})
            row = {'configuration': c.to_dict(), 'search': {'native_repair': {'final_checks': [], 'merge_searches': [
                {'lane': 'independent_native_graph', 'coarse_complete_records': [{'candidate_id': 'full', 'state': state_record(full)}],
                 'observed_growth_states': [{'state': state_record(source)}]}]}}}
            before = digest(row)
            with patch('paper_reconstruction_stage4.inputs.load_problem', return_value=p):
                index = unaccepted_gallery(root/'gallery', {'bundle': 'unused', 'evaluation_reference': str(reference)}, row)
            result = read_json(index.parent/'candidates.json')
            self.assertEqual(digest(row), before); self.assertFalse(result['primary_changed'])
            self.assertEqual(len(result['records']), 2)
            self.assertTrue(result['records'][0]['whole_page_error_available'])
            self.assertFalse(result['records'][1]['whole_page_error_available'])
            self.assertEqual(len(result['records'][1]['alignment_scope_piece_ids']), 2)
            self.assertTrue((index.parent/'candidate_002.png').exists())

    def test_empty_gallery_never_opens_an_evaluation_file(self):
        with tempfile.TemporaryDirectory() as folder, patch('benchmark.recovery_gallery_v17.read_json') as reader:
            row = {'search': {'native_repair': {'final_checks': [], 'merge_searches': []}}}
            index = unaccepted_gallery(Path(folder), {}, row)
            self.assertTrue(index.exists()); reader.assert_not_called()

    def test_report_labels_reused_results_and_new_envelope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'report/case').mkdir(parents=True)
            old = {'status': 'predicted', 'reused_comparison_result': str(root/'old'),
                'quality': {'component_sizes': [4], 'common_gauge': {'whole_page_error_available': True,
                    'non_anchor_probe_errors_reference_px': {'p95': 1.}}},
                'assessment': {'status': 'supported'}, 'evaluation': {'false_neighbor_joins': []},
                'native': {'pairs_with_fitted_checks': 0, 'possible_physical_pairs': 6, 'generation_budget_abstained': False},
                'retained': {'complete_candidates': 1}, 'reference_bound_diagnostic': {'reference_admitted': True}}
            rows = [{'case_id': 'case', 'variants': {'adaptive': {'status': 'failed'}, 'v16_portfolio': old}}]
            before = digest(rows)
            write_report(root, {'cases': [{'case_id': 'case', 'baseline': str(root/'old'), 'reference_image': str(root/'ref.png')}]}, rows)
            html = (root/'report/index.html').read_text()
            self.assertEqual(digest(rows), before)
            self.assertIn('v16_portfolio', html); self.assertIn('only the adaptive column is new', html)
            self.assertIn('20,000 aggregate checks', html)


if __name__ == '__main__':
    unittest.main()
