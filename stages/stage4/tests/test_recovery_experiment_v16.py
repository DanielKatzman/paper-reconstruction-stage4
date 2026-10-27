from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from benchmark.common import digest, read_json, WORKSPACE
from benchmark.recovery_experiment_v16 import ARMS, command_for
from benchmark.recovery_diagnosis_v16 import main as diagnose, write_report
from benchmark.recovery_gallery_v16 import rejected_gallery
from paper_reconstruction_stage4.page_geometry import state_record
from paper_reconstruction_stage4.config import SolverConfig, REPAIR_V16_POLICY
from tests.test_native_repair_v16 import context


class RecoveryExperimentV16Tests(unittest.TestCase):
    def test_explicit_profile_changes_only_the_declared_bounded_repair_envelope(self):
        base = read_json(WORKSPACE/'configs/joint_v15_combined_development.json')
        profile = read_json(ARMS['portfolio'])
        changed = {key for key in base if base[key] != profile[key]}
        self.assertEqual(changed, {'native_repair_policy', 'repair_source_limit', 'repair_seconds', 'repair_merge_checks'})
        self.assertEqual(SolverConfig.from_dict(profile).native_repair_policy, REPAIR_V16_POLICY)
        self.assertEqual(profile['repair_native_checks'], 768)
        self.assertEqual(profile['joint_wall_seconds'], base['joint_wall_seconds'])
        self.assertEqual(profile['repair_complete_checks'], base['repair_complete_checks'])

    def test_command_keeps_old_layouts_comparison_and_references_out_of_inference(self):
        request = {'snapshot': 'C:/frozen/v16', 'configs': {'portfolio': 'C:/frozen/config.json'}, 'profile': 'C:/frozen/profile.json',
                   'comparison_v15': {'root': 'C:/old-v15'}}
        case = {'case_id': 'case', 'bundle': 'C:/sealed/stage3', 'capture_profile': 'C:/capture.json',
                'baseline': 'C:/old-solution', 'evaluation_reference': 'C:/closed-reference.json'}
        argv = [str(value) for value in command_for(Path('C:/new'), request, case, 'portfolio')]
        self.assertEqual(argv[argv.index('--bundle')+1], case['bundle'])
        self.assertFalse(any(value in argv for value in ['C:/old-v15', case['baseline'], case['evaluation_reference']]))
        self.assertIn('solve', argv); self.assertNotIn('infer', argv)

    def test_diagnosis_cannot_open_evaluation_without_prediction_seal(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('benchmark.recovery_diagnosis_v16.read_json', side_effect=[{}, FileNotFoundError('missing seal')]), \
             patch('benchmark.recovery_diagnosis_v16.checked', return_value={}), \
             patch('benchmark.recovery_diagnosis_v16.measure') as measure:
            with self.assertRaises(FileNotFoundError):
                diagnose(['--run', folder])
            measure.assert_not_called()

    def test_rejected_gallery_preserves_full_poses_and_measures_without_selecting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p, c, ctx, source, full = context(folder)
            reference = root/'reference.json'
            reference.write_text(json.dumps({'pieces': {pid: {'native_to_reference': h.tolist()} for pid, h in full.poses.items()}}))
            record = {'candidate_id': 'failed', 'diagnostic_state': state_record(full),
                      'disposition': 'rejected_native_support', 'reason': 'missing witness',
                      'unsupported_bridges': [{'bridge_ids': ['exact-bridge'], 'cut': ['piece_00']}], 'lane': 'observed_area'}
            row = {'configuration': c.to_dict(), 'search': {'native_repair': {'rejected_complete_candidates': [record]}}}
            before = digest(row)
            with patch('paper_reconstruction_stage4.inputs.load_problem', return_value=p):
                index = rejected_gallery(root/'gallery', {'bundle': 'unused', 'evaluation_reference': str(reference)}, row)
            output = read_json(index.parent/'candidates.json')
            self.assertEqual(digest(row), before); self.assertFalse(output['primary_changed'])
            self.assertAlmostEqual(output['records'][0]['whole_page_p95_reference_px'], 0.)
            self.assertEqual(output['records'][0]['unsupported_bridges'][0]['bridge_ids'], ['exact-bridge'])
            self.assertTrue((index.parent/'candidate_001.png').exists())

    def test_report_uses_sealed_v15_preview_paths_without_new_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); report = root/'run/report'; (report/'case').mkdir(parents=True)
            old = root/'old'; old.mkdir(); Image.new('RGB', (20, 20), 'white').save(old/'preview.png')
            measurement = {'status': 'predicted', 'quality': {'component_sizes': [4], 'common_gauge': {
                'whole_page_error_available': True, 'non_anchor_probe_errors_reference_px': {'p95': 1.}}},
                'assessment': {'status': 'supported'}, 'evaluation': {'false_neighbor_joins': []},
                'native': {'native_checks': 0, 'pairs_with_fitted_checks': 0, 'possible_physical_pairs': 6, 'generation_budget_abstained': False},
                'retained': {'complete_candidates': 1}, 'reference_bound_diagnostic': {'reference_admitted': True},
                'stage4_subprocess_seconds': 1., 'reused_comparison_result': str(old)}
            rows = [{'case_id': 'case', 'variants': {'v15_combined': measurement, 'portfolio': {'status': 'failed'}}}]
            before = digest(rows)
            write_report(root/'run', {'cases': [{'case_id': 'case', 'baseline': str(old), 'reference_image': str(old/'preview.png')}]}, rows)
            self.assertEqual(digest(rows), before)
            html = (report/'index.html').read_text()
            self.assertIn('v15_combined', html); self.assertIn('only the portfolio column is new', html)
            self.assertNotIn('arms/v15_combined', html)


if __name__ == '__main__':
    unittest.main()
