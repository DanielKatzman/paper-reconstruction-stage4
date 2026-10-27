from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.common import atomic_json, read_json
from benchmark.recovery_report_v18_r2_fix import write_closure_gallery, main as diagnose
from scripts.repair_recovery_v18_r2_report import bootstrap, sha


class RecoveryReportRepairTests(unittest.TestCase):
    def test_native_first_route_renders_without_legacy_timing_or_omitted_count(self):
        audit = {'policy': 'native-component-closure.v1', 'sources': [], 'proposals': [],
            'generated_proposals': 0, 'fine_checks': 0, 'verified_closures': 0,
            'budget_abstained': False}
        result = {'run_id': 'native-first', 'search': {'component_closure': audit}}
        original = deepcopy(result)
        with tempfile.TemporaryDirectory() as folder:
            gallery = write_closure_gallery(Path(folder), {'case_id': 'case'}, result)
            html = Path(gallery['index']).read_text(encoding='utf-8')
            data = read_json(Path(gallery['index']).with_name('placements.json'))
        self.assertEqual(gallery['route_status'], 'not_run')
        self.assertIn('was not run', html)
        self.assertIn('unavailable', html)
        self.assertNotIn('elapsed_seconds', data['audit'])
        self.assertNotIn('sources_omitted_by_limit', data['audit'])
        self.assertEqual(result, original)

    def test_measured_route_keeps_real_timings_and_native_evidence(self):
        audit = {'generated_proposals': 1, 'fine_checks': 1, 'verified_closures': 0,
            'elapsed_seconds': 1.25, 'sources_omitted_by_limit': 3, 'budget_abstained': False,
            'proposals': [{'disposition': 'rejected_native_bridge', 'coarse_geometry': {'score': .9},
                'fine_geometry': {'fill_fraction': .92}, 'bridges': [], 'reason': 'native support failed'}]}
        with tempfile.TemporaryDirectory() as folder:
            value = write_closure_gallery(Path(folder), {'case_id': 'case'},
                {'run_id': 'measured', 'search': {'component_closure': audit}})
            html = Path(value['index']).read_text(encoding='utf-8')
            saved = read_json(Path(value['index']).with_name('placements.json'))
        self.assertEqual(value['route_status'], 'activity_recorded')
        self.assertIn('1.250', html)
        self.assertIn('native support failed', html)
        self.assertEqual(saved['audit'], audit)

    def test_absent_audit_remains_unavailable_instead_of_fabricating_zero_work(self):
        with tempfile.TemporaryDirectory() as folder:
            value = write_closure_gallery(Path(folder), {'case_id': 'case'}, {'run_id': 'unknown', 'search': {}})
            saved = read_json(Path(value['index']).with_name('placements.json'))
        self.assertEqual(value['route_status'], 'not_recorded')
        self.assertIsNone(value['generated'])
        self.assertIsNone(saved['audit'])

    def test_missing_prediction_barrier_blocks_diagnosis_before_reference_measurement(self):
        with patch('benchmark.recovery_report_v18_r2_fix.read_json', side_effect=[{}, FileNotFoundError('seal')]), \
             patch('benchmark.recovery_report_v18_r2_fix.checked', return_value={}), \
             patch('benchmark.recovery_report_v18_r2_fix.measure') as measure:
            with self.assertRaises(FileNotFoundError):
                diagnose(['--run', 'unused'])
            measure.assert_not_called()

    def test_repair_bootstrap_rejects_changed_controller_and_prediction_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); snapshot = root/'snapshot'; snapshot.mkdir()
            controller = root/'controller.py'; controller.write_text('original')
            source = snapshot/'source.py'; source.write_text('frozen')
            prediction = root/'result.json'; prediction.write_text('prediction')
            atomic_json(snapshot/'receipt.json', {'files': {'source.py': sha(source)}})
            request = {'schema_version': 'stage4.v18-recovery-ablation-request.v1', 'request_sha256': 'bound',
                'snapshot': str(snapshot), 'controllers': {str(controller): sha(controller)}}
            atomic_json(root/'request.json', request)
            atomic_json(root/'predictions_sealed.json', {'request_sha256': 'bound', 'references_opened': False,
                'files': {str(prediction): sha(prediction)}})
            self.assertEqual(bootstrap(root)[1], snapshot.resolve())
            controller.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'controller changed'):
                bootstrap(root)
            controller.write_text('original'); prediction.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'prediction changed'):
                bootstrap(root)


if __name__ == '__main__':
    unittest.main()
