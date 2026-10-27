import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.scale_pipeline_v14 import validate_cohort, commands_for
from benchmark.scale_pipeline_diagnosis_v14 import accurate_graph, main as diagnose


class ScalePipelineTests(unittest.TestCase):
    def cohort(self):
        return {'cases': [{'case_id': f'p{n}_{i}', 'pieces': n} for n in (4, 8, 12, 16) for i in range(2)],
                'capture_profile': {'shared_native_pixel_scale': True, 'per_piece_resize': False,
                    'perspective': False, 'rotation_family': 'quarter_turn', 'stage1_masks': 'generator_exact_bypass'}}

    def test_refuse_wrong_size_or_capture_assumptions(self):
        self.assertEqual(len(validate_cohort(self.cohort())), 8)
        for mutation in ('size', 'capture', 'duplicate'):
            cohort = copy.deepcopy(self.cohort())
            if mutation == 'size': cohort['cases'][-1]['pieces'] = 30
            elif mutation == 'capture': cohort['capture_profile']['per_piece_resize'] = True
            else: cohort['cases'][-1]['case_id'] = cohort['cases'][0]['case_id']
            with self.assertRaises(ValueError): validate_cohort(cohort)

    def test_stage3_writes_are_routed_to_stage4_worker(self):
        run = Path('C:/stage4/outputs/new')
        req = {'cohort_root': 'C:/immutable/captures', 'stage2_cpu_threads': 4,
               'snapshot': 'C:/frozen', 'config': 'C:/frozen/config.json', 'profile': 'C:/frozen/profile.json'}
        commands = dict(commands_for({'case_id': 'case', 'path': 'cases/case'}, run, req))
        native = [str(v) for v in commands['stage3_native']]
        self.assertIn('infer', native)
        self.assertIn(str(run / 'request.json'), native)
        self.assertTrue(native[2].endswith('scale_pipeline_worker_v14.py'))
        self.assertEqual([str(v) for v in commands['stage4']][-1], '--resume')
        self.assertFalse(any('evaluation_only' in str(v) for cmd in commands.values() for v in cmd))

    def test_diagnostics_cannot_open_reference_without_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Even a valid-looking request cannot reach case_diagnosis without a sealed barrier.
            with patch('benchmark.scale_pipeline_diagnosis_v14.read_json', side_effect=[{}, FileNotFoundError('barrier')]), \
                 patch('benchmark.scale_pipeline_diagnosis_v14.checked', return_value={}), \
                 patch('benchmark.scale_pipeline_diagnosis_v14.case_diagnosis') as evaluator:
                with self.assertRaises(FileNotFoundError): diagnose(['--run', str(root)])
                evaluator.assert_not_called()

    def test_tolerance_graph_retains_isolated_pieces(self):
        contacts = [{'piece_a': 'a', 'piece_b': 'b', 'error': 20.}, {'piece_a': 'b', 'piece_b': 'c', 'error': None}]
        strict = accurate_graph(contacts, ['a', 'b', 'c'], 'error', 4)
        loose = accurate_graph(contacts, ['a', 'b', 'c'], 'error', 32)
        self.assertEqual(strict['component_sizes'], [1, 1, 1])
        self.assertEqual(loose['component_sizes'], [2, 1])
        self.assertFalse(loose['connected'])


if __name__ == '__main__':
    unittest.main()
