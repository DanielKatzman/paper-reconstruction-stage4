import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import pipeline


class PipelineCpuTests(unittest.TestCase):
    def test_requested_thread_count_reaches_raw_capture_segmentation(self):
        class StopBeforeInference(Exception):
            pass
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            captures = root / 'captures'
            captures.mkdir()
            profile = root / 'capture.json'
            profile.write_text('{}', encoding='utf-8')
            args = SimpleNamespace(cpu_threads=4, captures=captures, stage1=None,
                mode='shared-scale-capture', capture_profile=profile,
                output=root / 'run', device='cpu', review='none')
            with patch.object(pipeline, 'execute', side_effect=StopBeforeInference) as execute, \
                    self.assertRaises(StopBeforeInference):
                pipeline.run(args)
            stage, argv, _output, label = execute.call_args.args
            self.assertEqual(stage, 'stage1')
            self.assertEqual(label, 'stage1')
            self.assertEqual(argv[argv.index('--cpu-threads') + 1], 4)
            self.assertEqual(pipeline.read(root/'run/solver_config.json'), pipeline.read(pipeline.CONFIG))

    def test_invalid_thread_count_is_rejected_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, 'CPU threads must be positive'):
            pipeline.run(SimpleNamespace(cpu_threads=0))

    def test_invalid_stage4_allowance_rejected_before_input_or_output(self):
        for seconds in (float('nan'), float('inf'), -1., 0.):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, 'Stage 4 seconds'):
                pipeline.run(SimpleNamespace(cpu_threads=4, stage4_seconds=seconds))

    def test_development_allowance_is_saved_before_inference(self):
        class StopBeforeInference(Exception):
            pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);captures=root/'captures';captures.mkdir()
            args=SimpleNamespace(cpu_threads=4,stage4_seconds=1200.,captures=captures,stage1=None,
                mode='shared-scale-capture',capture_profile=root/'profile.json',
                output=root/'run',device='cpu',review='none')
            with patch.object(pipeline,'execute',side_effect=StopBeforeInference), self.assertRaises(StopBeforeInference):
                pipeline.run(args)
            self.assertEqual(pipeline.read(root/'run/solver_config.json')['joint_wall_seconds'],1200.)
            self.assertEqual(pipeline.read(root/'run/request.json')['stage4_wall_seconds'],1200.)


if __name__ == '__main__':
    unittest.main()
