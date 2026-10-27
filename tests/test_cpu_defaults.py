"""Ensure public orchestration defaults do not require a GPU."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import pipeline
import stage3_worker


class CpuDefaults(unittest.TestCase):
    def test_public_pipeline_defaults_to_cpu(self):
        args=['pipeline.py','run','--stage1','unused','--output','unused-output']
        with patch.object(sys,'argv',args),patch.object(pipeline,'run') as run:
            pipeline.main()
        self.assertEqual(run.call_args.args[0].device,'cpu')

    def test_cuda_remains_an_explicit_option(self):
        args=['pipeline.py','run','--stage1','unused','--output','unused-output','--device','cuda']
        with patch.object(sys,'argv',args),patch.object(pipeline,'run') as run:
            pipeline.main()
        self.assertEqual(run.call_args.args[0].device,'cuda')

    def test_setup_defaults_to_cpu_only_torch(self):
        text=(ROOT/'tools/setup.ps1').read_text(encoding='utf-8')
        self.assertIn("$Device = 'cpu'",text)
        self.assertIn('https://download.pytorch.org/whl/cpu',text)
        self.assertIn('2.7.1+cpu',text)
        self.assertIn('check_cpu_runtime.py',text)


if __name__=='__main__':unittest.main()
