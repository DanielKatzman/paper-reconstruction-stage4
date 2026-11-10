import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import pipeline


class PipelineResultTests(unittest.TestCase):
    def check_result(self,components,connected,status):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pipeline.save(root/'stage4/result.json',{'status':status,'layouts':[
                {'components':components,'all_pieces_connected':connected}]})
            stream=io.StringIO()
            with redirect_stdout(stream): summary=pipeline.report_pipeline_result(root)
            self.assertEqual(pipeline.read(root/'pipeline_result.json'),summary)
            self.assertFalse(summary['reference_accuracy_checked'])
            return summary,stream.getvalue()

    def test_finished_run_with_disconnected_pieces_is_reported_as_unresolved(self):
        summary,message=self.check_result([['a','b'],['c'],['d']],False,'budget_exhausted')
        self.assertEqual(summary['component_sizes'],[2,1,1])
        self.assertFalse(summary['complete_assembly_candidate'])
        self.assertIn('unresolved assembly',message)

    def test_connected_output_is_a_candidate_without_a_reference_accuracy_claim(self):
        summary,message=self.check_result([['a','b','c','d']],True,'complete_candidate')
        self.assertTrue(summary['complete_assembly_candidate'])
        self.assertIn('complete assembly candidate',message)

    def test_connected_but_unaccepted_output_is_reported_as_unresolved(self):
        for status in ['budget_exhausted','interrupted','partial_candidate']:
            with self.subTest(status=status):
                summary,message=self.check_result([['a','b','c','d']],True,status)
                self.assertFalse(summary['complete_assembly_candidate'])
                self.assertIn('unresolved assembly',message)
