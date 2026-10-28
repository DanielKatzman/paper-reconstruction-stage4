from contextlib import redirect_stderr
import io
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_stage4_capture as runner


class CaptureRunnerTests(unittest.TestCase):
    def test_previous_assembly_is_not_an_accepted_input(self):
        argv=['run_stage4_capture.py','--upstream','unused','--output','unused','--initial-result','old-layout']
        with patch.object(sys,'argv',argv),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:runner.main()
        self.assertEqual(error.exception.code,2)

    def test_existing_output_is_rejected_before_configuration_or_saved_results_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);upstream=root/'upstream';(upstream/'stage3').mkdir(parents=True);(upstream/'stage1').mkdir()
            (upstream/'stage3/bundle.json').write_text('{}');(upstream/'stage1/capture_profile.json').write_text('{}')
            output=root/'outputs/existing';output.mkdir(parents=True);(output/'result.json').write_text('old-primary')
            argv=['run_stage4_capture.py','--upstream',str(upstream),'--output',str(output)]
            with patch.object(runner,'ROOT',root),patch.object(sys,'argv',argv),patch.object(runner,'read') as read,redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):runner.main()
            read.assert_not_called();self.assertEqual((output/'result.json').read_text(),'old-primary')

    def test_relative_paths_use_release_root_not_an_older_desktop_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(runner,'ROOT',root):
                self.assertEqual(runner.resolve(Path('outputs/fresh')),root/'outputs/fresh')


if __name__=='__main__':unittest.main()
