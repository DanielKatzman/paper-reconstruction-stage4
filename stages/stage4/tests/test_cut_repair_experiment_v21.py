from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark.common import atomic_json,digest,file_hash
from benchmark.cut_repair_experiment_v21_r4 import command_for,authenticate


class CutRepairControllerTests(unittest.TestCase):
    def test_cold_worker_does_not_receive_saved_layouts_or_references(self):
        request={'snapshot':'C:/frozen','config':'C:/frozen/config.json','profile':'C:/frozen/profile.json'}
        case={'case_id':'case','bundle':'C:/sealed/stage3','capture_profile':'C:/capture.json',
            'baseline':'C:/old/stage4','evaluation_reference':'C:/protected/reference.json'}
        argv=list(map(str,command_for(Path('C:/new'),request,case)))
        self.assertNotIn('--initial-result',argv)
        self.assertNotIn(case['baseline'],argv)
        self.assertNotIn(case['evaluation_reference'],argv)
        self.assertIn(case['bundle'],argv)
        self.assertIn(case['capture_profile'],argv)

    def test_tampered_serving_input_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input';path.write_bytes(b'original')
            request={'snapshot':'unused','implementation':{},'controllers':{},'bound_files':{str(path):file_hash(path)}}
            request['request_sha256']=digest(request)
            path.write_bytes(b'changed')
            with patch('benchmark.cut_repair_experiment_v21_r4.verify_snapshot',return_value={'implementation':{}}),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'Bound file changed'):
                    authenticate(request)

    def test_reference_diagnosis_cannot_start_before_prediction_barrier(self):
        from benchmark.cut_repair_diagnosis_v21_r4 import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request={};request['request_sha256']=digest(request)
            barrier={'request_sha256':request['request_sha256'],'references_opened':True}
            barrier['barrier_sha256']=digest(barrier)
            atomic_json(root/'request.json',request);atomic_json(root/'predictions_sealed.json',barrier)
            atomic_json(root/'progress.json',{})
            with patch('benchmark.cut_repair_diagnosis_v21_r4.authenticate',side_effect=AssertionError('Past the barrier')):
                with self.assertRaisesRegex(ValueError,'Unsealed predictions'):
                    main(['--run',str(root)])


if __name__=='__main__':unittest.main()
