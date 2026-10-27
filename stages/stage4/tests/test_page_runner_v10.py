import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark.common import atomic_json,read_json,digest,file_hash
from benchmark.page_experiment_v10 import preflight,verify_predictions,CONFIG,PROFILE
from paper_reconstruction_stage4.outputs import request_identity,build_result,output_session,publish
from paper_reconstruction_stage4.assessment import write_package
from paper_reconstruction_stage4.solver import solve
from tests.test_page_v10 import fixture


class PageRunnerTests(unittest.TestCase):
    def test_changed_or_unreviewed_request_fails_before_loading_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request={'cases':[]};request['request_sha256']=digest(request)
            atomic_json(root/'request.json',request)
            with patch('benchmark.page_experiment_v10.load_problem') as load:
                with self.assertRaisesRegex(ValueError,'only the reviewed'):preflight(root,CONFIG,PROFILE,['unknown'])
                self.assertFalse(load.called)
            request['request_sha256']='tampered';atomic_json(root/'request.json',request)
            with self.assertRaisesRegex(ValueError,'Changed digest'):preflight(root,CONFIG,PROFILE,['unknown'])

    def test_prediction_barrier_requires_all_packages_and_exact_result_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p,c,_,_,_=fixture(folder);result=solve(p,c);req=request_identity(p,c)
            out=root/'run/cases/toy/stage4';value=build_result(p,c,result,req)
            with output_session(out,req) as work:publish(out,work,value,p,result.states)
            case={'case_id':'toy','bundle_sha256':p.manifest['manifest_sha256'],
                  'control_result':str(out),'control_result_sha256':file_hash(out/'result.json'),
                  'constraint_binding':p.constraint_profile}
            request={'snapshot':'unused','implementation':req['implementation'],'controllers':{},'bound_files':{},
                     'preflight':{'cases':[case]},'configuration':c.to_dict()}
            with patch('benchmark.page_experiment_v10.verify_snapshot',return_value={'implementation':req['implementation']}):
                with self.assertRaises(FileNotFoundError):verify_predictions(root/'run',request)
                write_package(p,c,out)
                write_package(p,c,out,out.parent/'baseline_assessment')
                files=verify_predictions(root/'run',request)
                self.assertEqual(len(files),11)
                changed=copy.deepcopy(request);changed['preflight']['cases'][0]['bundle_sha256']='different'
                with self.assertRaisesRegex(ValueError,'prediction binding'):verify_predictions(root/'run',changed)
                (out.with_name('stage4.assessment')/'report.md').write_text('tampered')
                with self.assertRaisesRegex(ValueError,'file changed'):verify_predictions(root/'run',request)

    def test_recovery_budget_is_preserved_in_snapshot_configuration(self):
        from paper_reconstruction_stage4.config import SolverConfig
        raw=read_json(CONFIG);config=SolverConfig.from_dict(raw)
        self.assertEqual(config.to_dict(),raw)
        self.assertEqual(config.max_proposals,40000)
        self.assertFalse(config.refine)
        self.assertEqual(config.recovery_region_limit*config.recovery_poses_per_region,192)
        self.assertEqual(read_json(PROFILE)['page_aspect_ratio'],'unknown')


if __name__=='__main__':unittest.main()
