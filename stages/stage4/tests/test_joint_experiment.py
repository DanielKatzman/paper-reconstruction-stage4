from argparse import Namespace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from benchmark.common import atomic_json, digest, inventory, read_json
from benchmark.joint_experiment import make_plan, materialize, run
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.outputs import read_result, request_identity
from paper_reconstruction_stage4.seeds import attach_initial_result
from tests.fixtures import build_fixture

ROOT=Path(__file__).resolve().parents[1]


class ExternalExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        bundle=self.root/'bundle'; reference=build_fixture(bundle,rows=1,cols=2)
        manifest=read_json(bundle/'bundle.json')
        case_root=self.root/'cohort/cases/grid'; stage1=case_root/'stage1'; stage1.mkdir(parents=True)
        rows=[]
        for pid,piece in manifest['pieces'].items():
            shutil.copyfile(bundle/piece['image'],stage1/(pid+'.png'))
            shutil.copyfile(bundle/piece['mask'],stage1/(pid+'_mask.png'))
            rows.append({'capture_id':pid,'piece_path':pid+'.png','piece_mask_path':pid+'_mask.png'})
        (stage1/'segmentation_manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        atomic_json(case_root/'evaluation_only/reference_geometry.json',reference)
        cohort={'schema_version':'stage4.development-benchmark.v1','cases':[{'case_id':'known_grid_fixture','path':'cases/grid','pieces':2,'files':inventory(case_root)}]}
        cohort['benchmark_sha256']=digest(cohort); atomic_json(self.root/'cohort/benchmark.json',cohort)
        atomic_json(self.root/'baseline_config.json',{'solver':'structural_beam_v2','objective_version':'observable-structure.v2','max_seconds':10.})
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_frozen_structural_control.py'),'solve','--bundle',str(bundle),
                        '--config',str(self.root/'baseline_config.json'),'--output',str(self.root/'control')],capture_output=True,check=True)
        control=read_result(self.root/'control')
        self.spec={'schema_version':'stage4.external-experiment.v1','benchmark':str(self.root/'cohort/benchmark.json'),
            'benchmark_sha256':cohort['benchmark_sha256'],'case_id':'known_grid_fixture','bundle':str(bundle),
            'input_manifest_sha256':manifest['manifest_sha256'],'control_result':str(self.root/'control'),
            'control_run_id':control['run_id'],'control_config':str(self.root/'baseline_config.json')}
        self.spec_path=self.root/'spec.json'; atomic_json(self.spec_path,self.spec)

    def tearDown(self):self.temp.cleanup()

    def args(self,**changes):
        values=dict(spec=self.spec_path,output=self.root/'run',variant='baseline',config=None,plan_only=False,diagnose_only=False)
        values.update(changes); return Namespace(**values)

    def test_external_plan_binds_native_inputs_and_never_routes_upstream(self):
        request=make_plan(self.spec_path,self.root/'run','search')
        self.assertEqual(list(request['commands']),['stage4'])
        self.assertFalse(request['stage2_inference_or_publication_verified'])
        self.assertIn('--initial-result',request['commands']['stage4'])
        self.assertFalse((self.root/'run').exists())
        with self.assertRaises(ValueError):make_plan(self.spec_path,self.root/'bundle/forbidden','search')
        self.spec['input_manifest_sha256']='0'*64;atomic_json(self.spec_path,self.spec)
        with self.assertRaisesRegex(ValueError,'identity/case'):make_plan(self.spec_path,self.root/'run','search')

    def test_external_frozen_replay_and_offline_diagnosis_match_control(self):
        self.assertEqual(run(self.args()),0)
        comparison=read_json(self.root/'run/comparison.json')
        self.assertTrue(comparison['baseline_replay_parity']['passed'])
        self.assertTrue(comparison['evaluation_after_prediction'])
        self.assertTrue((self.root/'run/support_intervals.svg').is_file())
        self.assertFalse((self.root/'run/run.lock').exists())

    def test_failed_serving_child_never_opens_diagnostics(self):
        with patch('benchmark.joint_experiment.execute',return_value={'returncode':2}) as child:
            self.assertEqual(run(self.args()),1)
        self.assertEqual(child.call_count,1)
        self.assertEqual(read_json(self.root/'run/progress.json')['phase'],'failed')
        self.assertFalse((self.root/'run/comparison.json').exists())

    def test_interrupted_serving_child_never_opens_diagnostics(self):
        with patch('benchmark.joint_experiment.execute',side_effect=KeyboardInterrupt) as child:
            self.assertEqual(run(self.args()),130)
        self.assertEqual(child.call_count,1)
        self.assertEqual(read_json(self.root/'run/progress.json')['phase'],'interrupted')

    def test_initial_result_is_bound_into_run_identity(self):
        c=SolverConfig(solver='joint_beam_v3',objective_version='observable-structure.v2')
        p=load_problem(self.root/'bundle',c); before=request_identity(p,c)['run_id']
        attach_initial_result(p,self.root/'control'); after=request_identity(p,c)['run_id']
        self.assertNotEqual(before,after)
        self.assertEqual(p.input_summary['initial_result']['run_id'],self.spec['control_run_id'])
        p.manifest['manifest_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'different evidence'):attach_initial_result(p,self.root/'control')


if __name__=='__main__':unittest.main()
