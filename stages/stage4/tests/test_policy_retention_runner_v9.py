import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark.common import atomic_json,digest,file_hash,read_json
from benchmark.policy_retention_experiments import VARIANTS,validate_variant,serving_barrier,checked_v8
from paper_reconstruction_stage4.config import SolverConfig

ROOT=Path(__file__).resolve().parents[1]


class RetentionRunnerTests(unittest.TestCase):
    def baseline(self):
        return read_json(ROOT/'controls/stage4_v8_support_aware_development_r2/configs/joint_v8_combined_development.json')

    def test_named_presets_expose_budgets_and_preserve_physical_and_score_rules(self):
        base=self.baseline()
        for name,filename in VARIANTS.items():
            value=SolverConfig.from_dict(read_json(ROOT/'configs'/filename)).to_dict()
            validate_variant(name,value,base)
            self.assertEqual(value['maximum_assembly_collision'],base['maximum_assembly_collision'])
            self.assertEqual(value['continuation_weight'],base['continuation_weight'])
            self.assertEqual(value['max_proposals'],20000 if name=='capped' else 40000)
            for key in ('max_proposals','maximum_assembly_collision','continuation_weight','joint_incumbent_limit'):
                changed={**value,key:value[key]+1}
                with self.assertRaises(ValueError):validate_variant(name,changed,base)

    def test_reference_access_waits_for_every_variant_and_sealed_request(self):
        request={'schema_version':'stage4.v9-policy-retention-request.v1','case_ids':['a','b'],'variants':['retained','checked']}
        request['request_sha256']=digest(request)
        state={'phase':'diagnostics','cases':{cid:{v:{'status':'predicted'} for v in request['variants']} for cid in request['case_ids']}}
        serving_barrier(request,state)
        missing=copy.deepcopy(state);missing['cases']['b'].pop('checked')
        with self.assertRaisesRegex(ValueError,'incomplete'):serving_barrier(request,missing)
        with self.assertRaisesRegex(ValueError,'not reached'):serving_barrier(request,{**state,'phase':'predictions'})
        with self.assertRaisesRegex(ValueError,'changed'):serving_barrier({**request,'case_ids':['a']},state)

    def test_v8_reference_binds_request_receipt_summary_controller_and_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'v8';root.mkdir();baseline=Path(folder)/'v6';baseline.mkdir()
            integration=Path(folder)/'integration';integration.mkdir()
            config=root/'config.json';atomic_json(config,self.baseline())
            atomic_json(baseline/'experiment_receipt.json',{'old':True})
            atomic_json(integration/'integration_receipt.json',{'input':True})
            atomic_json(root/'report/summary.json',{'report':True})
            controller=root/'controller.py';controller.write_text('# bound')
            target=root/'cases/a/combined/stage4';atomic_json(target/'result.json',{'prediction':True})
            request={'schema_version':'stage4.v8-support-aware-request.v1','baseline':str(baseline),'baseline_request_sha256':'old',
                'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),
                'integration':str(integration),'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
                'snapshot':'unused','controllers':{str(controller):file_hash(controller)},
                'configurations':{'combined':{'path':str(config),'sha256':file_hash(config)}},
                'case_ids':['a'],'implementation':{'sha256':'source'}}
            request['request_sha256']=digest(request)
            atomic_json(root/'request.json',request)
            atomic_json(root/'progress.json',{'phase':'finished','cases':{'a':{'combined':{'result_sha256':file_hash(target/'result.json')}}}})
            atomic_json(root/'experiment_receipt.json',{'request_sha256':request['request_sha256'],
                'summary_sha256':file_hash(root/'report/summary.json')})
            result={'implementation':request['implementation'],'configuration':self.baseline(),'input':{'manifest_sha256':'same'}}
            with patch('benchmark.policy_retention_experiments.checked_baseline',return_value=({'request_sha256':'old'},None,None)), \
                 patch('benchmark.policy_retention_experiments.verify_snapshot',return_value={'implementation':request['implementation']}), \
                 patch('benchmark.policy_retention_experiments.read_result',return_value=result):
                checked_v8(root,baseline)
                controller.write_text('# edited')
                with self.assertRaisesRegex(ValueError,'controller changed'):checked_v8(root,baseline)
                controller.write_text('# bound')
                atomic_json(target/'result.json',{'different':True})
                with self.assertRaisesRegex(ValueError,'prediction changed'):checked_v8(root,baseline)


if __name__=='__main__':unittest.main()
