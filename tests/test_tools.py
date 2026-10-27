"""Boundary tests for new portable orchestration, independent of heavy inference."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import common
import pipeline
from construction import approximate_construction


class PortableRunnerTests(unittest.TestCase):
    def test_environment_cannot_import_an_old_inherited_source_path(self):
        with patch.dict(os.environ,{'PYTHONPATH':'old-sibling-source'}):
            env=common.environment('stage4')
        self.assertNotIn('old-sibling-source',env['PYTHONPATH'])
        self.assertIn(str(common.ROOT/'stages/stage4/src'),env['PYTHONPATH'])

    def test_output_rejects_inputs_and_existing_directories(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'input';source.mkdir()
            for output in [source,source/'nested',root]:
                with self.assertRaises(ValueError):pipeline.fresh(output,[source])

    def test_every_primary_is_sealed_before_any_reference_evaluation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run';events=[]
            cases=[{'case_id':c,'pieces':4,'bundle':'unused','capture_profile':'unused','evaluation_reference':'unused'} for c in ['one','two']]
            original=common.read
            def reader(path):
                return {'cases':cases} if Path(path)==common.ROOT/'baseline/cohort.json' else original(path)
            def solve(bundle,capture,target):
                target.mkdir(parents=True);common.save(target/'result.json',{'primary':target.parent.name});events.append('solve')
            def execute(stage,argv,parent,label):
                self.assertEqual(events.count('solve'),2)
                self.assertTrue((output/'predictions_sealed.json').exists())
                common.save(parent/'evaluation.json',{'supported_connected_zero_false_neighbors':True,'construction_regression_pass':True});events.append('evaluate')
            with patch.object(pipeline,'read',side_effect=reader),patch.object(pipeline,'solve',side_effect=solve),patch.object(pipeline,'execute',side_effect=execute):
                pipeline.benchmark(SimpleNamespace(cases=None,output=output))
            self.assertEqual(events,['solve','solve','evaluate','evaluate'])
            self.assertTrue(common.read(output/'review.json')['predictions_unchanged'])

    def test_evaluation_mutating_a_primary_is_detected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run';original=common.read
            case={'case_id':'one','pieces':4,'bundle':'unused','capture_profile':'unused','evaluation_reference':'unused'}
            def reader(path):return {'cases':[case]} if Path(path)==common.ROOT/'baseline/cohort.json' else original(path)
            def solve(bundle,capture,target):target.mkdir(parents=True);common.save(target/'result.json',{'primary':True})
            def execute(stage,argv,parent,label):
                common.save(parent/'stage4/result.json',{'tampered':True})
                common.save(parent/'evaluation.json',{'supported_connected_zero_false_neighbors':True,'construction_regression_pass':True})
            with patch.object(pipeline,'read',side_effect=reader),patch.object(pipeline,'solve',side_effect=solve),patch.object(pipeline,'execute',side_effect=execute):
                with self.assertRaisesRegex(ValueError,'Predictions changed'):pipeline.benchmark(SimpleNamespace(cases=None,output=output))

    def quality(self,error=30.,angle=0.,scale=1.):
        return {'inventory_coverage':1.,'common_gauge':{'whole_page_error_available':True,
            'non_anchor_probe_errors_reference_px':{'maximum':error},
            'per_piece':[{'local_orientation_error_degrees':angle,'local_scale_ratio':scale}]}}

    def test_approximate_correct_placement_does_not_need_four_pixel_precision(self):
        self.assertTrue(approximate_construction(self.quality()))

    def test_correct_neighbors_cannot_hide_a_wrong_orientation(self):
        self.assertFalse(approximate_construction(self.quality(angle=90.)))

    def test_correct_neighbors_cannot_hide_a_scaled_or_far_displaced_piece(self):
        self.assertFalse(approximate_construction(self.quality(scale=1.05)))
        self.assertFalse(approximate_construction(self.quality(error=500.)))

    def test_missing_disconnected_or_nonfinite_measurements_fail_closed(self):
        q=self.quality();q['common_gauge']['whole_page_error_available']=False
        self.assertFalse(approximate_construction(q))
        self.assertFalse(approximate_construction(self.quality(error=float('nan'))))


if __name__=='__main__':unittest.main()
