import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from paper_reconstruction_stage4.constraints import bind_profile,POLICY
from paper_reconstruction_stage4.outputs import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1]
EXACT=ROOT/'configs/shared_scale_complete_rectangle_v1.json'
PREDICTED=ROOT/'configs/shared_scale_complete_rectangle_predicted_v1.json'


class CaptureProfiles(unittest.TestCase):
    def bind(self,folder,origin='segformer_v3_predicted',profile=PREDICTED,updates=None,bound=True):
        path=Path(folder)/'capture.json'
        value={'schema_version':'stage1.capture-profile.v1','motion_model':'rotation_translation',
            'relative_piece_scale':1.,'shared_native_pixel_scale':True,'per_piece_resize':False,
            'perspective':False,'stage1_masks':origin,'mask_perturbation':False}
        value.update(updates or {});atomic_json(path,value)
        inputs=[{'sha256':file_hash(path),'size':path.stat().st_size}] if bound else []
        problem=SimpleNamespace(manifest={'provenance':{'inputs':inputs}},constraint_profile=None)
        bind_profile(problem,SimpleNamespace(page_policy=POLICY),profile,path)
        return problem

    def test_common_scale_predicted_masks_use_the_same_rigid_page_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.bind(folder)
            self.assertEqual(p.constraint_profile['profile']['relative_piece_scale'],1.)
            self.assertEqual(p.constraint_profile['profile']['mask_mode'],'predicted_segmentation')
            self.assertEqual(p.constraint_profile['profile']['motion_model'],'proper_rigid')
            self.assertEqual(p.capture_metadata['rotation_family'],'continuous_rigid')
            for family in ['continuous','continuous_rigid','quarter_turn']:
                p=self.bind(folder,updates={'rotation_family':family})
                self.assertEqual(p.capture_metadata['rotation_family'],
                    'quarter_turn' if family=='quarter_turn' else 'continuous_rigid')

    def test_exact_generator_profile_remains_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.bind(folder,origin='generator_exact_bypass',profile=EXACT)
            self.assertEqual(p.constraint_profile['profile'],json.loads(EXACT.read_text()))

    def test_mask_provenance_cannot_be_mislabeled(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'mask provenance'):self.bind(folder,profile=EXACT)
            with self.assertRaisesRegex(ValueError,'mask provenance'):self.bind(folder,origin='generator_exact_bypass')

    def test_scale_resize_and_perspective_guards_apply_to_real_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            for change in [{'relative_piece_scale':1.01},{'per_piece_resize':True},
                    {'perspective':True},{'shared_native_pixel_scale':False}]:
                with self.subTest(change=change),self.assertRaises(ValueError):self.bind(folder,updates=change)

    def test_input_profile_still_needs_an_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'not bound'):self.bind(folder,bound=False)


if __name__=='__main__':unittest.main()
