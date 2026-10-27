import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_stage1_for_stage2 import connected_mask,prepare
from common import inventory


class MaskBoundaryTests(unittest.TestCase):
    def test_single_four_connected_component_is_unchanged(self):
        mask=np.ones((12,12),bool);result,removed=connected_mask(mask)
        np.testing.assert_array_equal(result,mask);self.assertEqual(removed,0)

    def test_tiny_diagonal_fringe_is_bounded(self):
        mask=np.zeros((202,202),bool);mask[1:201,1:201]=True;mask[0,0]=True
        result,removed=connected_mask(mask)
        self.assertEqual(removed,1);self.assertFalse(result[0,0])

    def test_meaningful_diagonal_fragment_requires_review(self):
        mask=np.zeros((31,31),bool);mask[1:16,1:16]=True;mask[16:30,16:30]=True
        with self.assertRaisesRegex(ValueError,'meaningful fragment'):connected_mask(mask)

    def test_adapter_preserves_raw_files_and_native_crop_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'raw';source.mkdir();output=root/'adapted'
            mask=np.zeros((205,205),np.uint8);mask[2:202,2:202]=255;mask[1,1]=255
            Image.fromarray(mask).save(source/'full.png')
            crop=mask[1:202,1:202];Image.fromarray(crop).save(source/'crop.png')
            rgba=np.zeros((*crop.shape,4),np.uint8);rgba[:,:,:3]=123;rgba[:,:,3]=crop
            Image.fromarray(rgba).save(source/'piece.png')
            row={'accepted':True,'capture_id':'sample','bbox_xywh':[1,1,201,201],
                'mask_path':'full.png','piece_mask_path':'crop.png','piece_path':'piece.png'}
            (source/'segmentation_manifest.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            before=inventory(source);prepare(source,output)
            self.assertEqual(inventory(source),before)
            adapted=json.loads((output/'segmentation_manifest.jsonl').read_text())
            self.assertEqual(adapted['bbox_xywh'],[0,0,205,205])
            alpha=np.asarray(Image.open(output/'piece.png'))[:,:,3]
            np.testing.assert_array_equal(alpha,np.asarray(Image.open(output/'crop.png')))
            self.assertFalse(alpha[0].any());self.assertFalse(alpha[-1].any())
            self.assertTrue((output/'mask_boundary_adapter.json').exists())


if __name__=='__main__':unittest.main()
