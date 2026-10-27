from types import SimpleNamespace
import unittest
import numpy as np
from evaluation.pose_coverage import full_pose_error


class PhysicalPoseErrorTests(unittest.TestCase):
    def test_exact_seam_can_hide_wrong_full_piece_scale(self):
        reference = np.array([[1.,0.,27.],[0.,1.,0.],[0.,0.,1.]])
        stretched = np.array([[2.,0.,25.],[0.,1.,0.],[0.,0.,1.]])
        seam = np.array([[2.,y] for y in range(2,30)],float)
        from paper_reconstruction_contracts import map_points
        np.testing.assert_array_equal(map_points(seam,reference),map_points(seam,stretched))
        pixels = np.array([[x,y] for y in range(3,29) for x in range(3,29)],float)
        maps = {"a":np.eye(3),"b":reference}
        shapes = {"a":(32,32),"b":(32,32)}
        correct = full_pose_error(SimpleNamespace(a="a",b="b",transform=reference),{"a":pixels,"b":pixels},maps,shapes)
        wrong = full_pose_error(SimpleNamespace(a="a",b="b",transform=stretched),{"a":pixels,"b":pixels},maps,shapes)
        self.assertEqual(correct["worst_fraction_native_crop_diagonal"],0.)
        self.assertGreater(wrong["worst_fraction_native_crop_diagonal"],.5)
