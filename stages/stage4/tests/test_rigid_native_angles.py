from copy import deepcopy
from pathlib import Path
import tempfile,time
import unittest
import numpy as np
from scipy.ndimage import affine_transform
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.model import Edge
from paper_reconstruction_stage4.constraints import require_rigid
from paper_reconstruction_stage4.seam_poses_v18 import generate
from paper_reconstruction_stage4.native_bridge_v2 import make_record,validate_record
from paper_reconstruction_stage4.outputs import digest
from tests.test_native_repair_v14 import curved_pair


class RigidNativeAngles(unittest.TestCase):
    def test_observed_angle_is_used_but_wrong_translation_is_not_inherited(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,expected=curved_pair(folder);piece=p.pieces['beta']
            theta=.37;co,si=np.cos(theta),np.sin(theta);r=np.array([[co,-si],[si,co]])
            corners=np.array([[0,0],[piece.mask.shape[1]-1,0],[0,piece.mask.shape[0]-1],[piece.mask.shape[1]-1,piece.mask.shape[0]-1]])@r.T
            offset=2.-corners.min(axis=0);shape=np.ceil((corners+offset).max(axis=0)+3).astype(int)[::-1]
            inv=r.T;matrix=inv[[1,0]][:,[1,0]];shift=(-inv@offset)[[1,0]]
            piece.mask=affine_transform(piece.mask.astype(float),matrix,offset=shift,output_shape=tuple(shape),order=0)>0.5
            piece.contour['points_xy']=piece.contour['points_xy']@r.T+offset
            piece.contour['inward_normal_xy']=piece.contour['inward_normal_xy']@r.T
            frame=np.eye(3);frame[:2,:2]=r;frame[:2,2]=offset
            expected=expected@np.linalg.inv(frame);wrong=expected.copy();wrong[:2,2]+=[5000.,-6000.]
            p.edges=[Edge('observed-angle','alpha','beta',wrong,1.,[])]
            p.capture_metadata['rotation_family']='continuous_rigid'
            proposals,audit=generate(p,c,time.perf_counter()+25.,maximum_checks=64,samples=128,peaks=2)
            self.assertTrue(proposals);self.assertEqual(audit['relative_piece_scale'],1.)
            self.assertFalse(audit['pose_translation_inherited'])
            probes=np.array([[0.,0.],[50.,70.],[100.,100.]])
            errors=[float(np.max(np.linalg.norm(map_points(probes,np.asarray(row['transform_b_to_a']))-map_points(probes,expected),axis=1))) for row in proposals]
            self.assertLess(min(errors),5.)
            self.assertTrue(any(row['angle_seed_geometry_id']=='observed-angle' for row in proposals))
            for row in proposals:require_rigid(row['transform_b_to_a'])

    def record(self,family):
        theta=.37;c,s=np.cos(theta),np.sin(theta);h=[[c,-s,12.],[s,c,23.],[0.,0.,1.]]
        value={'policy':'stage4.native-component-bridge.v2','input_manifest_sha256':'a'*64,
            'pieces':['a','b'],'arcs':[{'piece':'a'},{'piece':'b'}],'transform_b_to_a':h,
            'source_candidate_id':'source','placement_proposal_id':'proposal','source_groups':[['a'],['b']],
            'merged_groups':[['a'],['b']],'role':'merge','proposed_transform_b_to_a':h,
            'capture_binding':{'rotation_family':family,'capture_profile_sha256':'b'*64},
            'producer':'stage4_observed_native_contours','upstream_pose_or_occurrence_claimed':False}
        return {**value,'bridge_id':digest(value)}

    def test_continuous_bridge_does_not_claim_quarter_turns(self):
        validate_record(self.record('continuous_rigid'))
        with self.assertRaisesRegex(ValueError,'undeclared rotation'):validate_record(self.record('quarter_turn'))

    def test_scaled_bridge_is_still_rejected(self):
        value=self.record('continuous_rigid');value['proposed_transform_b_to_a'][0][0]*=1.01
        value['bridge_id']=digest({k:v for k,v in value.items() if k!='bridge_id'})
        with self.assertRaises(ValueError):validate_record(value)

    def test_quarter_turn_generator_keeps_original_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_=curved_pair(folder)
            rows,audit=generate(p,c,time.perf_counter()+15.,maximum_checks=8,samples=128,peaks=2)
            self.assertEqual(audit['policy'],'native-contour-quarter-votes-geometry.v3')
            self.assertTrue(all('quarter_turn' in row for row in rows))


if __name__=='__main__':unittest.main()
