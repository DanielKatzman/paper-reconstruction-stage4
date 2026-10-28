from copy import deepcopy
import tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_stage4.observed_border_angles import lines,angle_seeds
from paper_reconstruction_stage4.capture_quality_repair import compact_area,quality_met,node_from_verified,best_cut,polish
from paper_reconstruction_stage4.observed_rigid_refinement import proposals as rigid_proposals
from paper_reconstruction_stage4.page import enclosing_rectangle
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.native_bridge import pair_options
from paper_reconstruction_stage4.geometry import relative
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.repair_search import Node
from paper_reconstruction_stage4.active_support import verify_result_support
from paper_reconstruction_stage4.outputs import serialize_layout
from paper_reconstruction_stage4.cut_repair_v21 import cut_options
from tests.test_capture_recovery import continuous,contacts
from tests.test_native_repair_v17 import native_proposals


def rectangle():
    return np.vstack([np.column_stack([np.arange(800),np.zeros(800)]),
        np.column_stack([np.full(500,800.),np.arange(500)]),
        np.column_stack([np.arange(800,0,-1),np.full(800,500.)]),
        np.column_stack([np.zeros(500),np.arange(500,0,-1)])])


def verified(folder):
    p,c,ctx,full=continuous(folder);rows=native_proposals(p,c,full)
    ids=sorted(p.pieces);tree={(ids[0],ids[1]),(ids[0],ids[2]),(ids[1],ids[3])}
    node=Node(full,contacts(p,rows,tree));ctx.repair_deadline=time.perf_counter()+10.
    candidate,verdict=ctx.finalize(singleton_state(p),node,Feasibility(p,c))
    if candidate is None:raise AssertionError(verdict)
    return p,c,ctx,candidate,rows


class CaptureQualityRepairTests(unittest.TestCase):
    def test_border_rotations_recover_missing_nonquarter_angle_without_pose_or_reference(self):
        xy=rectangle();theta=.213;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        first=lines(xy);second=lines(xy@r.T+[31.,-78.])
        seeds=angle_seeds(first,second)
        self.assertEqual(len(first),4)
        self.assertTrue(any(abs(np.arctan2(np.sin(s['angle']+theta),np.cos(s['angle']+theta)))<1e-6 for s in seeds))
        self.assertTrue(all(s['source_geometry_id'] is None and 'transform_b_to_a' not in s for s in seeds))
        self.assertTrue(all(s['observed_border_seed']['first_span']['line_rms_native_px']<1e-6 for s in seeds))

    def test_no_long_border_and_invalid_contours_cannot_create_angle_evidence(self):
        theta=np.linspace(0,2*np.pi,1000,endpoint=False)
        circle=300*np.column_stack([np.cos(theta),np.sin(theta)])
        self.assertFalse(lines(circle));self.assertFalse(angle_seeds([],lines(rectangle())))
        with self.assertRaises(ValueError):lines(np.full((5,2),np.nan))

    def test_vectorized_compactness_matches_original_pixel_centre_rectangle(self):
        rng=np.random.default_rng(1729)
        for count in [4,12,80]:
            for _ in range(5):
                cloud=rng.normal(size=(count,2))*[100.,500.]+rng.uniform(-1000,1000,2)
                self.assertAlmostEqual(compact_area(cloud),enclosing_rectangle(cloud)[0],places=6)

    def test_geometry_stop_uses_raster_uncertainty_and_does_not_accept_bad_shape(self):
        shape={'shape_gate_pass':True,'raster_fraction_uncertainty_estimate':.006,
            'unfilled_fraction':.004,'overlap_fraction':.001,'interior_hole_fraction':.001}
        self.assertTrue(quality_met(shape))
        self.assertFalse(quality_met({**shape,'unfilled_fraction':.03}))
        self.assertFalse(quality_met({**shape,'shape_gate_pass':False}))

    def test_verified_tree_rebuild_and_rigid_proposals_keep_scale_and_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state,rows=verified(folder);node=node_from_verified(ctx,state)
            self.assertFalse(node.state.selected);self.assertEqual(len(node.contacts),len(p.pieces)-1)
            proposal=next(rigid_proposals(ctx,node,time.perf_counter()+5.))
            poses,audit=proposal
            np.testing.assert_array_equal(poses[audit['anchor']],state.poses[audit['anchor']])
            for h in poses.values():
                np.testing.assert_allclose(h[:2,:2].T@h[:2,:2],np.eye(2),atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(h[:2,:2]),1.,places=12)
            self.assertFalse(audit['reference_used'])

    def test_supported_complete_tree_cut_can_improve_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state,rows=verified(folder);node=node_from_verified(ctx,state);pid=max(p.pieces)
            node.state.poses[pid]=node.state.poses[pid].copy();node.state.poses[pid][0,2]+=7.
            for contact in node.contacts:
                a,b=contact['pieces'];arcs,metrics=pair_options(p,a,b,node.state.poses,c,[])[0]
                contact.update(arcs=arcs,metrics=metrics,proposed_transform_b_to_a=relative(node.state.poses,a,b).tolist())
            audit={'cut_options_ranked':0,'fine_geometry_checks':0,'support_replays':0,'accepted_moves':[]}
            result=best_cut(ctx,node,rows,Feasibility(p,c),time.perf_counter()+10.,audit)
            self.assertIsNotNone(result)
            _,candidate,verdict=result
            self.assertEqual(verdict['disposition'],'verified_completion')
            layout=serialize_layout(p,candidate,0,{'valid':True},{'attempted':False})
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])

    def test_deadline_retains_original_verified_incumbent_and_input_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state,rows=verified(folder);before=deepcopy(state.poses)
            with patch('paper_reconstruction_stage4.capture_quality_repair.quality_met',return_value=False):
                answer=polish(ctx,state,time.perf_counter()-1.)
            self.assertIs(answer,state)
            for pid in before:np.testing.assert_array_equal(state.poses[pid],before[pid])
            audit=ctx.repair_audit['supported_capture_quality_repair'][-1]
            self.assertTrue(audit['deadline_reached']);self.assertTrue(audit['verified_incumbent_preserved'])

    def test_geometry_ranked_cut_path_preserves_every_motion_without_quadratic_corroboration(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state,rows=verified(folder);node=node_from_verified(ctx,state);moving=[max(p.pieces)]
            a=cut_options(node,moving,rows,c.cycle_tolerance_px,'continuous_rigid')
            b=cut_options(node,moving,rows,c.cycle_tolerance_px,'continuous_rigid',corroboration=False)
            self.assertEqual({r['proposal']['proposal_id'] for r in a},{r['proposal']['proposal_id'] for r in b})
            self.assertTrue(all(r['distinct_neighbor_witnesses']==0 for r in b))

    def test_supported_checkpoint_yields_to_recovery_only_once_per_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state,_=verified(folder);ctx.verified_incumbent=state
            with patch('paper_reconstruction_stage4.native_repair_v21.RepairContext._complete',return_value=None):
                self.assertIs(ctx._complete(None,{}),state)
                ctx.capture_quality_processed.add(__import__('paper_reconstruction_stage4.page_geometry',fromlist=['fingerprint']).fingerprint(state))
                self.assertIsNone(ctx._complete(None,{}))

    def test_locally_exhausted_high_quality_page_does_not_restart_unrelated_growth(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state,_=verified(folder)
            ctx.verified_incumbent=state
            shape={**ctx.verified_verdict['geometry'],'score':.995,'unfilled_fraction':.007}
            shape['raster_fraction_uncertainty_estimate']=.006
            ctx.verified_verdict={'geometry':shape,'disposition':'verified_completion'}
            ctx.repair_audit['supported_capture_quality_repair']=[{'local_neighborhood_exhausted':True}]
            with patch('paper_reconstruction_stage4.native_repair_v21.RepairContext.run_geometry',return_value=state) as search, \
                    patch.object(ctx,'polish_supported',return_value=state):
                self.assertIs(ctx.run_geometry(time.perf_counter()+5.),state)
            self.assertEqual(search.call_count,1)


if __name__=='__main__':unittest.main()
