from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform
from tests.test_cut_repair_v21 import upgraded
from tests.test_native_repair_v17 import native_proposals
from paper_reconstruction_stage4.constraints import require_native_rotation,CaptureRotationError
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.native_graph_v21 import NativeFrontier
from paper_reconstruction_stage4.native_graph_v17 import Growth
from paper_reconstruction_stage4.cut_repair_v21 import cut_options,moved_node
from paper_reconstruction_stage4.native_bridge import contour_arc
from paper_reconstruction_stage4.repair_search import Node
from paper_reconstruction_stage4.outputs import serialize_layout
from paper_reconstruction_stage4.archive_rotations import modes
from paper_reconstruction_stage4.native_partial import PartialRepair


def continuous(folder,rows=2,cols=2):
    p,c,_,full=upgraded(folder,rows,cols);pid=sorted(p.pieces)[1];piece=p.pieces[pid]
    angle=.37;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    h,w=piece.mask.shape;corners=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])@r.T
    offset=2.-corners.min(axis=0);shape=np.ceil((corners+offset).max(axis=0)+3).astype(int)[::-1]
    inv=r.T;piece.mask=affine_transform(piece.mask.astype(float),inv[[1,0]][:,[1,0]],
        offset=(-inv@offset)[[1,0]],output_shape=tuple(shape),order=0)>.5
    piece.contour['points_xy']=piece.contour['points_xy']@r.T+offset
    piece.contour['inward_normal_xy']=piece.contour['inward_normal_xy']@r.T
    frame=np.eye(3);frame[:2,:2]=r;frame[:2,2]=offset;full.poses[pid]=full.poses[pid]@np.linalg.inv(frame)
    Image.new('RGB',(piece.mask.shape[1],piece.mask.shape[0]),'white').save(piece.image_path)
    p.capture_metadata['rotation_family']='continuous_rigid'
    ctx=make_page_context(p,c,ActiveSupportIndex(p,c));p.page_context=ctx;p.structural=ctx.scorer()
    return p,c,ctx,full


def contacts(problem,rows,pairs):
    result=[]
    for r in rows:
        if tuple(r['pieces']) not in pairs:continue
        a,b=r['pieces'];result.append({'pieces':[a,b],'arcs':tuple(contour_arc(problem,v) for v in r['arcs']),
            'metrics':r['native_metrics'],'proposal_id':r['proposal_id'],'role':'merge','spread':r['shape_spread_ratio'],
            'source_groups':[[pid] for pid in sorted(problem.pieces)],'merged_groups':[[a],[b]],
            'proposed_transform_b_to_a':r['transform_b_to_a']})
    return result


class CaptureRecoveryTests(unittest.TestCase):
    def test_rotation_policy_rejects_unknown_nonrigid_and_undeclared_rotation(self):
        angle=.37;h=np.array([[np.cos(angle),-np.sin(angle),4.],[np.sin(angle),np.cos(angle),5.],[0.,0.,1.]])
        require_native_rotation(h,{'rotation_family':'continuous_rigid'})
        for binding,reason in [(None,'undeclared_rotation_family'),({'rotation_family':'quarter_turn'},'undeclared_rotation')]:
            with self.assertRaises(CaptureRotationError) as raised:require_native_rotation(h,binding)
            self.assertEqual(raised.exception.reason,reason)
        for bad in [h@np.diag([1.01,1.,1.]),h@np.diag([-1.,1.,1.])]:
            with self.assertRaises(CaptureRotationError) as raised:require_native_rotation(bad,{'rotation_family':'continuous_rigid'})
            self.assertEqual(raised.exception.reason,'nonrigid_transform')

    def test_continuous_graph_completion_replays_selected_cycles_without_rounding(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder);basis=singleton_state(p);physical=Feasibility(p,c)
            ctx.generated=native_proposals(p,c,full);ctx.expanded_generation_done=True
            ctx.run_geometry(time.perf_counter()+10.)
            self.assertIsNotNone(ctx.verified_incumbent)
            state=ctx.verified_incumbent
            self.assertEqual(len(state.components()),1)
            layout=serialize_layout(p,state,0,physical.state(state,final=True),{'attempted':False})
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])
            self.assertTrue(any(not np.allclose(np.asarray(j['native_bridge']['proposed_transform_b_to_a'])[:2,:2],
                np.rint(np.asarray(j['native_bridge']['proposed_transform_b_to_a'])[:2,:2]),atol=1e-7) for j in layout['selected_joins']))

    def test_continuous_seed_replay_cache_and_quarter_negative_control(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder);basis=singleton_state(p);physical=Feasibility(p,c)
            row=next(r for r in native_proposals(p,c,full) if sorted(p.pieces)[1] in r['pieces'])
            engine=NativeFrontier(ctx,basis,physical,lambda *a:None,'capture')
            self.assertIsNotNone(engine.seed(row));self.assertIsNotNone(engine.seed(row))
            self.assertEqual(engine.counts['native_decode_cache_hits'],1)
            p.capture_metadata['rotation_family']='quarter_turn'
            engine=NativeFrontier(ctx,basis,physical,lambda *a:None,'quarter')
            self.assertIsNone(engine.seed(row));self.assertEqual(engine.audit()['native_replay_failure_reasons'],{'undeclared_rotation':1})

    def test_continuous_cut_repair_preserves_unplaced_components(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder);ids=sorted(full.poses);rows=native_proposals(p,c,full)
            scope=ids[:3];partial=deepcopy(full);partial.groups={pid:(ids[0] if pid in scope else pid) for pid in ids}
            partial.poses[ids[3]]=np.eye(3);partial.poses[ids[1]][0,2]+=6.
            node=Node(partial,contacts(p,rows,{(ids[0],ids[1]),(ids[0],ids[2])}))
            options=cut_options(node,[ids[1]],[r for r in rows if all(pid in scope for pid in r['pieces'])],
                c.cycle_tolerance_px,'continuous_rigid')
            option=next(r for r in options if tuple(r['proposal']['pieces'])==(ids[0],ids[1]))
            moved=moved_node(ctx,singleton_state(p),node,[ids[1]],option,scope)
            self.assertTrue(np.allclose(moved.state.poses[ids[1]],full.poses[ids[1]]))
            np.testing.assert_array_equal(moved.state.poses[ids[3]],np.eye(3))
            self.assertEqual(moved.state.groups,partial.groups)

    def test_cold_search_runs_before_native_recovery(self):
        from paper_reconstruction_stage4.solver import solve
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder)
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate',return_value=([],{'native_checks':0})), \
                    patch('paper_reconstruction_stage4.archive_rotations.modes',return_value=({},{})):
                result=solve(p,c)
            self.assertTrue(all(r['executed'] for r in result.diagnostics['cold_portfolio']['arms']))
            self.assertTrue(result.diagnostics['native_repair']['native_search_deferred_for_cold_routes'])

    def test_archive_angle_preparation_cannot_be_starved_by_native_search(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder);order=[]
            def generated(*args,**kwargs):order.append('working');return [],{'native_checks':0}
            def archived(*args,**kwargs):order.append('archive');return {},{'additional_angle_modes':0}
            def searched(*args,**kwargs):order.append('search');return None
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate',side_effect=generated), \
                    patch('paper_reconstruction_stage4.archive_rotations.modes',side_effect=archived), \
                    patch.object(ctx,'run_geometry',side_effect=searched):
                ctx.augment_finalists([singleton_state(p)],time.perf_counter()+5.)
            self.assertEqual(order,['working','archive','search'])
            self.assertTrue(ctx.expanded_generation_done)

    def test_stalled_partial_repair_reopens_growth_without_moving_the_free_piece(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder);ids=sorted(p.pieces);scope=ids[:3]
            rows=native_proposals(p,c,full);state=deepcopy(full)
            state.groups={pid:(ids[0] if pid in scope else pid) for pid in ids}
            state.poses[ids[3]]=np.eye(3);state.poses[ids[1]][0,2]+=40.
            growth=Growth(Node(state,contacts(p,rows,{(ids[0],ids[1]),(ids[0],ids[2])})),ids[0],'fixture')
            ctx.generated=rows;ctx._update_frontiers();job=PartialRepair(growth);repaired=[]
            for _ in range(30):
                node,worked=job.step(ctx,ctx.native_physical)
                if node is not None:repaired.append(node)
                if job.exhausted:break
            self.assertTrue(repaired)
            node=next(n for n in repaired if np.allclose(n.state.poses[ids[1]],full.poses[ids[1]]))
            np.testing.assert_array_equal(node.state.poses[ids[3]],np.eye(3))
            engine=ctx.frontiers['all_modes_fallback'];before=engine.seed_cursor
            self.assertTrue(engine.inject(Growth(node,ids[0],'repaired')))
            self.assertEqual(engine.seed_cursor,before)
            self.assertGreater(len(engine.sessions),0)

    def test_native_partial_is_verified_and_free_pieces_cannot_supply_fake_cycles(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder,2,3);ids=sorted(p.pieces);scope=[ids[i] for i in [0,1,3,4]]
            rows=native_proposals(p,c,full);state=deepcopy(full)
            state.groups={pid:(min(scope) if pid in scope else pid) for pid in ids}
            for pid in ids:
                if pid not in scope:state.poses[pid]=np.eye(3)
            tree={(ids[0],ids[1]),(ids[0],ids[3]),(ids[1],ids[4])}
            growth=Growth(Node(state,contacts(p,rows,tree)),ids[0],'fixture')
            ctx.native_basis=singleton_state(p);ctx.native_physical=Feasibility(p,c)
            ctx.repair_deadline=time.perf_counter()+10.;ctx.observe_growth(growth)
            for _ in range(12):
                if not ctx.partial_jobs:break
                ctx.extra_task_step('capture_partial_support')
            self.assertTrue(ctx.partial_candidates)
            saved=next(iter(ctx.partial_candidates.values()))
            self.assertEqual(sorted(map(len,saved.components()),reverse=True),[4,1,1])
            layout=serialize_layout(p,saved,0,ctx.native_physical.state(saved,final=True),{'attempted':False})
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])
            self.assertTrue(all(j['piece_a'] in scope and j['piece_b'] in scope for j in layout['selected_joins']))

    def test_targeted_archive_angles_are_validated_and_translation_is_not_imported(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder);a,b=sorted(p.pieces)[:2]
            angle=.71;h=np.array([[np.cos(angle),-np.sin(angle),99999.],[np.sin(angle),np.cos(angle),-99999.],[0.,0.,1.]])
            row={'piece_a':a,'piece_b':b,'transform_b_to_a':h.tolist(),'hypothesis_id':'archived'}
            bundle=SimpleNamespace(manifest=p.manifest,hypotheses=lambda:iter([row]))
            with patch('paper_reconstruction_stage4.archive_rotations.load_bundle',return_value=bundle):
                recovered,audit=modes(p,{(a,b)},time.perf_counter()+5.)
                empty,expired=modes(p,{(a,b)},time.perf_counter()-1.)
            self.assertEqual(recovered[a,b][0]['source_geometry_id'],'archived')
            self.assertNotIn('transform_b_to_a',recovered[a,b][0])
            self.assertTrue(audit['archive_validation_exhausted']);self.assertFalse(audit['translation_inherited'])
            self.assertFalse(empty);self.assertTrue(expired['budget_abstained'])


if __name__=='__main__':unittest.main()
