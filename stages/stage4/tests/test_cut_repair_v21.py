from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_native_repair_v18 import fixture
from tests.test_native_repair_v17 import native_proposals
from paper_reconstruction_stage4.config import SolverConfig, STRAIGHT_DEFERRED_GROWTH
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, verify_result_support
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.native_graph_v21 import NativeFrontier
from paper_reconstruction_stage4.cut_repair_v21 import cut_options, moved_node, smaller_side
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.repair_search import Node
from paper_reconstruction_stage4.outputs import serialize_layout, digest


def upgraded(folder,rows=2,cols=2):
    p,c,_,full=fixture(folder,rows,cols)
    c=replace(c,native_growth_policy=STRAIGHT_DEFERRED_GROWTH,native_cut_repair=True).validate()
    ctx=make_page_context(p,c,ActiveSupportIndex(p,c)); p.page_context=ctx; p.structural=ctx.scorer()
    return p,c,ctx,full


class CutRepairTests(unittest.TestCase):
    def test_policy_is_explicit_and_old_normalized_configs_are_unchanged(self):
        old=json.loads(Path('configs/joint_v19_geometry_expansion_development.json').read_text())
        self.assertEqual(SolverConfig.from_dict(old).to_dict(),old)
        new={**old,'native_cut_repair':True}
        self.assertEqual(SolverConfig.from_dict(new).to_dict(),new)
        with self.assertRaises(ValueError):
            replace(SolverConfig(),native_cut_repair=True).validate()

    def test_frontier_advances_without_reseeding_after_a_pause(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder); basis=singleton_state(p)
            engine=NativeFrontier(ctx,basis,Feasibility(p,c),lambda *args:None,'all')
            rows=native_proposals(p,c,full); engine.add(rows)
            for _ in range(3):engine.step('beam')
            generator=engine.fast; before=engine.counts['merge_checks']
            engine.step('dfs'); engine.step('beam')
            self.assertIs(engine.fast,generator)
            self.assertGreater(engine.counts['merge_checks'],before)
            self.assertGreater(engine.counts['backtracking_seed_sessions_started'],0)
            count=len(engine.seed_order); self.assertFalse(engine.add(rows))
            self.assertEqual(len(engine.seed_order),count)

    def test_new_modes_reopen_exhausted_frontier_and_can_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder); basis=singleton_state(p); physical=Feasibility(p,c)
            correct=native_proposals(p,c,full); bad=deepcopy(correct[0])
            bad['transform_b_to_a'][0][2]+=1000.;bad['proposal_id']=digest(bad)
            ctx.repair_deadline=time.perf_counter()+10.
            engine=NativeFrontier(ctx,basis,physical,lambda node,shape:ctx.finalize(basis,node,physical)[0],'all')
            engine.add([bad])
            for _ in range(100):
                engine.step('beam');engine.step('dfs')
                if engine.fast_done and engine.dfs_done:break
            self.assertTrue(engine.fast_done and engine.dfs_done)
            seeded=engine.counts['backtracking_seed_sessions_started']
            engine.add(correct);self.assertEqual(engine.counts['backtracking_seed_sessions_started'],seeded)
            for _ in range(100):
                engine.step('beam');engine.step('dfs')
                if engine.answer is not None:break
            self.assertIsNotNone(engine.answer)
            layout=serialize_layout(p,engine.answer,0,physical.state(engine.answer,final=True),{'attempted':False})
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])

    def test_scheduler_starts_dfs_without_waiting_for_failed_beams(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder);ctx.generated=native_proposals(p,c,full)
            ctx.expanded_generation_done=True
            with patch.object(ctx,'_complete',return_value=None):
                result=ctx.run_geometry(time.perf_counter()+.6)
            self.assertIsNone(result)
            audit=ctx.repair_audit['merge_searches'][-1]
            self.assertGreater(audit.get('backtracking_seed_sessions_started',0),0)
            self.assertFalse(audit['frontiers_restarted'])
            frames={k:id(v) for k,v in ctx.frontiers.items()}
            ctx.run_geometry(time.perf_counter()+.1)
            self.assertEqual({k:id(v) for k,v in ctx.frontiers.items()},frames)

    def test_empty_or_expired_search_does_not_claim_impossibility(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder);ctx.generated=native_proposals(p,c,full)
            self.assertIsNone(ctx.run_geometry(time.perf_counter()-1.))
            audit=ctx.repair_audit['merge_searches'][-1]
            self.assertEqual(audit['stop_reason'],'overall_deadline')
            self.assertFalse(audit['proposal_graph_exhausted'])

    def test_group_repair_preserves_both_sides_and_replays_bridge(self):
        from paper_reconstruction_stage4.native_bridge import contour_arc
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder);rows=native_proposals(p,c,full)
            # A spanning tree of the 2x2 grid with a two-piece right side.
            ids=sorted(full.poses); tree_pairs={(ids[0],ids[1]),(ids[0],ids[2]),(ids[1],ids[3])}
            contacts=[]
            for r in rows:
                if tuple(r['pieces']) in tree_pairs:
                    a,b=r['pieces'];contacts.append({'pieces':[a,b],'arcs':tuple(contour_arc(p,x) for x in r['arcs']),
                        'proposal_id':r['proposal_id'],'role':'merge','source_groups':[[a],[b]],
                        'merged_groups':[[a],[b]],'proposed_transform_b_to_a':r['transform_b_to_a']})
            moving=(ids[1],ids[3]); shifted=deepcopy(full)
            for pid in moving:shifted.poses[pid][0,2]+=6.
            node=Node(shifted,contacts)
            options=cut_options(node,moving,rows,c.cycle_tolerance_px)
            option=next(r for r in options if tuple(r['proposal']['pieces'])==(ids[0],ids[1]))
            altered=moved_node(ctx,singleton_state(p),node,moving,option)
            for pid in ids:self.assertTrue(np.allclose(altered.state.poses[pid],full.poses[pid]))
            self.assertTrue(np.allclose(shifted.poses[moving[0]][0,2],full.poses[moving[0]][0,2]+6.))
            self.assertEqual(len(altered.contacts),3)
            self.assertEqual(smaller_side({'cut':list(moving),'other_side':[ids[0],ids[2]]}),(ids[0],ids[2]))

    def test_local_ordering_does_not_remove_wider_or_rotated_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder,1,2); rows=native_proposals(p,c,full); node=Node(full)
            for shift in (5.,500.):
                row=deepcopy(rows[0]);row['transform_b_to_a'][0][2]+=shift;row['proposal_id']=digest(row);rows.append(row)
            options=cut_options(node,[max(full.poses)],rows,c.cycle_tolerance_px)
            self.assertTrue(any(r['local_ordering_tier'] for r in options))
            self.assertTrue(any(not r['local_ordering_tier'] for r in options))

    def test_opt_in_solver_still_publishes_independently_verified_straight_grid(self):
        from paper_reconstruction_stage4.solver import solve
        from paper_reconstruction_stage4.outputs import build_result,validate_result,request_identity
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder);rows=native_proposals(p,c,full)
            with patch('paper_reconstruction_stage4.seam_poses_v18.generate',return_value=(rows,{'native_checks':0})):
                result=solve(p,c)
            self.assertEqual(result.termination,'verified_construction')
            published=build_result(p,c,result,request_identity(p,c));validate_result(published)
            self.assertTrue(verify_result_support(p,published,c)['valid'])

    def test_rectangle_alone_cannot_accept_a_sliding_uncorroborated_bridge(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder,1,2);basis=singleton_state(p);ph=Feasibility(p,c)
            ctx.repair_deadline=time.perf_counter()+5.;verdicts=[]
            def complete(node,shape):
                candidate,verdict=ctx.finalize(basis,node,ph)
                verdicts.append(verdict);return candidate
            engine=NativeFrontier(ctx,basis,ph,complete,'all');engine.add(native_proposals(p,c,full))
            for _ in range(100):
                engine.step('beam');engine.step('dfs')
                if engine.fast_done and engine.dfs_done:break
            self.assertIsNone(engine.answer)
            self.assertTrue(verdicts[0]['geometry']['shape_gate_pass'])
            self.assertIn('selected-cycle',verdicts[0]['reason'])

    def test_existing_verified_incumbent_survives_a_later_expired_search(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder)
            ctx.verified_incumbent=full
            ctx.verified_verdict={'geometry':{'score':.97,'shape_gate_pass':True},'disposition':'verified_completion'}
            candidate=ctx.run_geometry(time.perf_counter()-1.)
            self.assertIs(candidate,full)
            self.assertTrue(ctx.repair_audit['verified_incumbent']['preserved_after_deadline'])
            self.assertFalse(ctx.repair_audit['verified_incumbent']['existing_geometry_stop_target_met'])

    def test_marginal_verified_incumbent_does_not_end_at_the_beam_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder);basis=singleton_state(p)
            engine=NativeFrontier(ctx,basis,Feasibility(p,c),lambda *args:None,'test')
            ctx.frontiers={'test':engine};ctx.expanded_generation_done=True
            ctx.verified_incumbent=full
            ctx.verified_verdict={'geometry':{'score':.97,'shape_gate_pass':True},'disposition':'verified_completion'}
            progressed=[]
            def later_work():
                progressed.append(True);yield
                ctx.verified_verdict={'geometry':{'score':.99,'shape_gate_pass':True},'disposition':'verified_completion'}
                engine.answer=full;yield
            engine.dfs=later_work()
            ctx.run_geometry(time.perf_counter()+2.,fast_only=True)
            self.assertTrue(progressed)
            self.assertTrue(ctx.repair_audit['verified_incumbent']['existing_geometry_stop_target_met'])

    def test_expensive_support_turn_is_charged_time_so_growth_is_not_starved(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder);clock=[0.];support_calls=[]
            class Frontier:
                fast_done=False;dfs_done=True;answer=None;deepest=1
                checks=0
                def step(self,kind):
                    clock[0]+=.1;self.checks+=1;return True
                def audit(self):
                    return {'merge_checks':self.checks,'rejections':{},'elapsed_seconds':self.checks*.1,
                        'observed_growth_states':[]}
            engine=Frontier();ctx.frontiers={'nonstraight_first':engine}
            ctx.expanded_generation_done=True;ctx.support_jobs['costly']={'exhausted':False}
            def expensive():
                support_calls.append(True);clock[0]+=3.;return None,True
            with patch('paper_reconstruction_stage4.native_repair_v21.time.perf_counter',side_effect=lambda:clock[0]), \
                    patch.object(ctx,'_support_turn',side_effect=expensive):
                ctx.run_geometry(20.)
            self.assertGreater(engine.checks,100)
            self.assertLessEqual(len(support_calls),2)
            self.assertGreaterEqual(ctx.repair_audit['scheduler_work']['seconds_by_task']['support'],3.)


if __name__=='__main__':unittest.main()
