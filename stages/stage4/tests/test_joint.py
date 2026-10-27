import copy
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, slice_arc, validate_active_record, verify_result_support
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.outputs import atomic_json, build_result, digest, output_session, publish, read_result, request_identity
from paper_reconstruction_stage4.refine_joint import corrected_state, refine_state
from paper_reconstruction_stage4.search.common import Budget, SearchResult
from paper_reconstruction_stage4.search.joint_beam import JointSearch, diverse_retain
from paper_reconstruction_stage4.seeds import attach_initial_result
from paper_reconstruction_stage4.solver import solve
from paper_reconstruction_stage4.support import SupportIndex, contact_error, native_arc
from tests.fixtures import build_fixture, translation

ROOT = Path(__file__).resolve().parents[1]


def config(**kwargs):
    values = dict(solver="joint_beam_v3", objective_version="observable-structure.v2", max_seconds=10.,
                  joint_wall_seconds=20., joint_finalize_seconds=2., joint_trials=4, beam_width=8, portfolio_size=2)
    values.update(kwargs)
    return SolverConfig(**values).validate()


class JointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, **kwargs):
        build_fixture(self.root/"input", **kwargs)
        return load_problem(self.root/"input", config())

    def scaled(self, cols=2):
        problem = self.fixture(rows=1, cols=cols)
        for piece in problem.pieces.values():
            piece.contour["points_xy"] *= 4
            piece.contour["arc_length_px"] *= 4
            piece.mask = np.asarray(Image.fromarray(piece.mask).resize((128,128),Image.Resampling.NEAREST),bool)
            path=self.root/(piece.id+".png")
            Image.new("RGB",(128,128),(240,240,240)).save(path); piece.image_path=path
        return problem

    def active_config(self, **kwargs):
        return config(objective_version="observable-joint.v3",support_policy="active-subarc.v1",**kwargs)

    def test_discrete_search_preserves_valid_inventory_and_distinct_backbones(self):
        problem=self.fixture(alternatives=True)
        result=solve(problem,config(joint_trials=8,repair_rounds=1,max_repair_checks=300))
        self.assertEqual(len(result.states[0].components()),1)
        self.assertGreater(result.diagnostics['retained_backbones'],1)
        self.assertTrue(all(r['valid'] for r in result.diagnostics['final_feasibility']))
        self.assertTrue(result.diagnostics['complete_candidate_records'])
        self.assertTrue(all(len(s.poses)==4 for s in result.states))

    def test_joint_search_budget_keeps_physical_inventory(self):
        problem=self.fixture(rows=5,cols=6)
        result=solve(problem,config(max_proposals=10,joint_trials=8))
        self.assertEqual(result.termination,'proposal_budget')
        self.assertLessEqual(result.diagnostics['proposals'],10)
        self.assertEqual(len(result.states[0].poses),30)
        self.assertTrue(result.diagnostics['final_feasibility'][0]['valid'])

    def test_authenticated_control_survives_beam_pruning(self):
        from paper_reconstruction_stage4.model import singleton_state
        p=self.fixture(rows=1,cols=2);c=config(beam_width=1,portfolio_size=1)
        fallback=singleton_state(p);p.initial_states=[fallback]
        support=SupportIndex(p,c);p.structural=JointScorer(p,c,support)
        result=JointSearch(p,c,Feasibility(p,c),support).solve()
        self.assertTrue(any(not s.selected for s in result.states))
        self.assertEqual(result.diagnostics['protected_feasible_control_states'],1)

    def test_joint_path_keeps_reflection_collision_and_pole_guards(self):
        p=self.fixture(rows=1,cols=2);edge=p.edges[0]
        for h in [np.eye(3),np.diag([-1.,1.,1.]),np.array([[1.,0.,0.],[0.,1.,0.],[-1/31.,0.,1.]])]:
            edge.transform=h
            result=solve(p,config(joint_trials=1))
            self.assertFalse(result.states[0].selected)
            self.assertTrue(result.diagnostics['final_feasibility'][0]['valid'])

    def test_three_join_rebuild_can_cross_a_lower_scoring_basin(self):
        c=config(repair_rounds=1,max_repair_checks=100,joint_region_joins=3,joint_rebuild_width=4)
        search=object.__new__(JointSearch); search.config=c; search.trace=[];search.rejections=Counter()
        search.edges={}; search.completed=[]
        original=State({'a':np.eye(3)},{'a':'a'},('bad1','bad2','bad3'),score=1.)
        search.regions=lambda s:[('bad1','bad2','bad3')]
        search.rebuild=lambda kept,budget: State({'a':np.eye(3)},{'a':'a'},score=0.)
        search.ordered_candidates=lambda s,noise=0:[type('Choice',(),{'id':'g'+str(len(s.selected)+1)})()]
        def attempt(s,e):
            ids=(*s.selected,e.id)
            return State(dict(s.poses),dict(s.groups),ids,score=2. if len(ids)==3 else .1)
        search.attempt=attempt
        pool,report=search.rebuild_regions([original],Budget(c))
        self.assertEqual(pool[0].selected,('g1','g2','g3'))
        self.assertEqual(report['maximum_removed_joins'],3)
        self.assertEqual(len(report['trace'][0]['depth_counts']),3)

    def test_joint_score_search_ablation_is_exact_v2_when_no_active_support(self):
        from paper_reconstruction_stage4.structural import StructuralScorer
        p=self.fixture(rows=1,cols=2); c=config(); support=SupportIndex(p,c)
        edge=p.edges[0]; state=State({'piece_00':np.eye(3),'piece_01':translation(27)}, {'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),assignments={edge.id:edge.occurrences[0]['occurrence_id']})
        self.assertEqual(StructuralScorer(p,c,support).evaluate(state),JointScorer(p,c,support).evaluate(state))

    def test_mixed_page_styles_do_not_multiply_a_weak_component_prior(self):
        p=self.fixture(); raw=config(structural_compactness_weight=0.,structural_appearance_weight=0.,structural_boundary_weight=0.,structural_corroboration_weight=0.)
        mean=replace(raw,objective_version='observable-joint.v4')
        poses={'piece_00':np.eye(3),'piece_01':translation(27),'piece_02':translation(0,27),'piece_03':translation(27,27)}
        horizontal=tuple(e.id for e in p.edges if e.a[-1] in ['0','2'] and int(e.b[-1])-int(e.a[-1])==1)
        full_ids=tuple(e.id for e in p.edges)[:3]
        full=State(poses,{pid:'piece_00' for pid in poses},full_ids)
        partial=State({'piece_00':np.eye(3),'piece_01':translation(27),'piece_02':np.eye(3),'piece_03':translation(27)},
                      {'piece_00':'piece_00','piece_01':'piece_00','piece_02':'piece_02','piece_03':'piece_02'},horizontal)
        scores=[]
        for c in [raw,mean]:
            scorer=JointScorer(p,c,SupportIndex(p,c))
            for i,pid in enumerate(sorted(p.pieces)):
                scorer.profiles[pid].update(angle_radians=0. if i<2 else np.deg2rad(30),axis_confidence=1.,line_pitch_native_px=None)
            scores.append((scorer.evaluate(partial)['score'],scorer.evaluate(full)['score']))
        self.assertGreater(scores[0][0],scores[0][1])
        self.assertGreater(scores[1][1],scores[1][0])

    def test_partial_contact_gets_observed_contiguous_active_support(self):
        p=self.scaled(); c=self.active_config(); index=ActiveSupportIndex(p,c); edge=p.edges[0]
        poses={'piece_00':np.eye(3),'piece_01':translation(108,54)}
        raw=SupportIndex(p,c).assign((edge.id,),poses,{edge.id:edge})
        self.assertIsNone(raw[0])
        assigned,reason,nodes=index.assign((edge.id,),poses,{edge.id:edge})
        self.assertIsNotNone(assigned,reason)
        state=State(poses,{'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),(edge.id,),assigned)
        index.bind(state)
        record=state.active_support[edge.id]
        self.assertGreaterEqual(min(record['coverage']),.2)
        self.assertLess(max(record['coverage']),.7)
        self.assertLess(contact_error(index.arcs_for(state,edge.id),poses),c.contact_tolerance_px)
        validate_active_record(record,c,index.arcs[assigned[edge.id]])
        self.assertLess(record['normal_cosine_median'],-.9)

    def test_active_support_rejects_tiny_contacts_and_same_side_normals(self):
        p=self.scaled(); c=self.active_config(); edge=p.edges[0]
        poses={'piece_00':np.eye(3),'piece_01':translation(108,100)}
        self.assertIsNone(ActiveSupportIndex(p,c).assign((edge.id,),poses,{edge.id:edge})[0])
        p.pieces['piece_01'].contour['inward_normal_xy'] *= -1
        poses['piece_01']=translation(108,54)
        self.assertIsNone(ActiveSupportIndex(p,c).assign((edge.id,),poses,{edge.id:edge})[0])

    def test_two_neighbors_can_use_disjoint_active_parts_of_a_broad_contour(self):
        from paper_reconstruction_stage4.model import Edge
        p=self.scaled(cols=3);c=self.active_config()
        p.pieces['piece_00'].contour['points_xy'][:,1] *= 2
        first=p.edges[0]
        occurrence=copy.deepcopy(first.occurrences[0])
        occurrence.update(occurrence_id='second-observed-contact',target_key='piece_02/left',target_piece='piece_02')
        second=Edge('second-geometry','piece_00','piece_02',translation(108,116),.9,[occurrence])
        p.edges=[first,second]
        poses={'piece_00':np.eye(3),'piece_01':translation(108,8),'piece_02':translation(108,116)}
        index=ActiveSupportIndex(p,c);edges={e.id:e for e in p.edges}
        assignments,reason,_=index.assign(tuple(edges),poses,edges)
        self.assertIsNotNone(assignments,reason)
        state=State(poses,{pid:'piece_00' for pid in poses},tuple(edges),tuple(edges),assignments)
        index.bind(state)
        self.assertEqual(len(state.active_support),2)
        from paper_reconstruction_stage4.support import overlaps
        self.assertFalse(any(overlaps(a,b,c) for a in index.arcs_for(state,first.id) for b in index.arcs_for(state,second.id)))

    def test_native_wrapped_support_slice_remains_in_raw_interval(self):
        p=self.scaled(); candidate=next(k for k in p.candidates if k.endswith('/left'))
        raw=native_arc(p,candidate,None,(0.,1.))
        sliced=slice_arc(p,raw,.2,.8)
        self.assertAlmostEqual(sliced.length,raw.length*.6)
        for a,b in sliced.intervals:
            self.assertAlmostEqual(sum(max(0.,min(b,d)-max(a,c)) for c,d in raw.intervals),b-a)

    def test_active_support_serialization_is_sealed_and_tampering_rejected(self):
        p=self.scaled(); c=self.active_config(); index=ActiveSupportIndex(p,c); edge=p.edges[0]
        poses={'piece_00':np.eye(3),'piece_01':translation(108,54)}
        assigned,reason,_=index.assign((edge.id,),poses,{edge.id:edge})
        state=State(poses,{'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),(edge.id,),assigned)
        index.bind(state); p.structural=JointScorer(p,c,index); state.score=p.structural.evaluate(state)['score']
        report=Feasibility(p,c).state(state,final=True); self.assertTrue(report['valid'])
        result=SearchResult([state],{'final_feasibility':[report],'refinement':[{'attempted':False}]},'exhausted_candidates')
        request=request_identity(p,c);value=build_result(p,c,result,request)
        with output_session(self.root/'out',request) as work: publish(self.root/'out',work,value,p,[state])
        sealed=read_result(self.root/'out'); self.assertEqual(sealed['schema_version'],'stage4.assembly-result.v2')
        self.assertTrue(verify_result_support(p,sealed,c)['valid'])
        record=copy.deepcopy(sealed['layouts'][0]['selected_joins'][0]['active_support']);record['arcs'][0]['length']=1.
        with self.assertRaises(ValueError):validate_active_record(record,c)

    def test_centered_joint_correction_preserves_gauge_and_incoming_projective_row(self):
        p=self.fixture(rows=1,cols=2)
        h=translation(27); h[2,0]=.0001
        state=State({'piece_00':np.eye(3),'piece_01':h},{'piece_00':'piece_00','piece_01':'piece_00'})
        moved=corrected_state(state,p,np.array([.5,.3,.2,-.1]),np.array([.1,.1,10.,10.]))
        np.testing.assert_array_equal(moved.poses['piece_00'],np.eye(3))
        np.testing.assert_array_equal(moved.poses['piece_01'][2],h[2])

    def test_full_score_refinement_improves_tree_and_keeps_one_gauge(self):
        p=self.fixture(rows=1,cols=2,drift=4.)
        c=config(refine=True,joint_refinement=True,refine_translation_px=10.,refine_rotation_degrees=1.,refine_log_scale=.01,
                 refine_regularization=.00001,refine_seconds=5.,refine_max_nfev=20,structural_boundary_weight=1.,structural_appearance_weight=0.,structural_compactness_weight=0.)
        result=solve(p,c)
        report=result.diagnostics['refinement'][0]
        self.assertTrue(report['accepted'],report)
        self.assertIn('full declared',report['optimized_terms'])
        self.assertLess(abs(result.states[0].poses['piece_01'][0,2]-27),1.)
        np.testing.assert_array_equal(result.states[0].poses['piece_00'],np.eye(3))

    def test_joint_config_rejects_unversioned_active_support_and_unbounded_regions(self):
        for changes in ({'support_policy':'active-subarc.v1'},{'joint_region_joins':30},{'active_minimum_fraction':0.},{'joint_finalize_seconds':30.}):
            with self.assertRaises(ValueError):config(**changes)


if __name__=='__main__':unittest.main()
