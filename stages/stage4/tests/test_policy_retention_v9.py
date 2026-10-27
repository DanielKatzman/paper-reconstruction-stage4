from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from benchmark.common import read_json
from paper_reconstruction_stage4.config import SolverConfig,RETAINED_POLICY,RETENTION_DEFAULTS
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import State,singleton_state
from paper_reconstruction_stage4.support import SupportIndex
from paper_reconstruction_stage4.search.incumbents import Incumbents
from paper_reconstruction_stage4.search.joint_beam import JointSearch
from paper_reconstruction_stage4.search.retained_portfolio import source_config
from paper_reconstruction_stage4.solver import solve
from tests.fixtures import build_fixture,translation

ROOT=Path(__file__).resolve().parents[1]


class PolicyRetentionTests(unittest.TestCase):
    def config(self,checked=False,**kw):
        value=dict(read_json(ROOT/'configs/joint_v9_retained_development.json'))
        value.update(max_proposals=400,max_seconds=15.,joint_wall_seconds=25.,joint_finalize_seconds=2.,
            joint_trials=4,beam_width=8,portfolio_size=3,joint_retention_route_proposals=150)
        if checked:value['joint_finalist_policy']='checked-incumbents.v1'
        value.update(kw)
        return SolverConfig.from_dict(value)

    def fixture(self,folder,checked=False,**kw):
        path=Path(folder)/'input';build_fixture(path,rows=1,cols=2)
        config=self.config(checked,**kw);problem=load_problem(path,config)
        support=SupportIndex(problem,config);problem.structural=JointScorer(problem,config,support)
        search=JointSearch(problem,config,Feasibility(problem,config),support)
        edge=problem.edges[0]
        state=State({'piece_00':np.eye(3),'piece_01':translation(27)},
            {'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),(edge.id,),
            {edge.id:edge.occurrences[0]['occurrence_id']},score=42.)
        bank=Incumbents(search,time.perf_counter()+10.);route=bank.route('test',time.perf_counter()+10.)
        return problem,config,search,state,bank,route

    def test_historical_normalizations_do_not_gain_retention_fields(self):
        for name in ('stage4_speed_reuse_v1','stage4_v8_support_aware_development_r2'):
            for path in (ROOT/'controls'/name/'configs').glob('*.json'):
                value=read_json(path)
                self.assertEqual(SolverConfig.from_dict(value).to_dict(),value)
                self.assertFalse(set(value)&set(RETENTION_DEFAULTS))
        with self.assertRaises(ValueError):SolverConfig(joint_finalist_policy='checked-incumbents.v1').validate()
        with self.assertRaises(ValueError):self.config(joint_final_check_limit=10000)

    def test_route_objectives_and_orderings_are_independent(self):
        c=self.config(checked=True)
        expected=[('v4_constructions','observable-joint.v4','diverse-constructions.v1'),
            ('v6_component','observable-joint.v6-attachment','component-beam.v1'),
            ('v8_component','observable-joint.v8-continuation','component-beam.v3-lookahead')]
        for name,objective,policy in expected:
            value=source_config(c,name,100,3.)
            self.assertEqual(value.objective_version,objective)
            self.assertEqual(value.joint_search_policy,policy)
            self.assertEqual(value.joint_finalist_policy,'late.v1')
            self.assertEqual(value.max_proposals,100)
            self.assertEqual(value.contact_tolerance_px,c.contact_tolerance_px)

    def test_invalid_higher_score_cannot_evict_checked_incumbent(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,good,bank,route=self.fixture(folder)
            self.assertTrue(route.consider(good,'good')['valid'])
            bad=replace(good,poses={**good.poses,'piece_01':translation(20)},score=100.)
            self.assertFalse(route.consider(bad,'bad')['valid'])
            self.assertEqual(len(route.states),1)
            np.testing.assert_allclose(route.states[0].poses['piece_01'],translation(27))
            self.assertEqual(good.score,42.)

    def test_final_check_cache_uses_exact_poses_and_does_not_mutate_source_score(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,_,_,state,bank,route=self.fixture(folder)
            route.consider(state,'first');route.consider(state,'again')
            self.assertEqual(route.checks,1);self.assertGreater(bank.counts['cache_hits'],0)
            slightly=replace(state,poses={**state.poses,'piece_01':translation(27.+1e-10)})
            route.consider(slightly,'different')
            self.assertEqual(route.checks,2)
            self.assertEqual(state.score,42.)
            self.assertAlmostEqual(route.states[0].score,problem.structural.evaluate(route.states[0])['score'])

    def test_unverified_candidates_abstain_without_displacing_safe_incumbent(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,state,_,route=self.fixture(folder,joint_final_check_limit=1)
            route.consider(state,'safe')
            other=replace(state,poses={**state.poses,'piece_01':translation(20)},score=200.)
            self.assertIsNone(route.consider(other,'budget_exhausted'))
            self.assertEqual(len(route.states),1)
            self.assertGreater(route.counts['unverified_abstentions'],0)

    def test_checked_mode_rejects_complete_collisions_and_keeps_trees_legal(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,state,_,route=self.fixture(folder,checked=True)
            self.assertTrue(route.admit_complete(state))
            bad=replace(state,poses={**state.poses,'piece_01':translation(20)})
            self.assertFalse(route.admit_complete(bad))
            partial=replace(bad,selected=(),tree=(),assignments={},groups={p:p for p in bad.poses})
            self.assertTrue(route.admit_complete(partial))

    def test_near_limit_pair_check_is_bounded_and_threshold_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            _,c,_,_,bank,route=self.fixture(folder,checked=True,joint_early_pair_checks=1)
            coarse={'valid':True,'overlap':.07,'reason':'checked'}
            actual=route.check_pair('piece_00','piece_01',translation(20),coarse)
            self.assertFalse(actual['valid'])
            self.assertEqual(c.maximum_assembly_collision,.08)
            self.assertIs(route.check_pair('piece_00','piece_01',translation(19),coarse),coarse)
            self.assertEqual(bank.pair_checks,1)

    def test_cached_geometry_still_passes_through_complete_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,search,_,_,route=self.fixture(folder,checked=True)
            search.incumbents=route
            with patch.object(route,'admit_complete',return_value=False) as check:
                for _ in range(2):self.assertIsNone(search.attempt(singleton_state(p),p.edges[0]))
                self.assertEqual(check.call_count,2)
                self.assertGreater(search.execution.counts['cache_hits'],0)

    def test_three_cold_routes_share_one_global_budget_and_common_scores(self):
        for checked in (False,True):
            with self.subTest(checked=checked),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'input';build_fixture(root,rows=2,cols=2,alternatives=True)
                c=self.config(checked,max_proposals=200,joint_retention_route_proposals=80)
                p=load_problem(root,c);result=solve(p,c)
                arms=result.diagnostics['cold_portfolio']['arms']
                self.assertEqual([a['name'] for a in arms],['v4_constructions','v6_component','v8_component'])
                self.assertTrue(all(a['executed'] for a in arms))
                used=sum(a['proposals'] for a in arms)+len(result.diagnostics['attachment_deletions']['checks'])
                self.assertEqual(used,result.diagnostics['proposals']);self.assertLessEqual(used,200)
                self.assertTrue(all(len(s.poses)==4 for s in result.states))
                self.assertTrue(all(r['valid'] for r in result.diagnostics['final_feasibility']))
                self.assertFalse(result.diagnostics['cold_portfolio']['initial_result_seeds'])
                for state in result.states:self.assertAlmostEqual(state.score,p.structural.evaluate(state)['score'])

    def test_tiny_budget_and_initial_result_seeds_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'input';build_fixture(root,rows=1,cols=2)
            c=self.config(max_proposals=1)
            p=load_problem(root,c);result=solve(p,c)
            self.assertLessEqual(result.diagnostics['proposals'],1)
            p=load_problem(root,c);p.initial_states=[singleton_state(p)]
            with self.assertRaisesRegex(ValueError,'cold states'):solve(p,c)


if __name__=='__main__':unittest.main()
