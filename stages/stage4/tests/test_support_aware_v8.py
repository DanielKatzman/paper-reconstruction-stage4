import contextlib
import copy
from dataclasses import replace
from collections import Counter
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from benchmark.common import atomic_json, digest, file_hash, read_json
from paper_reconstruction_stage4.config import SolverConfig, LOOKAHEAD_POLICY, LOOKAHEAD_DEFAULTS
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.search.common import Budget
from paper_reconstruction_stage4.search.joint_beam import JointSearch
from paper_reconstruction_stage4.search.lookahead import expand, choose_children, next_contacts
from paper_reconstruction_stage4.solver import solve
from tests.fixtures import build_fixture
from tests import test_continuation_v7 as v7

ROOT=Path(__file__).resolve().parents[1]


class SupportAwareTests(unittest.TestCase):
    def config(self,variant='combined',**kw):
        value=v7.ContinuationTests().config(objective_version='observable-joint.v6-attachment').to_dict()
        if variant != 'search':
            value['objective_version']='observable-joint.v8-continuation'
        if variant != 'score':
            value['joint_portfolio_component_policy']=LOOKAHEAD_POLICY
        value.update(kw)
        return SolverConfig.from_dict(value)

    def fixture(self,folder,**kw):
        return v7.ContinuationTests().fixture(folder,mismatch=True,objective_version='observable-joint.v8-continuation',**kw)

    def test_v8_measures_strong_axis_content_without_changing_v7(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,scorer,state=self.fixture(folder)
            for profile in scorer.profiles.values():profile['axis_confidence']=1.
            terms=scorer.evaluate(state)
            self.assertGreater(terms['continuation_evidence']['cost'],.15)
            self.assertEqual(terms['continuation_evidence']['policy'],'all-axis-native-continuation.v2')
            old=JointScorer(p,replace(c,objective_version='observable-joint.v7-continuation'),scorer.support)
            for profile in old.profiles.values():profile['axis_confidence']=1.
            self.assertEqual(old.evaluate(state)['continuation_evidence']['cost'],0.)

    def test_v8_blank_and_unreliable_content_remain_neutral(self):
        from PIL import Image
        for colour in ((230,230,230),(10,150,10)):
            with self.subTest(colour=colour),tempfile.TemporaryDirectory() as folder:
                p,c,scorer,state=self.fixture(folder)
                for piece in p.pieces.values():
                    Image.fromarray(np.tile(np.array(colour,np.uint8),(32,32,1))).save(piece.image_path)
                self.assertEqual(scorer.evaluate(state)['continuation_evidence']['cost'],0.)

    def test_v8_zero_weight_matches_v6_and_is_gauge_invariant(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,scorer,state=self.fixture(folder,continuation_weight=0.)
            terms=scorer.evaluate(state)
            old=JointScorer(p,replace(c,objective_version='observable-joint.v6-attachment',continuation_weight=1.25),scorer.support)
            for profile in old.profiles.values():profile['axis_confidence']=.1
            self.assertEqual(terms['score'],old.evaluate(state)['score'])
            self.assertEqual(scorer.continuation_context.report()['profiled_pieces'],0)
        with tempfile.TemporaryDirectory() as folder:
            _,_,scorer,state=self.fixture(folder)
            original=scorer.evaluate(state)['continuation_evidence']['cost']
            gauge=np.array([[0.,-2.,15.],[2.,0.,30.],[0.,0.,1.]])
            moved=replace(state,poses={p:gauge@h for p,h in state.poses.items()})
            self.assertEqual(original,scorer.evaluate(moved)['continuation_evidence']['cost'])

    def test_assigned_priority_observes_content_the_unassigned_priority_misses(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,scorer,state=self.fixture(folder,joint_portfolio_component_policy=LOOKAHEAD_POLICY)
            for profile in scorer.profiles.values():profile['axis_confidence']=1.
            unassigned=replace(state,assignments={})
            self.assertEqual(scorer.evaluate(unassigned)['continuation_evidence']['cost'],0.)
            search=JointSearch(p,c,Feasibility(p,c),scorer.support)
            eid=state.selected[0]
            self.assertGreater(scorer.evaluate(unassigned)['score']-search.priorities[eid],.15)
            self.assertGreater(search.ordering_audit['counts']['continuation_observed_options'],0)
            self.assertGreater(search.ordering_audit['counts']['assigned_edges'],0)

    def test_unassignable_ordering_does_not_remove_the_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,scorer,state=self.fixture(folder,joint_portfolio_component_policy=LOOKAHEAD_POLICY)
            with patch.object(scorer.support,'assign',return_value=(None,'no_valid_contact_support',0)):
                search=JointSearch(p,c,Feasibility(p,c),scorer.support)
            self.assertEqual(len(search.candidates),len(p.edges))
            self.assertEqual(search.ordering_audit['counts']['unassignable_edges'],len(p.edges))

    def test_deferred_selection_can_keep_a_better_seventh_valid_child(self):
        c=self.config(max_proposals=100)
        edges=[SimpleNamespace(id=str(i),a='p',b='p') for i in range(8)]
        children={e.id:State({'p':np.eye(3)},{'p':'p'},(e.id,),score=float(i)) for i,e in enumerate(edges)}
        search=SimpleNamespace(config=c,edges={e.id:e for e in edges},problem=object(),
            strata={e.id:('pose','support') for e in edges},rejections=Counter(),remember=Mock(),
            attempt=lambda source,e:children[e.id])
        budget=Budget(c)
        with patch('paper_reconstruction_stage4.search.lookahead.family_order',return_value=iter(edges)), \
             patch('paper_reconstruction_stage4.search.lookahead.topology',side_effect=lambda s,e:s.selected), \
             patch('paper_reconstruction_stage4.search.lookahead.pose_signature',side_effect=lambda s,p:s.selected):
            selected,extra,trace=expand(search,None,budget,40,0,6)
        self.assertEqual(trace['distinct_valid_children_compared'],8)
        self.assertIn(children['7'],selected)
        self.assertEqual(len(selected),6)
        self.assertEqual(budget.proposals,8)

    def test_lookahead_guides_selection_without_replacing_actual_scores(self):
        states=[State({'p':np.eye(3)},{'p':'p'},(str(i),),score=float(i)) for i in range(8)]
        future=[s.score for s in states];future[0]=100.
        selected=choose_children(states,future,6)
        self.assertIn(7,selected);self.assertIn(0,selected)
        self.assertEqual(states[0].score,0.)
        neutral=choose_children(states,[s.score for s in states],6)
        self.assertEqual(set(neutral),set(range(2,8)))

    def test_next_contacts_use_distinct_physical_pairs_and_reserve_an_external_probe(self):
        edges=[SimpleNamespace(id=str(i),a=a,b=b) for i,(a,b) in enumerate(
            [('a','c'),('a','c'),('b','d'),('b','d')])]
        search=SimpleNamespace(candidates=edges,priorities={e.id:10.-i for i,e in enumerate(edges)},edges={})
        state=State({p:np.eye(3) for p in 'abcd'},{p:('a' if p!='d' else 'd') for p in 'abcd'})
        with patch('paper_reconstruction_stage4.search.lookahead.family_order',return_value=iter(edges[2:])):
            result=next_contacts(search,state,'a',2)
        self.assertEqual([(e.a,e.b) for e in result],[('a','c'),('b','d')])

    def test_split_components_have_valid_gauges_without_changing_internal_placement(self):
        from paper_reconstruction_stage4.attachments import without_join
        from paper_reconstruction_stage4.geometry import relative
        from tests.fixtures import translation
        edges={'ab':SimpleNamespace(a='a',b='b'),'bc':SimpleNamespace(a='b',b='c')}
        state=State({p:translation(i*27) for i,p in enumerate('abc')},{p:'a' for p in 'abc'},
                    ('ab','bc'),('ab','bc'),{'ab':'first','bc':'second'})
        split=without_join(state,edges,'ab',canonical=True)
        for component in split.components():
            np.testing.assert_allclose(split.poses[min(component)],np.eye(3),atol=1e-10)
            for a in component:
                for b in component:np.testing.assert_allclose(relative(state.poses,a,b),relative(split.poses,a,b))
        np.testing.assert_array_equal(state.poses['b'],translation(27))

    def test_lookahead_and_deletion_share_the_global_proposal_budget(self):
        for limit in (1,20,100):
            with self.subTest(limit=limit),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'input';build_fixture(root,rows=2,cols=3,alternatives=True)
                c=self.config(max_proposals=limit)
                result=solve(load_problem(root,c),c)
                arms=result.diagnostics['cold_portfolio']['arms']
                counts=sum(a.get('proposals',0) for a in arms)
                counts+=len(result.diagnostics['attachment_deletions']['checks'])
                self.assertEqual(counts,result.diagnostics['proposals'])
                self.assertLessEqual(counts,limit)
                for arm in arms:
                    if arm.get('policy')==LOOKAHEAD_POLICY:
                        beam=arm['search']['component_beam']
                        self.assertLessEqual(beam['lookahead_proposals'],beam['lookahead_proposal_limit'])
                self.assertTrue(all(len(s.poses)==6 for s in result.states))
                self.assertTrue(all(r['valid'] for r in result.diagnostics['final_feasibility']))

    def test_legacy_config_shape_and_v4_arm_are_preserved(self):
        for folder in ('stage4_speed_reuse_v1','stage4_continuation_v7_development'):
            for path in (ROOT/'controls'/folder/'configs').glob('*.json'):
                value=read_json(path)
                self.assertFalse(set(SolverConfig.from_dict(value).to_dict()) & set(LOOKAHEAD_DEFAULTS))
        with self.assertRaises(ValueError):SolverConfig(joint_lookahead_checks=3).validate()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'input';build_fixture(root,rows=1,cols=2)
            c=self.config(joint_ordering_occurrences=4,joint_lookahead_checks=3)
            result=solve(load_problem(root,c),c)
            arm=result.diagnostics['cold_portfolio']['arms'][0]
            self.assertEqual(arm['source_objective'],'observable-joint.v4')
            self.assertFalse(result.diagnostics['cold_portfolio']['initial_result_seeds'])

    def test_runner_binds_each_ablation_and_rejects_unrelated_changes(self):
        from benchmark.support_aware_experiments import validate_variant,diagnose,CONTROLLERS
        baseline=v7.ContinuationTests().config(objective_version='observable-joint.v6-attachment').to_dict()
        for name in ('score','search','combined'):
            c=self.config(name).to_dict();validate_variant(name,c,baseline)
            for key in ('max_proposals','contact_tolerance_px','structural_angle_weight'):
                with self.assertRaisesRegex(ValueError,'unrelated policy'):
                    validate_variant(name,{**c,key:c[key]+1},baseline)
        for path in CONTROLLERS:self.assertTrue((ROOT/path).is_file(),path)
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);atomic_json(run/'progress.json',{'phase':'predictions'});atomic_json(run/'request.json',{})
            with patch('benchmark.support_aware_experiments.load_cohort',side_effect=AssertionError('reference opened')):
                with self.assertRaisesRegex(ValueError,'barrier'):diagnose(['--run',str(run)])

    def test_all_three_ablations_publish_resume_and_diagnose_after_barrier(self):
        from benchmark.support_aware_experiments import diagnose
        from paper_reconstruction_stage4.cli import main as cli
        from paper_reconstruction_stage4.outputs import implementation_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cid='known_grid_fixture';integration=root/'integration';bundle=integration/'cases'/cid/'stage3'
            reference=build_fixture(bundle,rows=2,cols=2,alternatives=True)
            baseline=root/'baseline';run=root/'run';cohort=root/'cohort'
            original=v7.ContinuationTests().config(objective_version='observable-joint.v6-attachment')
            old_config=root/'old.json';atomic_json(old_config,original.to_dict())
            prior=baseline/'cases'/cid/'reuse/stage4'
            configurations={};entries={};variants=('score','search','combined')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli(['solve','--bundle',str(bundle),'--config',str(old_config),'--output',str(prior)]),0)
                for name in variants:
                    path=root/(name+'.json');atomic_json(path,self.config(name).to_dict())
                    target=run/'cases'/cid/name/'stage4'
                    args=['solve','--bundle',str(bundle),'--config',str(path),'--output',str(target)]
                    self.assertEqual(cli(args),0);sealed=file_hash(target/'result.json')
                    self.assertEqual(cli([*args,'--resume']),0)
                    self.assertEqual(file_hash(target/'result.json'),sealed)
                    configurations[name]={'path':str(path),'sha256':file_hash(path)}
                    entries[name]={'status':'predicted','result_sha256':sealed}
            atomic_json(baseline/'experiment_receipt.json',{'fixture':True})
            atomic_json(integration/'integration_receipt.json',{'fixture':True})
            atomic_json(cohort/'puzzle/evaluation_only/reference_geometry.json',reference)
            implementation=implementation_identity()
            request={'schema_version':'stage4.v8-support-aware-request.v1','baseline':str(baseline),'integration':str(integration),
                'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),
                'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
                'baseline_request_sha256':'fixture','benchmark_manifest':'fixture','controllers':{},
                'configurations':configurations,'snapshot':str(root/'snapshot'),'implementation':implementation,
                'variants':list(variants),'case_ids':[cid]}
            request['request_sha256']=digest(request);atomic_json(run/'request.json',request)
            atomic_json(run/'progress.json',{'phase':'diagnostics','cases':{cid:entries}})
            with patch('benchmark.support_aware_experiments.checked_baseline',return_value=({'request_sha256':'fixture'},{},{})), \
                 patch('benchmark.support_aware_experiments.verify_snapshot',return_value={'implementation':implementation}), \
                 patch('benchmark.support_aware_experiments.load_cohort',return_value=(cohort,{'cases':[{'case_id':cid,'path':'puzzle'}]})):
                self.assertEqual(diagnose(['--run',str(run)]),0)
            summary=read_json(run/'report/summary.json')
            self.assertEqual(len(summary['cases']),3)
            self.assertTrue(all(c['support_verification']['valid'] for c in summary['cases']))
            self.assertFalse(summary['default_promoted'])
            for name in variants:
                self.assertEqual(file_hash(run/'cases'/cid/name/'stage4/result.json'),entries[name]['result_sha256'])


if __name__=='__main__':unittest.main()
