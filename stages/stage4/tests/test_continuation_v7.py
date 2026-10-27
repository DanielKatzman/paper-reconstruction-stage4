import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.support import SupportIndex
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.continuation import Continuation,image_profile,sample
from paper_reconstruction_stage4.model import State,singleton_state
from paper_reconstruction_stage4.solver import solve
from tests.fixtures import build_fixture,translation


class ContinuationTests(unittest.TestCase):
    def config(self,**kw):
        values=dict(solver='joint_beam_v3',objective_version='observable-joint.v7-continuation',
            joint_search_policy='cold-portfolio.v1',joint_portfolio_component_policy='component-beam.v1',
            joint_execution_policy='exact-reuse.v1',attachment_uncertainty_cost=0.,
            max_proposals=100,max_seconds=10.,joint_wall_seconds=20.,joint_finalize_seconds=1.,
            beam_width=8,portfolio_size=3,joint_trials=4)
        values.update(kw);return SolverConfig(**values).validate()

    def fixture(self,folder,mismatch=False,**kw):
        root=Path(folder)/'input';build_fixture(root,rows=1,cols=2)
        config=self.config(**kw);problem=load_problem(root,config)
        for index,(pid,piece) in enumerate(problem.pieces.items()):
            rgb=np.full((32,32,3),255,np.uint8)
            if index==0 or not mismatch:rgb[8:24,2:30]=[20,70,20]
            path=Path(folder)/(pid+'.png');Image.fromarray(rgb).save(path);piece.image_path=path
        support=SupportIndex(problem,config);edge=problem.edges[0]
        edge.occurrences[0]['pose']['alignment']['underconstrained']=True
        scorer=JointScorer(problem,config,support);problem.structural=scorer
        for profile in scorer.profiles.values():profile['axis_confidence']=.1
        state=State({'piece_00':np.eye(3),'piece_01':translation(27)},
            {'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),(edge.id,),
            {edge.id:edge.occurrences[0]['occurrence_id']})
        return problem,config,scorer,state

    def test_legacy_configurations_keep_exact_normalization_and_parameters_are_versioned(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'controls/stage4_speed_reuse_v1/configs').glob('*.json'):
            value=json.loads(path.read_text());self.assertEqual(SolverConfig.from_dict(value).to_dict(),value)
        with self.assertRaisesRegex(ValueError,'versioned continuation'):
            SolverConfig(continuation_weight=.5).validate()
        for kw in ({'continuation_weight':3.},{'continuation_min_coverage':0.},
                   {'continuation_samples':1000},{'continuation_image_side':4096}):
            with self.assertRaises(ValueError):self.config(**kw)

    def test_consistent_colours_have_less_cost_than_visible_discontinuity(self):
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            _,_,a,sa=self.fixture(first);_,_,b,sb=self.fixture(second,mismatch=True)
            good=a.evaluate(sa)['continuation_evidence'];bad=b.evaluate(sb)['continuation_evidence']
            self.assertLess(good['cost'],1e-6)
            self.assertGreater(bad['cost'],.15)
            self.assertFalse(bad['hard_rejection'])
            self.assertTrue(bad['joins'][0]['bidirectional'])

    def test_strong_text_abstains_without_reading_extra_images(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,scorer,state=self.fixture(folder,mismatch=True)
            for profile in scorer.profiles.values():profile['axis_confidence']=1.
            with patch('paper_reconstruction_stage4.continuation.image_profile',side_effect=AssertionError('unexpected image')):
                result=scorer.evaluate(state)
            self.assertEqual(result['continuation_evidence']['cost'],0.)
            self.assertEqual(scorer.continuation_context.report()['profiled_pieces'],0)

    def test_unassigned_proposal_does_not_receive_fallback_penalty(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,scorer,state=self.fixture(folder,mismatch=True)
            state.assignments={}
            self.assertEqual(scorer.evaluate(state)['continuation_evidence']['cost'],0.)

    def test_missing_visible_band_coverage_abstains_instead_of_using_background(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,_,scorer,state=self.fixture(folder,mismatch=True)
            problem.pieces['piece_01'].mask[:]=False
            result=scorer.evaluate(state)['continuation_evidence']
            self.assertEqual(result['cost'],0.)
            self.assertIsNone(result['joins'][0]['mismatch'])

    def test_blank_white_piece_is_not_dark_point_stretched(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,scorer,state=self.fixture(folder,mismatch=True)
            for piece in problem.pieces.values():
                Image.fromarray(np.full((32,32,3),230,np.uint8)).save(piece.image_path)
            self.assertEqual(scorer.evaluate(state)['continuation_evidence']['cost'],0.)

    def test_masked_background_is_not_used_as_colour_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,_,_,_=self.fixture(folder)
            piece=problem.pieces['piece_00'];rgb=np.full((32,32,3),230,np.uint8)
            rgb[~piece.mask]=[255,0,0];Image.fromarray(rgb).save(piece.image_path)
            profile=image_profile(piece,16)
            values,visible=sample(profile,np.array([[15.,15.],[31.,31.]]))
            self.assertTrue(visible[0]);self.assertFalse(visible[1])
            self.assertLess(float(np.max(values[0])-np.min(values[0])),.01)

    def test_no_reliable_paper_normalization_abstains(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,_,scorer,state=self.fixture(folder,mismatch=True)
            Image.fromarray(np.tile(np.array([10,150,10],np.uint8),(32,32,1))).save(problem.pieces['piece_00'].image_path)
            result=scorer.evaluate(state)['continuation_evidence']
            self.assertEqual(result['cost'],0.)
            self.assertFalse(result['joins'][0]['normalization_reliable'])

    def test_cost_is_gauge_invariant_and_cached_by_native_proposal_support(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,scorer,state=self.fixture(folder,mismatch=True)
            old=scorer.evaluate(state)['continuation_evidence']['cost']
            gauge=np.array([[0.,-2.,150.],[2.,0.,30.],[0.,0.,1.]])
            moved=replace(state,poses={p:gauge@h for p,h in state.poses.items()})
            self.assertEqual(scorer.evaluate(moved)['continuation_evidence']['cost'],old)
            self.assertGreater(scorer.continuation_context.counts['cache_hits'],0)

    def test_assignable_corroboration_discounts_soft_continuation_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            _,config,scorer,state=self.fixture(folder,mismatch=True)
            context=Continuation(scorer);edge=state.selected[0]
            bridge={'geometry_id':edge,'pieces':['piece_00','piece_01'],'axis_confidence':.1,'assignable_corroboration':False}
            old=context.evaluate(state,{'bridges':[bridge]})['cost'];bridge['assignable_corroboration']=True
            new=context.evaluate(state,{'bridges':[bridge]})['cost']
            self.assertAlmostEqual(new,old*(1-config.attachment_corroboration_discount))

    def test_selected_cycles_are_not_treated_as_unsupported_bridges(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'input';build_fixture(root,rows=2,cols=2)
            config=self.config();problem=load_problem(root,config);support=SupportIndex(problem,config)
            for edge in problem.edges:edge.occurrences[0]['pose']['alignment']['underconstrained']=True
            scorer=JointScorer(problem,config,support)
            for p in scorer.profiles.values():p['axis_confidence']=.1
            poses={p:translation(i%2*27,i//2*27) for i,p in enumerate(sorted(problem.pieces))}
            ids=tuple(e.id for e in problem.edges)
            state=State(poses,{p:'piece_00' for p in poses},ids,ids[:3],
                {e.id:e.occurrences[0]['occurrence_id'] for e in problem.edges})
            self.assertEqual(scorer.evaluate(state)['continuation_evidence']['cost'],0.)
            self.assertEqual(scorer.continuation_context.report()['profiled_pieces'],0)

    def test_occurrence_keys_and_measurement_cache_capacity_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,scorer,state=self.fixture(folder,mismatch=True,maximum_cache_entries=1)
            first=scorer.evaluate(state)['continuation_evidence']['cost']
            edge=state.selected[0];old=state.assignments[edge];new='different-assigned-occurrence'
            scorer.support.arcs[new]=scorer.support.arcs[old]
            moved=replace(state,assignments={edge:new})
            # Call the measurement directly to test native cache identity;
            # occurrence metadata lookup belongs to the attachment layer.
            a=scorer.continuation_context.measurement(state,edge)
            b=scorer.continuation_context.measurement(moved,edge)
            self.assertEqual(a['mismatch'],b['mismatch'])
            self.assertEqual(len(scorer.continuation_context.cache),1)
            self.assertGreaterEqual(scorer.continuation_context.counts['cache_misses'],2)

    def test_accuracy_runner_forbids_unrelated_changes_and_obeys_barrier(self):
        from benchmark.accuracy_experiments import validate_accuracy_config,diagnose
        from benchmark.common import atomic_json
        baseline=replace(self.config(),objective_version='observable-joint.v6-attachment').to_dict()
        validate_accuracy_config('continuation',self.config().to_dict(),baseline)
        for key in ('max_proposals','attachment_axis_weight','joint_portfolio_fraction'):
            changed={**self.config().to_dict(),key:self.config().to_dict()[key]+1}
            with self.assertRaisesRegex(ValueError,'unrelated score policy'):
                validate_accuracy_config('continuation',changed,baseline)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/'request.json',{});atomic_json(root/'progress.json',{'phase':'predictions'})
            with self.assertRaisesRegex(ValueError,'barrier'):diagnose(['--run',str(root)])

    def test_accuracy_publication_resume_and_diagnostic_report(self):
        import contextlib
        import io
        from benchmark.common import atomic_json,digest,file_hash,read_json
        from benchmark.accuracy_experiments import diagnose
        from paper_reconstruction_stage4.cli import main as cli
        from paper_reconstruction_stage4.outputs import implementation_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cid='known_grid_fixture';integration=root/'integration';bundle=integration/'cases'/cid/'stage3'
            reference=build_fixture(bundle,rows=2,cols=2,alternatives=True)
            baseline=root/'baseline';run=root/'run';cohort=root/'cohort'
            candidate=self.config();original=replace(candidate,objective_version='observable-joint.v6-attachment')
            old_config=root/'old.json';new_config=root/'new.json'
            atomic_json(old_config,original.to_dict());atomic_json(new_config,candidate.to_dict())
            prior=baseline/'cases'/cid/'reuse/stage4';target=run/'cases'/cid/'continuation/stage4'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli(['solve','--bundle',str(bundle),'--config',str(old_config),'--output',str(prior)]),0)
                args=['solve','--bundle',str(bundle),'--config',str(new_config),'--output',str(target)]
                self.assertEqual(cli(args),0);sealed=file_hash(target/'result.json')
                self.assertEqual(cli([*args,'--resume']),0);self.assertEqual(file_hash(target/'result.json'),sealed)
            atomic_json(baseline/'experiment_receipt.json',{'fixture':True})
            atomic_json(integration/'integration_receipt.json',{'fixture':True})
            atomic_json(cohort/'puzzle/evaluation_only/reference_geometry.json',reference)
            identity=implementation_identity()
            request={'schema_version':'stage4.v7-accuracy-request.v1','baseline':str(baseline),'integration':str(integration),
                'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),
                'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
                'baseline_request_sha256':'fixture','benchmark_manifest':'fixture','controllers':{},
                'configurations':{'continuation':{'path':str(new_config),'sha256':file_hash(new_config)}},
                'snapshot':str(root/'snapshot'),'implementation':identity,'variants':['continuation'],'case_ids':[cid]}
            request['request_sha256']=digest(request);atomic_json(run/'request.json',request)
            atomic_json(run/'progress.json',{'phase':'diagnostics','cases':{cid:{'continuation':{'status':'predicted','result_sha256':sealed}}}})
            with patch('benchmark.accuracy_experiments.checked_baseline',return_value=({'request_sha256':'fixture'},{},{})), \
                    patch('benchmark.accuracy_experiments.verify_snapshot',return_value={'implementation':identity}), \
                    patch('benchmark.accuracy_experiments.load_cohort',return_value=(cohort,{'cases':[{'case_id':cid,'path':'puzzle'}]})):
                self.assertEqual(diagnose(['--run',str(run)]),0)
            summary=read_json(run/'report/summary.json')
            self.assertTrue(summary['cases'][0]['support_verification']['valid'])
            self.assertTrue((run/'report/index.html').is_file())
            self.assertEqual(file_hash(target/'result.json'),sealed)

    def test_zero_weight_preserves_v6_score_and_skips_image_work(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,scorer,state=self.fixture(folder,mismatch=True,continuation_weight=0.)
            off=scorer.evaluate(state)
            baseline=replace(config,objective_version='observable-joint.v6-attachment',continuation_weight=1.25)
            legacy=JointScorer(problem,baseline,scorer.support)
            for p in legacy.profiles.values():p['axis_confidence']=.1
            self.assertEqual(off['score'],legacy.evaluate(state)['score'])
            self.assertEqual(scorer.continuation_context.report()['profiled_pieces'],0)

    def test_v7_cold_portfolio_and_final_checks_preserve_inventory_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'input';build_fixture(root,rows=2,cols=2,alternatives=True)
            config=self.config(continuation_weight=.5,max_proposals=70)
            result=solve(load_problem(root,config),config)
            self.assertLessEqual(result.diagnostics['proposals'],70)
            self.assertTrue(all(len(s.poses)==4 for s in result.states))
            self.assertTrue(all(r['valid'] for r in result.diagnostics['final_feasibility']))
            self.assertFalse(result.diagnostics['evaluation_truth_opened'])
            self.assertFalse(result.diagnostics['cold_portfolio']['initial_result_seeds'])


if __name__=='__main__':unittest.main()
