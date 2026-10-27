import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.geometry import probes,warp_check
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.search.common import identity
from paper_reconstruction_stage4.search.execution import ExactReuse, state_key
from paper_reconstruction_stage4.search.joint_beam import JointSearch
from paper_reconstruction_stage4.active_support import ActiveSupportIndex
from paper_reconstruction_stage4.solver import solve
from tests.fixtures import build_fixture


class SpeedReuseTests(unittest.TestCase):
    def config(self, **kw):
        values=dict(solver='joint_beam_v3',objective_version='observable-joint.v6-attachment',
            joint_search_policy='cold-portfolio.v1',joint_portfolio_component_policy='component-beam.v1',
            attachment_uncertainty_cost=0.,support_policy='active-subarc.v1',
            max_seconds=30.,joint_wall_seconds=40.,joint_finalize_seconds=1.,
            max_proposals=350,beam_width=8,portfolio_size=3,joint_trials=4)
        values.update(kw)
        return SolverConfig(**values).validate()

    def fixture(self, folder, **kw):
        root=Path(folder)/'input';build_fixture(root,rows=2,cols=2,alternatives=True)
        config=self.config(joint_execution_policy='exact-reuse.v1',**kw)
        problem=load_problem(root,config);support=ActiveSupportIndex(problem,config)
        problem.structural=JointScorer(problem,config,support)
        return problem,config,JointSearch(problem,config,Feasibility(problem,config),support)

    def test_historical_config_serialization_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'controls/stage4_retention_v6_conservative_development/configs').glob('*.json'):
            value=json.loads(path.read_text())
            self.assertEqual(SolverConfig.from_dict(value).to_dict(),value)
        with self.assertRaisesRegex(ValueError,'Execution reuse'):
            SolverConfig(joint_execution_policy='exact-reuse.v1').validate()
        with self.assertRaisesRegex(ValueError,'execution policy'):
            self.config(joint_execution_policy='unknown')

    def test_static_probe_cache_is_exact_and_mutation_isolated(self):
        for shape,grid in (((31,47),2),((38,73),5),((1,1),3)):
            x,y=np.meshgrid(np.linspace(0,shape[1]-1,grid),np.linspace(0,shape[0]-1,grid))
            expected=np.column_stack((x.ravel(),y.ravel()))
            first=probes(shape,grid);first[:]=-42
            np.testing.assert_array_equal(probes(shape,grid),expected)

    def test_affine_fast_path_matches_sampled_warp_guards_and_projective_fallback(self):
        original=self.config();fast=replace(original,joint_execution_policy='exact-reuse.v1')
        rng=np.random.default_rng(19)
        matrices=[np.eye(3),np.diag([-1.,1.,1.]),np.diag([1e-14,1.,1.]),
            np.array([[1.,0.,3.],[0.,1.,-9.],[.0001,-.0002,1.]])]
        for scale in (original.minimum_local_scale,original.maximum_local_scale):
            for v in (np.nextafter(scale,0.),scale,np.nextafter(scale,float('inf'))):
                matrices.append(np.diag([v,v,1.]))
        for _ in range(80):
            h=np.eye(3);h[:2,:2]=rng.normal(size=(2,2))*rng.uniform(.1,8.)
            h[:2,2]=rng.normal(size=2)*200;matrices.append(h)
        for h in matrices:
            self.assertEqual(warp_check(h,(340,460),(280,560),original),
                             warp_check(h,(340,460),(280,560),fast))

    def test_speed_runner_rejects_score_and_budget_changes(self):
        from benchmark.speed_experiments import validate_speed_config
        value=self.config().to_dict();fast={**value,'joint_execution_policy':'exact-reuse.v1'}
        validate_speed_config(fast,value)
        for key in ('max_proposals','attachment_axis_weight','joint_portfolio_fraction'):
            changed={**fast,key:fast[key]+1}
            with self.assertRaisesRegex(ValueError,'only the execution policy'):
                validate_speed_config(changed,value)

    def test_speed_diagnostics_obey_the_serving_barrier(self):
        from benchmark.common import atomic_json
        from benchmark.speed_experiments import diagnose
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/'request.json',{})
            atomic_json(root/'progress.json',{'phase':'predictions'})
            with self.assertRaisesRegex(ValueError,'barrier'):
                diagnose(['--run',str(root)])

    def test_speed_publication_resume_and_comparison_report_on_observable_fixture(self):
        import contextlib
        import io
        from benchmark.common import atomic_json,digest,file_hash,read_json
        from benchmark.speed_experiments import diagnose
        from paper_reconstruction_stage4.cli import main as cli
        from paper_reconstruction_stage4.outputs import implementation_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cid='known_grid_fixture';integration=root/'integration'
            bundle=integration/'cases'/cid/'stage3'
            reference=build_fixture(bundle,rows=2,cols=2,alternatives=True)
            baseline=root/'baseline';run=root/'run';cohort=root/'cohort'
            baseline_config=self.config();fast=replace(baseline_config,joint_execution_policy='exact-reuse.v1')
            first=root/'old.json';second=root/'new.json'
            atomic_json(first,baseline_config.to_dict());atomic_json(second,fast.to_dict())
            prior=baseline/'cases'/cid/'conservative/stage4';target=run/'cases'/cid/'reuse/stage4'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli(['solve','--bundle',str(bundle),'--config',str(first),'--output',str(prior)]),0)
                args=['solve','--bundle',str(bundle),'--config',str(second),'--output',str(target)]
                self.assertEqual(cli(args),0)
                sealed=file_hash(target/'result.json')
                self.assertEqual(cli([*args,'--resume']),0)
                self.assertEqual(file_hash(target/'result.json'),sealed)
            atomic_json(baseline/'experiment_receipt.json',{'fixture':True})
            atomic_json(integration/'integration_receipt.json',{'fixture':True})
            atomic_json(cohort/'puzzle/evaluation_only/reference_geometry.json',reference)
            identity=implementation_identity()
            request={'schema_version':'stage4.v6-speed-request.v1','baseline':str(baseline),
                'integration':str(integration),'baseline_receipt_sha256':file_hash(baseline/'experiment_receipt.json'),
                'integration_receipt_sha256':file_hash(integration/'integration_receipt.json'),
                'baseline_request_sha256':'fixture','benchmark_manifest':'fixture','controllers':{},
                'configuration':{'path':str(second),'sha256':file_hash(second)},
                'snapshot':str(root/'snapshot'),'implementation':identity,'case_ids':[cid]}
            request['request_sha256']=digest(request)
            atomic_json(run/'request.json',request)
            atomic_json(run/'progress.json',{'phase':'diagnostics','cases':{cid:{'status':'predicted','result_sha256':sealed}}})
            with patch('benchmark.speed_experiments.checked_baseline',return_value=({'request_sha256':'fixture'},{},{})), \
                    patch('benchmark.speed_experiments.verify_snapshot',return_value={'implementation':identity}), \
                    patch('benchmark.speed_experiments.load_cohort',return_value=(cohort,{'cases':[{'case_id':cid,'path':'puzzle'}]})):
                self.assertEqual(diagnose(['--run',str(run)]),0)
            summary=read_json(run/'report/summary.json')
            self.assertTrue(summary['cases'][0]['preview_bytes_identical'])
            self.assertTrue(summary['cases'][0]['support_verification']['valid'])
            self.assertTrue((run/'report/index.html').is_file())
            self.assertEqual(file_hash(target/'result.json'),sealed)

    def test_cache_reuses_success_without_sharing_mutable_candidate_data(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder)
            source=singleton_state(problem)
            edge=next(e for e in problem.edges if search.attempt(source,e) is not None)
            first=search.attempt(source,edge);expected=copy.deepcopy(first)
            first.poses[edge.a][0,2]+=100;first.assignments.clear()
            repeated=search.attempt(source,edge)
            self.assertEqual(identity(repeated),identity(expected))
            self.assertEqual(repeated.active_support,expected.active_support)
            self.assertGreater(search.execution.counts['cache_hits'],0)

    def test_cache_keys_preserve_subpixel_pose_and_active_record_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder)
            first=singleton_state(problem);second=copy.deepcopy(first)
            second.poses[min(second.poses)][0,2]=1e-10
            self.assertNotEqual(state_key(first),state_key(second))
            first.active_support={'a':{'support_id':'same','coverage':[1.,1.]}}
            second=copy.deepcopy(first);second.active_support['a']['coverage'][0]=.9
            self.assertNotEqual(state_key(first),state_key(second))

    def test_cross_objective_hit_is_rescored_with_requesting_scorer(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder)
            source=singleton_state(problem)
            edge=next(e for e in problem.edges if search.attempt(source,e) is not None)
            class ConstantScore:
                def evaluate(self,state):return {'score':123.25}
            search.problem=replace(problem,structural=ConstantScore())
            result=search.attempt(source,edge)
            self.assertEqual(result.score,123.25)
            self.assertGreater(search.execution.counts['score_replay_calls'],0)

    def test_objective_exceptions_are_not_reused_as_cross_policy_geometry_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder);source=singleton_state(problem)
            edge=next(e for e in problem.edges if search.attempt(source,e) is not None)
            search.execution.cache.clear()
            class BrokenScore:
                def evaluate(self,state):raise ValueError('policy-specific score failure')
            original=search.problem
            search.problem=replace(problem,structural=BrokenScore())
            self.assertIsNone(search.attempt(source,edge))
            self.assertEqual(search.execution.counts['exception_results_not_cached'],1)
            search.problem=original
            self.assertIsNotNone(search.attempt(source,edge))

    def test_cached_hit_score_exception_matches_uncached_rejection_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder);source=singleton_state(problem)
            edge=next(e for e in problem.edges if search.attempt(source,e) is not None)
            class BrokenScore:
                def evaluate(self,state):raise ValueError('policy-specific score failure')
            original=search.problem;search.problem=replace(problem,structural=BrokenScore())
            before=search.rejections['invalid_composed_transform']
            self.assertIsNone(search.attempt(source,edge))
            self.assertEqual(search.rejections['invalid_composed_transform'],before+1)
            search.problem=original
            self.assertIsNotNone(search.attempt(source,edge))

    def test_rejected_attempt_replays_reason_but_uses_bounded_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder)
            source=singleton_state(problem)
            # A competing geometry is rejected before physical/support checks.
            edge=problem.edges[0];source.selected=(edge.id,)
            self.assertIsNone(search.attempt(source,edge))
            misses=search.execution.counts['cache_misses']
            self.assertIsNone(search.attempt(source,edge))
            self.assertEqual(search.execution.counts['cache_misses'],misses)
            self.assertEqual(search.execution.counts['rejected_attempt_hits'],1)
            search.execution.capacity=2
            for e in problem.edges[:4]:search.attempt(singleton_state(problem),e)
            self.assertLessEqual(len(search.execution.cache),2)
            self.assertGreater(search.execution.counts['evictions'],0)

    def test_cached_and_uncached_fixture_search_keep_same_primaries_and_work_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'input';build_fixture(root,rows=2,cols=2,alternatives=True)
            baseline=self.config()
            old=solve(load_problem(root,baseline),baseline)
            fast=replace(baseline,joint_execution_policy='exact-reuse.v1')
            new=solve(load_problem(root,fast),fast)
            self.assertEqual([identity(s) for s in old.states],[identity(s) for s in new.states])
            self.assertEqual([s.score for s in old.states],[s.score for s in new.states])
            self.assertEqual(old.diagnostics['proposals'],new.diagnostics['proposals'])
            self.assertEqual(old.rejections,new.rejections)
            self.assertEqual(old.diagnostics['assignment_nodes'],new.diagnostics['assignment_nodes'])
            self.assertLessEqual(new.diagnostics['proposals'],fast.max_proposals)
            self.assertTrue(all(r['valid'] for r in new.diagnostics['final_feasibility']))
            self.assertGreater(new.diagnostics['execution_reuse']['counts']['cache_hits'],0)
            self.assertTrue(all(len(s.poses)==4 for s in new.states))
            self.assertFalse(new.diagnostics['evaluation_truth_opened'])

    def test_reused_hits_do_not_allow_budget_overrun(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,search=self.fixture(folder,max_proposals=5)
            result=solve(problem,config)
            self.assertLessEqual(result.diagnostics['proposals'],5)
            self.assertTrue(all(len(s.poses)==4 for s in result.states))


if __name__=='__main__':unittest.main()
