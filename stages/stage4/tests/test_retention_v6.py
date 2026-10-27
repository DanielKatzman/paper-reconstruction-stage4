import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_stage4.attachments import evaluate, without_join
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import State, singleton_state
from paper_reconstruction_stage4.search.common import identity
from paper_reconstruction_stage4.search.joint_beam import JointSearch
from paper_reconstruction_stage4.search.component_beam import family_order
from paper_reconstruction_stage4.solver import solve
from paper_reconstruction_stage4.support import SupportIndex
from tests.fixtures import build_fixture, translation


class RetentionV6Tests(unittest.TestCase):
    def config(self,**kw):
        args=dict(solver='joint_beam_v3',objective_version='observable-joint.v6-attachment',
            joint_search_policy='cold-portfolio.v1',max_seconds=5.,joint_wall_seconds=10.,
            joint_finalize_seconds=1.,max_proposals=100,joint_trials=4,beam_width=8,portfolio_size=3)
        args.update(kw)
        return SolverConfig(**args).validate()

    def fixture(self,folder,**kw):
        root=Path(folder)/'input';build_fixture(root,**kw)
        config=self.config();problem=load_problem(root,config);support=SupportIndex(problem,config)
        problem.structural=JointScorer(problem,config,support)
        return problem,config,support

    def test_previous_v5_config_roundtrip_is_exact_and_new_parameters_are_versioned(self):
        root=Path(__file__).resolve().parents[1]
        import json
        from importlib.util import spec_from_file_location,module_from_spec
        import sys
        spec=spec_from_file_location('_frozen_v5_config',root/'controls/stage4_quality_v5_combined_development/src/paper_reconstruction_stage4/config.py')
        module=module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        for path in (root/'controls/stage4_quality_v5_combined_development/configs').glob('*.json'):
            value=json.loads(path.read_text())
            self.assertEqual(SolverConfig.from_dict(value).to_dict(),module.SolverConfig.from_dict(value).to_dict())
        with self.assertRaises(ValueError):SolverConfig(attachment_uncertainty_cost=.9).validate()
        with self.assertRaises(ValueError):SolverConfig(joint_portfolio_fraction=.2).validate()

    def test_cold_portfolio_rejects_warm_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,support=self.fixture(folder,rows=1,cols=2)
            problem.initial_states=[singleton_state(problem)]
            with self.assertRaisesRegex(ValueError,'initial-result'):
                JointSearch(problem,config,Feasibility(problem,config),support).solve()

    def test_shared_budget_and_all_arm_candidates_survive_common_objective_union(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,support=self.fixture(folder,rows=2,cols=2,alternatives=True)
            result=JointSearch(problem,config,Feasibility(problem,config),support).solve()
            arms=result.diagnostics['cold_portfolio']['arms']
            self.assertEqual(len(arms),2)
            self.assertLessEqual(sum(a.get('proposals',0) for a in arms),config.max_proposals)
            self.assertEqual(result.diagnostics['proposals'],sum(a.get('proposals',0) for a in arms))
            for arm in arms:
                for candidate in arm.get('candidates',[]):
                    self.assertTrue(any(set(s.selected)==set(candidate['selected_geometry_ids']) and
                        abs(s.score-candidate['common_score'])<1e-8 for s in result.states))
            self.assertFalse(result.diagnostics['cold_portfolio']['cross_arm_pruning'])
            self.assertFalse(result.diagnostics['evaluation_truth_opened'])

    def test_proposal_exhaustion_and_final_guards_keep_all_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,_=self.fixture(folder,rows=3,cols=3)
            config=replace(config,max_proposals=4)
            result=solve(problem,config)
            self.assertLessEqual(result.diagnostics['proposals'],4)
            self.assertTrue(all(len(s.poses)==9 for s in result.states))
            self.assertTrue(all(v['valid'] for v in result.diagnostics['final_feasibility']))

    def test_join_deletion_splits_graph_without_moving_any_piece(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,_,_=self.fixture(folder,rows=1,cols=3)
            ids=[e.id for e in problem.edges];poses={p:translation(i*27) for i,p in enumerate(sorted(problem.pieces))}
            state=State(poses,{p:'piece_00' for p in poses},tuple(ids),tuple(ids),{e.id:e.occurrences[0]['occurrence_id'] for e in problem.edges})
            dropped=without_join(state,{e.id:e for e in problem.edges},ids[0])
            self.assertEqual(sorted(map(len,dropped.components())),[1,2])
            self.assertEqual(len(state.selected),2)
            for p in poses:np.testing.assert_array_equal(dropped.poses[p],state.poses[p])

    def test_weak_tree_join_has_soft_cost_not_a_mandatory_second_contact(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,support=self.fixture(folder,rows=1,cols=2)
            edge=problem.edges[0]
            edge.raw['pose']['alignment']['underconstrained']=True
            edge.occurrences[0]['pose']['alignment']['underconstrained']=True
            state=State({'piece_00':np.eye(3),'piece_01':translation(27)},
                {'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),(edge.id,),{edge.id:edge.occurrences[0]['occurrence_id']})
            result=problem.structural.evaluate(state)
            self.assertGreater(result['penalties']['attachment_uncertainty'],0.)
            self.assertGreater(result['score'],problem.structural.evaluate(singleton_state(problem))['score'])
            self.assertFalse(result['attachment_evidence']['bridges'][0]['hard_rejection'])

    def test_axis_disagreement_cost_is_confidence_gated(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,support=self.fixture(folder,rows=1,cols=2)
            edge=problem.edges[0];edge.occurrences[0]['pose']['alignment']['underconstrained']=True
            state=State({'piece_00':np.eye(3),'piece_01':translation(27)},
                {'piece_00':'piece_00','piece_01':'piece_00'},(edge.id,),assignments={edge.id:edge.occurrences[0]['occurrence_id']})
            scorer=problem.structural
            for p in scorer.profiles.values():p.update(axis_confidence=1.,angle_radians=0.)
            base={'corroborating_unselected_pairs':[]}
            first=evaluate(scorer,state,base)['cost']
            scorer.profiles['piece_01']['angle_radians']=np.pi/2;scorer.attachment_cache.clear()
            second=evaluate(scorer,state,base)['cost']
            self.assertGreater(second,first)
            scorer.profiles['piece_01']['axis_confidence']=.1;scorer.attachment_cache.clear()
            self.assertAlmostEqual(evaluate(scorer,state,base)['cost'],first)

    def test_corroboration_requires_assignable_support_and_is_gauge_invariant(self):
        with tempfile.TemporaryDirectory() as folder:
            problem,config,support=self.fixture(folder,rows=2,cols=2)
            edges=problem.edges;selected=edges[:3];extra=edges[3]
            for edge in selected:edge.occurrences[0]['pose']['alignment']['underconstrained']=True
            poses={p:translation(i%2*27,i//2*27) for i,p in enumerate(sorted(problem.pieces))}
            state=State(poses,{p:'piece_00' for p in poses},tuple(e.id for e in selected),
                assignments={e.id:e.occurrences[0]['occurrence_id'] for e in selected})
            base={'corroborating_unselected_pairs':[{'geometry_id':extra.id,'pieces':[extra.a,extra.b],'crop_probe_rms_px':0.}]}
            with patch.object(support,'assign',return_value=(None,'support_conflict',1)):
                blocked=evaluate(problem.structural,state,base)
            problem.structural.attachment_cache.clear()
            assigned={**state.assignments,extra.id:extra.occurrences[0]['occurrence_id']}
            with patch.object(support,'assign',return_value=(assigned,'assigned',1)):
                allowed=evaluate(problem.structural,state,base)
                self.assertLess(allowed['cost'],blocked['cost'])
                rotation=np.array([[0.,-2.,70.],[2.,0.,-30.],[0.,0.,1.]])
                moved=replace(state,poses={p:rotation@h for p,h in poses.items()})
                self.assertAlmostEqual(evaluate(problem.structural,moved,base)['cost'],allowed['cost'])

            problem.structural.attachment_cache.clear()
            # An alternate assignment could free an old occupied arc; the
            # actual selected state still occupies it and must get no credit.
            support.arcs[extra.occurrences[0]['occurrence_id']]=support.arcs[selected[0].occurrences[0]['occurrence_id']]
            with patch.object(support,'assign',return_value=(assigned,'assigned',1)):
                occupied=evaluate(problem.structural,state,base)
                self.assertAlmostEqual(occupied['cost'],blocked['cost'])

    def test_rounded_duplicate_union_keeps_the_higher_score(self):
        from paper_reconstruction_stage4.search.common import retain
        first=State({'p':np.eye(3)},{'p':'p'},score=2.)
        near=replace(first,poses={'p':translation(1e-8)},score=1.)
        self.assertEqual(identity(first),identity(near))
        self.assertEqual(retain([first,near],2)[0].score,2.)

    def test_family_order_visits_other_physical_attachments_before_more_local_poses(self):
        from types import SimpleNamespace
        a=SimpleNamespace(id='a',a='p0',b='p1');b=SimpleNamespace(id='b',a='p0',b='p1');c=SimpleNamespace(id='c',a='p0',b='p2')
        search=SimpleNamespace(candidates=[a,b,c],priorities={'a':1.,'b':.9,'c':.5})
        state=State({p:np.eye(3) for p in ('p0','p1','p2')},{p:p for p in ('p0','p1','p2')})
        self.assertEqual([e.id for e in family_order(search,state)],['a','c','b'])

    def test_controller_ablation_cannot_increase_budgets_or_mix_policy_bindings(self):
        import json
        from benchmark.retention_experiments import VARIANTS,validate_variant
        root=Path(__file__).resolve().parents[1]
        baseline=SolverConfig.from_dict(json.loads((root/'configs/joint_scale_component_beam_development.json').read_text())).to_dict()
        for name,path in VARIANTS.items():
            config=SolverConfig.from_dict(json.loads((root/'configs'/path).read_text())).to_dict()
            validate_variant(name,config,baseline)
            with self.assertRaisesRegex(ValueError,'budgets'):
                validate_variant(name,{**config,'max_proposals':999999},baseline)
        with self.assertRaisesRegex(ValueError,'binding'):
            validate_variant('retention',{**baseline,'joint_search_policy':'component-beam.v2'},baseline)

    def test_retention_diagnostics_cannot_open_references_before_predictions(self):
        from benchmark.retention_experiments import diagnose
        from benchmark.common import atomic_json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'request.json',{})
            atomic_json(root/'progress.json',{'phase':'predictions'})
            with self.assertRaisesRegex(ValueError,'barrier'):
                diagnose(['--run',str(root)])

    def test_diagnostic_bootstrap_uses_the_result_snapshot_in_a_fresh_interpreter(self):
        import subprocess,sys
        from benchmark.common import atomic_json,read_json
        root=Path(__file__).resolve().parents[1]
        snapshot=root/'controls/stage4_quality_v5_combined_development'
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            atomic_json(run/'progress.json',{'phase':'finished'})
            atomic_json(run/'request.json',{'snapshot':str(snapshot),'implementation':read_json(snapshot/'receipt.json')['implementation']})
            code='from benchmark.snapshot_diagnosis import select_core; import sys; print(select_core(sys.argv[1])["sha256"])'
            result=subprocess.run([sys.executable,'-B','-c',code,str(run)],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),read_json(snapshot/'receipt.json')['implementation']['sha256'])

    def test_snapshot_bootstrap_obeys_the_prediction_barrier_before_loading_source(self):
        from benchmark.snapshot_diagnosis import select_core
        from benchmark.common import atomic_json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'progress.json',{'phase':'predictions'})
            with self.assertRaisesRegex(ValueError,'barrier'):
                select_core(root)
