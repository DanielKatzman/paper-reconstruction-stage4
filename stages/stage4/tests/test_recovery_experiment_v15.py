from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from benchmark.common import digest,read_json,atomic_json,WORKSPACE
from benchmark.recovery_experiment_v15 import ARMS,command_for,run_attempt
from benchmark.recovery_diagnosis_v15 import retained_accuracy,native_diagnosis,write_report,measure,main as diagnose


class RecoveryExperimentV15Tests(unittest.TestCase):
    def test_interrupted_attempt_time_survives_resume_in_immutable_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);entry={'status':'working','attempts':[]};saved=[]
            def interrupted(*args,**kwargs):
                atomic_json(root/'stage4.attempt.json',{'status':'interrupted','elapsed_seconds':7.,'returncode':130})
                raise KeyboardInterrupt
            with patch('benchmark.recovery_experiment_v15.execute',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):run_attempt(['python'],root,entry,lambda:saved.append(True))
            self.assertEqual(entry['status'],'interrupted')
            self.assertEqual(read_json(root/'attempts/attempt_001.json')['elapsed_seconds'],7.)
            with patch('benchmark.recovery_experiment_v15.execute',return_value={'status':'complete','elapsed_seconds':2.,'returncode':0}):
                run_attempt(['python'],root,entry,lambda:saved.append(True))
            self.assertEqual(sum(a['elapsed_seconds'] for a in entry['attempts']),9.)
            self.assertEqual(read_json(root/'attempts/attempt_001.json')['elapsed_seconds'],7.)
            self.assertTrue((root/'attempts/attempt_002.json').exists());self.assertEqual(len(saved),2)

    def test_attempt_history_recovers_a_commit_before_progress_was_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);first={'status':'complete','elapsed_seconds':7.,'returncode':0,
                                   'existing_result_before_invocation':False}
            atomic_json(root/'attempts/attempt_001.json',first)
            entry={'status':'working','attempts':[]}
            with patch('benchmark.recovery_experiment_v15.execute',return_value={'status':'complete','elapsed_seconds':2.,'returncode':0}):
                run_attempt(['python'],root,entry,lambda:None)
            self.assertEqual(sum(a['elapsed_seconds'] for a in entry['attempts']),9.)
            self.assertEqual(read_json(root/'attempts/attempt_001.json'),first)

    def test_ablations_change_only_the_explicit_policy(self):
        base=read_json(WORKSPACE/'configs/joint_v14_native_repair_development.json')
        for arm,path in ARMS.items():
            value=read_json(path)
            self.assertEqual({k:v for k,v in value.items() if k!='native_repair_policy'},
                             {k:v for k,v in base.items() if k!='native_repair_policy'})

    def test_serving_command_reuses_evidence_without_solution_or_reference_seeds(self):
        request={'snapshot':'C:/frozen/v15','configs':{'combined':'C:/frozen/config.json'},'profile':'C:/frozen/profile.json'}
        case={'case_id':'case','bundle':'C:/sealed/stage3','capture_profile':'C:/input/capture.json',
              'baseline':'C:/old/solution','evaluation_reference':'C:/closed/truth.json'}
        argv=[str(v) for v in command_for(Path('C:/new'),request,case,'combined')]
        self.assertEqual(argv[argv.index('--bundle')+1],case['bundle'])
        self.assertNotIn(case['baseline'],argv);self.assertNotIn(case['evaluation_reference'],argv)
        self.assertNotIn('infer',argv);self.assertIn('solve',argv)

    def test_reference_measurement_is_unreachable_without_a_barrier(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('benchmark.recovery_diagnosis_v15.read_json',side_effect=[{},FileNotFoundError('prediction barrier')]), \
             patch('benchmark.recovery_diagnosis_v15.checked',return_value={}), \
             patch('benchmark.recovery_diagnosis_v15.measure') as measure:
            with self.assertRaises(FileNotFoundError):diagnose(['--run',folder])
            measure.assert_not_called()

    def test_offline_candidate_measurement_reports_a_better_pose_without_selecting_it(self):
        ident=np.eye(3);correct=np.eye(3);correct[0,2]=100.
        wrong=correct.copy();wrong[0,2]=500.
        reference={'pieces':{'a':{'native_to_reference':ident.tolist()},'b':{'native_to_reference':correct.tolist()}}}
        def record(name,h):
            return {'candidate_id':name,'native_to_component':{'a':ident.tolist(),'b':h.tolist()},
                'groups':{'a':'a','b':'a'},'selected_geometry_ids':[],'tree_geometry_ids':[],
                'occurrence_ids':{},'active_support':{},'source_score':1.,'disposition':'eligible','geometry':{'score':1.}}
        result={'layouts':[{'pieces':[{'piece_id':p,'native_probes':[[0,0],[10,10]]} for p in ('a','b')]}],
            'search':{'geometry_candidate_audit':{'selected_candidate_id':'chosen','candidates':[record('chosen',wrong),record('other',correct)]}}}
        before=digest(result);audit=retained_accuracy(result,reference)
        self.assertEqual(audit['complete_candidates'],2)
        self.assertEqual(audit['best_complete_by_reference_diagnostic']['candidate_id'],'other')
        self.assertEqual(digest(result),before)
        self.assertEqual(result['search']['geometry_candidate_audit']['selected_candidate_id'],'chosen')

    def test_protected_null_generation_and_separate_budget_flags_are_reportable(self):
        reference={'pieces':{'a':{},'b':{}},'contacts':[]};upstream={'stages':{'stage3':{'contacts':[]}}}
        native={'native_generation':None,'merge_searches':[],'budget_abstained':False}
        result={'search':{'native_repair':native}}
        measured=native_diagnosis(result,reference,upstream)
        self.assertEqual(measured['pairs_with_fitted_checks'],0)
        self.assertFalse(measured['generation_budget_abstained'])
        native['native_generation']={'native_checks':1,'maximum_native_checks':1,'budget_abstained':True,'untested_pair_count':0,
            'pairs':[{'pieces':['a','b'],'native_checks':1}]}
        measured=native_diagnosis(result,reference,upstream)
        self.assertTrue(measured['generation_budget_abstained'])
        self.assertFalse(measured['merge_budget_abstained']);self.assertFalse(measured['route_budget_abstained'])

    def test_report_handles_protected_baseline_and_failed_arm_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);baseline=root/'baseline';baseline.mkdir();Image.new('RGB',(20,20),'white').save(baseline/'preview.png')
            ref=root/'reference.png';Image.new('RGB',(20,20),'white').save(ref)
            report=root/'run/report';(report/'case').mkdir(parents=True)
            case={'case_id':'case','baseline':str(baseline),'reference_image':str(ref)}
            row={'case_id':'case','variants':{'baseline':{'status':'predicted','quality':{'component_sizes':[2],
                'common_gauge':{'whole_page_error_available':True,'non_anchor_probe_errors_reference_px':{'p95':2.}}},
                'native':{'pairs_with_fitted_checks':0,'possible_physical_pairs':1,'generation_budget_abstained':False},
                'retained':{'complete_candidates':1},'assessment':{'status':'supported'},
                'evaluation':{'false_neighbor_joins':[]},'reference_bound_diagnostic':{'reference_admitted':True},
                'stage4_subprocess_seconds':1.},'combined':{'status':'failed'}}}
            before=(baseline/'preview.png').read_bytes()
            write_report(root/'run',{'cases':[case]},[row])
            self.assertTrue((report/'index.html').is_file());self.assertTrue((report/'case/diagnosis.json').is_file())
            self.assertIn('failed',(report/'index.html').read_text())
            self.assertEqual((baseline/'preview.png').read_bytes(),before)

    def test_complete_measurement_path_replays_native_evidence_and_reference_bounds(self):
        from tests.test_native_repair_v15 import context
        from paper_reconstruction_stage4.native_repair_v15 import RepairContext
        from paper_reconstruction_stage4.repair_search import Node
        from paper_reconstruction_stage4.native_bridge import pair_options
        from paper_reconstruction_stage4.feasibility import Feasibility
        from paper_reconstruction_stage4.outputs import build_result,request_identity
        from paper_reconstruction_stage4.search.common import SearchResult
        from paper_reconstruction_stage4.page_geometry import state_record,fingerprint
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=context(folder)
            state=deepcopy(source);state.poses=full.poses;state.groups={pid:min(full.poses) for pid in full.poses}
            contacts=[]
            for i,(a,b) in enumerate((('piece_00','piece_02'),('piece_01','piece_03'))):
                arcs,metrics=pair_options(p,a,b,state.poses,c,[])[0]
                contacts.append({'pieces':[a,b],'arcs':arcs,'metrics':metrics,'role':'merge' if i==0 else 'cycle',
                    'proposal_id':str(i),'proposed_transform_b_to_a':(np.linalg.inv(state.poses[a])@state.poses[b]).tolist(),
                    'source_groups':source.components(),'merged_groups':source.components()})
            candidate,verdict=ctx.finalize(source,Node(state,contacts),Feasibility(p,c))
            self.assertIsNotNone(candidate,verdict)
            search={**ctx.report(),'final_feasibility':[{'valid':True}],
                    'refinement':[{'attempted':False}],'constraint_profile':p.constraint_profile}
            search['geometry_candidate_audit']={'selected_candidate_id':fingerprint(candidate),
                'candidates':[{'candidate_id':fingerprint(candidate),**state_record(candidate),
                               'disposition':'eligible','geometry':{'score':1.}}]}
            result=build_result(p,c,SearchResult([candidate],search,'exhausted_candidates'),request_identity(p,c))
            reference={'schema_version':'stage4.evaluation-reference.v1','pieces':{
                pid:{'native_to_reference':h.tolist()} for pid,h in full.poses.items()},'contacts':[],
                'reference_size_hw':[224,224]}
            case={'bundle':'unused','bundle_sha256':p.manifest['manifest_sha256'],
                  'capture_profile':'unused','constraint_binding':p.constraint_profile}
            request={'implementation':result['implementation'],'profile':'unused'}
            with patch('benchmark.recovery_diagnosis_v15.read_result',return_value=result), \
                 patch('benchmark.recovery_diagnosis_v15.read_json',return_value={'layouts':[{'status':'supported'}]}), \
                 patch('paper_reconstruction_stage4.inputs.load_problem',return_value=p), \
                 patch('paper_reconstruction_stage4.constraints.bind_profile'):
                value,row=measure(case,Path(folder)/'stage4',request,reference,
                                  {'stages':{'stage3':{'contacts':[]}}},'combined')
            self.assertEqual(value,result)
            self.assertTrue(row['independent_native_support_verification']['valid'])
            self.assertTrue(row['reference_bound_diagnostic']['reference_admitted'])
            self.assertEqual(row['quality']['component_sizes'],[4])


if __name__=='__main__':unittest.main()
