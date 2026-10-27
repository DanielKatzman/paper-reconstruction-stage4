from dataclasses import replace
from pathlib import Path
import copy
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_stage4.config import SolverConfig,AMBIGUITY_SELECTION
from paper_reconstruction_stage4.continuation import band_information
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.outputs import serialize_layout,canonical
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.assessment import recovery_request,validate_recovery
from paper_reconstruction_stage4.solver import solve
from tests.test_page_v10 import fixture,ROOT
from tests.test_geometry_topk_v11 import partition


BLANK={'appearance_information':0.,'appearance_evidence_status':'uninformative_for_placement','mismatch':0.}


def upgraded(folder,rows=1,cols=4):
    p,c,old,full,_=fixture(folder,rows,cols)
    c=replace(c,page_selection_policy=AMBIGUITY_SELECTION,page_top_k=5,page_geometry_k=5,
        page_diversity_k=2,page_final_candidate_limit=32,joint_finalize_seconds=10.,joint_wall_seconds=30.).validate()
    ctx=make_page_context(p,c,old.support);p.page_context=ctx
    return p,c,ctx,full


class AmbiguityTests(unittest.TestCase):
    def test_legacy_configs_and_new_envelope(self):
        for name in ('joint_v10_page_closure_development.json','joint_v11_geometry_topk_development.json'):
            raw=json.loads((ROOT/'configs'/name).read_text())
            self.assertEqual(raw,SolverConfig.from_dict(raw).to_dict())
        with self.assertRaises(ValueError):SolverConfig(ambiguity_risk_weight=.2).validate()
        with tempfile.TemporaryDirectory() as folder:
            _,c,_,_=upgraded(folder)
            for updates in ({'ambiguity_deletion_checks':65},{'ambiguity_measurement_limit':33},
                            {'ambiguity_partial_slots':33},{'page_final_candidate_limit':20}):
                with self.assertRaises(ValueError):replace(c,**updates).validate()

    def test_constant_colour_is_not_displacement_information(self):
        for colour in (0.,.5,1.):self.assertEqual(band_information(np.full((65,3),colour)),0.)
        self.assertGreater(band_information(np.tile(np.linspace(0.,1.,65)[:,None],(1,3))),.9)

    @patch('paper_reconstruction_stage4.continuation.Continuation.measurement',return_value=BLANK)
    def test_long_straight_partial_is_uncertain_without_hard_rejection(self,_):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder)
            state=partition(full,ctx.edges,[['piece_00','piece_01','piece_02'],['piece_03']])
            f=ctx.features(state);a=f['ambiguity']
            self.assertTrue(a['unresolved_geometry_ids'])
            self.assertTrue(any(j['native_length_over_sqrt_smaller_area']>.12 for j in f['contacts']['joins']))
            self.assertFalse(a['straight_seams_automatically_rejected'])
            self.assertGreater(a['risk_cost'],0.)
            self.assertLess(ctx.utility(state,f),state.score)
            self.assertEqual(ctx.assess(state)['geometry']['status'],'partial')
            self.assertIn('unresolved_straight_or_underconstrained_bridge',ctx.assess(state)['reason_codes'])
            self.assertTrue(Feasibility(p,c).state(state,final=True)['valid'])

    @patch('paper_reconstruction_stage4.continuation.Continuation.measurement',return_value=BLANK)
    def test_duplicate_pair_fit_does_not_resolve_slide(self,_):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder)
            state=partition(full,ctx.edges,[['piece_00','piece_01'],['piece_02'],['piece_03']])
            contacts=ctx.contacts(state);join=contacts['joins'][0]
            join['independent_witness_pairs']=[join['pieces']]
            self.assertTrue(ctx.ambiguity(state,contacts)['unresolved_geometry_ids'])
            ctx.ambiguity_cache.clear();join['independent_witness_pairs']=[['piece_00','piece_02']]
            # Native contact verification is responsible for producing this
            # distinct cut witness; the ambiguity layer must preserve it.
            self.assertFalse(ctx.ambiguity(state,contacts)['unresolved_geometry_ids'])

    def test_selected_cycle_protects_short_supported_contact(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state=upgraded(folder,2,2)
            from paper_reconstruction_stage4.active_support import slice_arc
            oid=state.assignments[state.selected[0]]
            ctx.support.arcs[oid]=tuple(slice_arc(p,a,0.,.04) for a in ctx.support.arcs[oid])
            self.assertFalse(ctx.features(state)['ambiguity']['unresolved_geometry_ids'])
            self.assertEqual(ctx.assess(state)['status'],'supported')

    @patch('paper_reconstruction_stage4.continuation.Continuation.measurement',return_value=BLANK)
    def test_one_and_two_join_deletions_preserve_rigid_native_poses(self,_):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder)
            state=partition(full,ctx.edges,[['piece_00','piece_01','piece_02'],['piece_03']])
            states=ctx.augment_finalists([state],time.perf_counter()+20.)
            self.assertTrue(any(len(r['removed_geometry_ids'])==2 for r in ctx.deletion_audit['checks']))
            self.assertLessEqual(len(ctx.deletion_audit['checks']),c.ambiguity_deletion_checks)
            for candidate in states:
                self.assertTrue(Feasibility(p,c).state(candidate,final=True)['valid'])
                for group in candidate.components():
                    for a in group:
                        for b in group:
                            np.testing.assert_allclose(np.linalg.inv(candidate.poses[a])@candidate.poses[b],
                                np.linalg.inv(state.poses[a])@state.poses[b],atol=1e-8)

    @patch('paper_reconstruction_stage4.continuation.Continuation.measurement',return_value=BLANK)
    def test_v3_reserves_occupied_and_connection_regions_and_discloses_pairs(self,_):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder,2,4)
            state=partition(full,ctx.edges,[['piece_00','piece_01'],['piece_02','piece_03','piece_04','piece_05','piece_06','piece_07']])
            a=ctx.assess(state);layout=serialize_layout(p,state,0,{'valid':True},{'attempted':False})
            r=recovery_request(p,c,{'layouts':[layout]},a,{})
            validate_recovery(r);self.assertEqual(r['schema_version'],'stage4.pose-recovery-request.v3')
            kinds={v['region_kind'] for v in r['regions']}
            self.assertIn('check_join',kinds);self.assertIn('connect_components',kinds)
            cross=next(v for v in r['regions'] if v['region_kind']=='connect_components')
            self.assertEqual(len(cross['candidate_piece_pairs']),12)
            self.assertTrue(cross['pair_possibilities_exceed_budget'])
            occupied=next(v for v in r['regions'] if v['region_kind']=='check_join')
            self.assertTrue(occupied['occupied_spans_may_be_reconsidered']);self.assertTrue(occupied['native_spans'])
            self.assertNotIn('reference',canonical(r).decode())
            bad=copy.deepcopy(r);bad['true_neighbor_pairs']=[]
            with self.assertRaises(Exception):validate_recovery(bad)
            bad=copy.deepcopy(r);bad['regions'][0]['pair_possibilities_exceed_budget']=not bad['regions'][0]['pair_possibilities_exceed_budget']
            with self.assertRaises(ValueError):validate_recovery(bad)

    def test_new_solver_does_not_read_truth_and_checks_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder,2,2)
            original=Path.open
            def guarded(path,*args,**kwargs):
                if 'reference_geometry' in str(path) or 'evaluation_only' in str(path):raise AssertionError('truth read')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):result=solve(p,c)
            self.assertTrue(result.diagnostics['geometry_candidate_audit']['all_retained_candidates_checked'])
            self.assertEqual(result.diagnostics['reconstruction_assessment']['status'],'supported')
            self.assertIn('ambiguity_policy',result.diagnostics['page_selection']['features'])

    def test_v3_package_and_gallery_are_sealed_and_resumable(self):
        from paper_reconstruction_stage4.outputs import request_identity,build_result,output_session,publish,file_hash
        from paper_reconstruction_stage4.assessment import write_package,verify_package
        from benchmark.ambiguity_experiment_v12 import write_gallery
        with tempfile.TemporaryDirectory() as folder:
            p,c,_,_=upgraded(folder,2,2)
            solved=solve(p,c);request=request_identity(p,c);value=build_result(p,c,solved,request)
            out=Path(folder)/'stage4'
            with output_session(out,request) as work:publish(out,work,value,p,solved.states)
            sha=file_hash(out/'result.json');package=write_package(p,c,out)
            self.assertEqual(write_package(p,c,out),package)
            self.assertEqual(verify_package(out.with_name('stage4.assessment')),package)
            self.assertEqual(file_hash(out/'result.json'),sha)
            recovery=json.loads((out.with_name('stage4.assessment')/'pose_recovery_request.json').read_text())
            self.assertEqual(recovery['schema_version'],'stage4.pose-recovery-request.v3')
            gallery=write_gallery(Path(folder)/'report',{'case_id':'toy','bundle':str(p.root)},value,{'configuration':c.to_dict()})
            records=json.loads(Path(gallery['candidates_json']).read_text())['candidates']
            self.assertEqual(records[0]['candidate_id'],value['search']['geometry_candidate_audit']['selected_candidate_id'])

    @patch('paper_reconstruction_stage4.continuation.Continuation.measurement',return_value=BLANK)
    def test_deletions_respect_remaining_global_proposals(self,_):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder)
            partial=partition(full,ctx.edges,[['piece_00','piece_01','piece_02'],['piece_03']])
            ctx.finalization_proposal_allowance=0
            self.assertEqual(len(ctx.augment_finalists([partial],time.perf_counter()+20.)),1)
            self.assertFalse(ctx.deletion_audit['checks'])


if __name__=='__main__':unittest.main()
