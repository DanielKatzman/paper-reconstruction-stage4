from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_stage4.config import SolverConfig,GEOMETRY_SELECTION
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.page_geometry import GeometryContext,GeometryBank,fingerprint
from paper_reconstruction_stage4.model import State,singleton_state
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import serialize_layout,canonical
from paper_reconstruction_stage4.assessment import recovery_request,validate_recovery
from paper_reconstruction_stage4.solver import solve
from tests.test_page_v10 import fixture,ROOT
from tests.fixtures import translation


def upgraded(folder,rows=2,cols=2,small=True):
    p,c,old,state,capture=fixture(folder,rows,cols)
    kw=dict(page_selection_policy=GEOMETRY_SELECTION,joint_finalize_seconds=10.,joint_wall_seconds=30.)
    if small:kw.update(page_top_k=5,page_geometry_k=5,page_diversity_k=2,page_final_candidate_limit=24)
    c=replace(c,**kw).validate();context=make_page_context(p,c,old.support);p.page_context=context
    return p,c,context,state


def partition(state,edges,groups):
    owners={p:min(group) for group in groups for p in group};poses={}
    for group in groups:
        inv=np.linalg.inv(state.poses[min(group)])
        poses.update({p:inv@state.poses[p] for p in group})
    selected=tuple(e for e in state.selected if owners[edges[e].a]==owners[edges[e].b])
    return State(poses,owners,selected,selected,{e:state.assignments[e] for e in selected},state.score)


class GeometryTopKTests(unittest.TestCase):
    def test_legacy_configurations_do_not_gain_new_fields(self):
        import json
        for name in ('joint_v9_retained_development.json','joint_v10_page_development.json','joint_v10_page_closure_development.json'):
            raw=json.loads((ROOT/'configs'/name).read_text());self.assertEqual(SolverConfig.from_dict(raw).to_dict(),raw)
        with self.assertRaises(ValueError):SolverConfig(page_top_k=1).validate()
        with tempfile.TemporaryDirectory() as folder:
            _,c,_,_=upgraded(folder)
            with self.assertRaises(ValueError):replace(c,page_final_candidate_limit=4).validate()
            with self.assertRaises(ValueError):replace(c,page_component_area_slack=.5).validate()

    def test_geometry_score_and_hard_rejection_ignore_high_source_score(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,good=upgraded(folder,3,3)
            bad=replace(good,poses={**good.poses,'piece_04':good.poses['piece_05'].copy()},score=1000.)
            gf=ctx.features(good);bf=ctx.features(bad)
            self.assertEqual(gf['geometry']['status'],'page_like')
            self.assertEqual(bf['geometry']['status'],'rejected')
            self.assertIn('large_interior_hole',bf['geometry']['reason_codes'])
            self.assertGreater(gf['geometry']['score'],bf['geometry']['score'])
            class Scorer:
                def evaluate(self,s):return {'score':s.score}
            p.structural=Scorer()
            states,report=ctx.prioritize_finalists(p,c,[bad,good],Feasibility(p,c))
            self.assertEqual(fingerprint(states[0]),fingerprint(good))
            self.assertTrue(any(r['disposition']=='rejected_geometry' for r in ctx.candidate_audit['candidates']))

    def test_correct_page_below_old_top_50_survives_geometry_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,good=upgraded(folder,3,3,small=False)
            class Scorer:
                def evaluate(self,s):return {'score':np.float64(s.score)}
            p.structural=Scorer();bank=GeometryBank(ctx,time.perf_counter()+60.)
            # Higher-scoring overlapping/hollow layouts remain inside the same
            # outer rectangle. A bounding-box-only score cannot distinguish them.
            for i in range(150):
                poses={k:h.copy() for k,h in good.poses.items()}
                poses['piece_04']=translation(48+(i%15)*.9,2+(i//15)*1.1)
                bank.consider(replace(good,poses=poses,score=200.-i*.01),'misleading_solver_rank')
            good.score=-10.;bank.consider(good,'low_old_rank_valid_rectangle')
            self.assertTrue(any(fingerprint(s)==fingerprint(good) for s in bank.finalists()))
            kept=next(r for r in bank.report()['retained_candidates'] if r['candidate_id']==fingerprint(good))
            self.assertEqual(kept['old_score_rank_among_measured_completions'],151)
            self.assertLessEqual(len(bank.states),c.page_top_k+c.page_geometry_k+c.page_diversity_k)
            self.assertFalse(bank.report()['is_certified_global_k_best'])
            canonical(bank.report())
            self.assertGreaterEqual(bank.counts['completed_states_measured'],151)

    def test_partial_bound_does_not_require_rectangle_fill(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=upgraded(folder)
            edges={e.id:e for e in p.edges};groups=[['piece_00','piece_01','piece_02'],['piece_03']]
            good=partition(full,edges,groups)
            self.assertTrue(ctx.component_bounds(good)['valid'])
            self.assertEqual(ctx.features(good)['geometry']['status'],'partial')
            self.assertNotIn('unfilled_fraction',ctx.features(good))
            bad=replace(good,poses={**good.poses,'piece_02':translation(180,180)})
            self.assertFalse(ctx.component_bounds(bad)['valid'])
            self.assertIn('component_exceeds_page_area_budget',ctx.assess(bad)['reason_codes'])
            self.assertFalse(Feasibility(p,c).state(bad,final=True)['valid'])
            self.assertFalse(ctx.component_bounds(good)['independent_component_rectangles_summed'])

    def test_partial_weak_bridge_cost_and_reason_are_active(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state=upgraded(folder)
            edges={e.id:e for e in p.edges}
            state=partition(state,edges,[['piece_00','piece_01'],['piece_02','piece_03']])
            from paper_reconstruction_stage4.active_support import slice_arc
            eid=state.selected[0];oid=state.assignments[eid]
            ctx.support.arcs[oid]=tuple(slice_arc(p,a,0.,.04) for a in ctx.support.arcs[oid])
            f=ctx.features(state)
            self.assertLess(ctx.utility(state,f),state.score)
            self.assertIn('short_straight_ambiguous_attachment',ctx.assess(state)['reason_codes'])

    def test_rotation_translation_and_pose_diversity(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state=upgraded(folder)
            angle=.31;a,b=np.cos(angle),np.sin(angle);g=np.array([[a,-b,210.],[b,a,-88.],[0.,0.,1.]])
            rotated=replace(state,poses={pid:g@h for pid,h in state.poses.items()})
            self.assertAlmostEqual(ctx.component_bounds(state)['components'][0]['ratio_to_total_observed_area'],
                ctx.component_bounds(rotated)['components'][0]['ratio_to_total_observed_area'],places=6)
            self.assertEqual(ctx.pose_family(state),ctx.pose_family(rotated))
            self.assertAlmostEqual(ctx.features(state)['geometry']['score'],ctx.features(rotated)['geometry']['score'],delta=.01)

    def test_recovery_targets_disconnected_groups_and_isolated_piece(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,state=upgraded(folder,rows=2,cols=4)
            edges={e.id:e for e in p.edges}
            for groups in ([['piece_00','piece_01'],['piece_02','piece_03','piece_04','piece_05','piece_06','piece_07']],
                           [['piece_00','piece_01','piece_02','piece_03','piece_05','piece_06','piece_07'],['piece_04']]):
                partial=partition(state,edges,groups);assessment=ctx.assess(partial)
                layout=serialize_layout(p,partial,0,{'valid':True},{'attempted':False})
                request=recovery_request(p,c,{'layouts':[layout]},assessment,{})
                validate_recovery(request)
                self.assertEqual(request['schema_version'],'stage4.pose-recovery-request.v2')
                self.assertTrue(any(r['region_kind']=='connect_components' for r in request['regions']))
                self.assertLessEqual(len(request['regions']),6)
                if len(groups[-1])==1:
                    self.assertTrue(any(r['region_kind']=='uncovered_piece' and r['piece_ids']==['piece_04'] for r in request['regions']))
                self.assertNotIn('reference',canonical(request).decode())
                request['true_neighbor_pairs']=[['piece_00','piece_07']]
                with self.assertRaises(Exception):validate_recovery(request)

    def test_new_solver_audits_topk_without_reading_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_,_=upgraded(folder)
            original=Path.open
            def guarded(path,*a,**kw):
                if 'evaluation_only' in str(path) or 'reference_geometry' in str(path):raise AssertionError('truth read')
                return original(path,*a,**kw)
            with patch.object(Path,'open',guarded):result=solve(p,c)
            audit=result.diagnostics['geometry_candidate_audit']
            self.assertTrue(audit['all_retained_candidates_checked'])
            self.assertTrue(audit['candidates'])
            self.assertEqual(result.diagnostics['reconstruction_assessment']['geometry']['status'],'page_like')
            self.assertFalse(audit['evaluation_truth_opened'])

    def test_gallery_and_v2_package_are_sealed_and_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_,_=upgraded(folder)
            from paper_reconstruction_stage4.outputs import request_identity,build_result,output_session,publish,file_hash
            from paper_reconstruction_stage4.assessment import write_package,verify_package
            from benchmark.page_geometry_experiment_v11 import write_gallery
            result=solve(p,c);request=request_identity(p,c);value=build_result(p,c,result,request)
            out=Path(folder)/'stage4'
            with output_session(out,request) as work:publish(out,work,value,p,result.states)
            sha=file_hash(out/'result.json')
            package=write_package(p,c,out)
            self.assertEqual(write_package(p,c,out),package)
            self.assertEqual(verify_package(out.with_name('stage4.assessment')),package)
            self.assertEqual(file_hash(out/'result.json'),sha)
            gallery=write_gallery(Path(folder)/'report',{'case_id':'toy','bundle':str(p.root)},value,{'configuration':c.to_dict()})
            self.assertTrue(Path(gallery['index']).is_file())
            self.assertGreater(gallery['records'],0)
            self.assertGreater(gallery['previews'],0)


if __name__=='__main__':unittest.main()
