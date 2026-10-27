from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.constraints import bind_profile, rigid_check
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import State, singleton_state
from paper_reconstruction_stage4.outputs import atomic_json, digest, file_hash, request_identity, build_result, output_session, publish, read_result
from paper_reconstruction_stage4.page import PageContext, PageBank, LIMITS
from paper_reconstruction_stage4.active_support import ActiveSupportIndex
from paper_reconstruction_stage4.assessment import write_package, verify_package, validate_recovery
from paper_reconstruction_stage4.solver import solve
from tests.fixtures import build_fixture, translation

ROOT = Path(__file__).resolve().parents[1]


def fixture(folder, rows=2, cols=2):
    root = Path(folder); bundle = root/'input'; build_fixture(bundle, rows=rows, cols=cols)
    capture = root/'capture.json'
    atomic_json(capture, {'schema_version': 'stage1.capture-profile.v1', 'motion_model': 'rotation_translation',
        'relative_piece_scale': 1., 'shared_native_pixel_scale': True, 'per_piece_resize': False,
        'perspective': False, 'stage1_masks': 'generator_exact_bypass', 'mask_perturbation': False,
        'capture_canvas_hw': [9999,9999]})
    manifest = json.loads((bundle/'bundle.json').read_text())
    manifest['provenance']['inputs'] = [{'sha256': file_hash(capture), 'size': capture.stat().st_size}]
    manifest['manifest_sha256'] = digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
    atomic_json(bundle/'bundle.json', manifest)
    c = json.loads((ROOT/'configs/joint_v10_page_development.json').read_text())
    c.update(max_proposals=100, max_seconds=10., joint_wall_seconds=25., joint_finalize_seconds=2.,
             joint_trials=3, beam_width=8, joint_retention_route_proposals=40)
    config = SolverConfig.from_dict(c); problem = load_problem(bundle, config)
    bind_profile(problem, config, ROOT/'configs/shared_scale_complete_rectangle_v1.json', capture)
    support = ActiveSupportIndex(problem, config); context = PageContext(problem, config, support)
    poses = {p: translation(27*(i%cols), 27*(i//cols)) for i,p in enumerate(sorted(problem.pieces))}
    selected = tuple(e.id for e in problem.edges)
    assignments = {e.id: e.occurrences[0]['occurrence_id'] for e in problem.edges}
    state = State(poses, {p:min(poses) for p in poses}, selected, selected, assignments, 1.)
    return problem, config, context, state, capture


class PagePolicyTests(unittest.TestCase):
    def test_rigid_contract_and_numerical_tolerance(self):
        angle=.327; c,s=np.cos(angle),np.sin(angle)
        h=np.array([[c,-s,34.],[s,c,-75.],[0.,0.,1.]])
        self.assertTrue(rigid_check(h)['valid'])
        self.assertTrue(rigid_check(h@h)['valid'])
        for bad in (np.diag([1.01,1.01,1]), np.diag([-1,1,1]),
                    np.array([[1,.01,0],[0,1,0],[0,0,1]]), np.array([[1,0,0],[0,1,0],[.0001,0,1]])):
            self.assertFalse(rigid_check(bad)['valid'])
        noisy=h.copy(); noisy[0,0]+=1e-8
        self.assertTrue(rigid_check(noisy)['valid'])
        self.assertFalse(rigid_check(np.full((3,3),np.nan))['valid'])

    def test_profiles_are_explicit_and_legacy_normalization_unchanged(self):
        raw=json.loads((ROOT/'configs/joint_v9_retained_development.json').read_text())
        self.assertEqual(SolverConfig.from_dict(raw).to_dict(),raw)
        with tempfile.TemporaryDirectory() as folder:
            p,c,_,_,capture=fixture(folder)
            first=request_identity(p,c)['run_id']
            with self.assertRaises(ValueError):bind_profile(p,c,None,capture)
            with self.assertRaises(ValueError):bind_profile(p,replace(c,page_policy='disabled.v1'),ROOT/'configs/shared_scale_complete_rectangle_v1.json',capture)
            atomic_json(capture, {'shared_native_pixel_scale': True})
            with self.assertRaises(ValueError):bind_profile(p,c,ROOT/'configs/shared_scale_complete_rectangle_v1.json',capture)
            p.constraint_profile=None
            self.assertNotEqual(first,request_identity(p,c)['run_id'])
            with self.assertRaises(ValueError):solve(p,c)
        with self.assertRaises(ValueError):replace(c,refine=True,joint_refinement=True).validate()
        with self.assertRaises(ValueError):replace(c,recovery_region_limit=7).validate()

    def test_rotation_translation_and_identifier_invariance(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,context,state,_=fixture(folder)
            f=context.features(state)
            theta=.391; a,b=np.cos(theta),np.sin(theta)
            gauge=np.array([[a,-b,333.],[b,a,-172.],[0,0,1.]])
            rotated=replace(state, poses={pid:gauge@h for pid,h in reversed(list(state.poses.items()))})
            other=context.features(rotated)
            self.assertAlmostEqual(f['rectangle_area_px2'],other['rectangle_area_px2'],places=6)
            self.assertAlmostEqual(f['fill_fraction'],other['fill_fraction'],delta=.015)
            self.assertEqual(context.assess(state)['status'],'supported')
            self.assertEqual(context.assess(rotated)['status'],'supported')
            # Rename piece identifiers including graph/candidate/arc descriptors.
            from paper_reconstruction_stage4.support import Arc
            names={pid:'renamed_'+str(len(p.pieces)-i) for i,pid in enumerate(p.pieces)}
            support=context.support
            for arcs in support.arcs.values():
                for arc in arcs:arc.piece=names.get(arc.piece,arc.piece)
            for edge in p.edges:edge.a,edge.b=names[edge.a],names[edge.b]
            p.pieces={names[k]:v for k,v in reversed(list(p.pieces.items()))}
            renamed=replace(state,poses={names[k]:v for k,v in state.poses.items()},groups={names[k]:names[v] for k,v in state.groups.items()})
            renamed_context=PageContext(p,c,support)
            self.assertAlmostEqual(f['fill_fraction'],renamed_context.features(renamed)['fill_fraction'],places=6)
            self.assertEqual(renamed_context.assess(renamed)['status'],'supported')

    def test_outlier_hole_partial_and_weak_rectangle(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,context,state,_=fixture(folder,rows=1,cols=2)
            good=context.features(state)
            bad=replace(state,poses={**state.poses,'piece_01':translation(110)})
            self.assertGreater(context.features(bad)['unfilled_fraction'],.4)
            self.assertEqual(context.assess(bad)['status'],'unresolved')
            partial=singleton_state(p)
            self.assertNotIn('unfilled_fraction',context.features(partial))
            self.assertIn('incomplete_candidate',context.assess(partial)['reason_codes'])
            # Perfect rectangle but tiny straight support remains ambiguous.
            from paper_reconstruction_stage4.active_support import slice_arc
            oid=state.assignments[state.selected[0]]
            context.support.arcs[oid]=tuple(slice_arc(p,a,0.,.04) for a in context.support.arcs[oid])
            context.cache.clear()
            self.assertIn('short_straight_ambiguous_attachment',context.assess(state)['reason_codes'])
            self.assertGreater(good['fill_fraction'],.95)
        with tempfile.TemporaryDirectory() as folder:
            _,_,context,state,_=fixture(folder,rows=3,cols=3)
            # Move middle tile out: enclosed missing region plus a distant tile.
            hole=replace(state,poses={**state.poses,'piece_04':translation(200,200)})
            self.assertGreater(context.features(hole)['interior_hole_fraction'],0.)

    def test_short_contact_in_independent_loop_not_blanket_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p,_,context,state,_=fixture(folder)
            from paper_reconstruction_stage4.active_support import slice_arc
            oid=state.assignments[state.selected[0]]
            context.support.arcs[oid]=tuple(slice_arc(p,a,0.,.04) for a in context.support.arcs[oid])
            result=context.assess(state)
            self.assertTrue(any(j['short_straight'] for j in result['features']['contacts']['joins']))
            self.assertNotIn('short_straight_ambiguous_attachment',result['reason_codes'])
            self.assertEqual(result['status'],'supported')

    def test_independence_cache_bank_caps_and_old_check_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,context,state,_=fixture(folder)
            # Duplicate same support as a new hypothesis cannot add a vote.
            from paper_reconstruction_stage4.model import Edge
            e=p.edges[0]; p.edges.append(Edge('duplicate',e.a,e.b,e.transform,e.weight,e.occurrences,e.raw))
            contacts=context.contacts(state)
            self.assertFalse(contacts['independent_witnesses'])
            context.features(state);context.features(state)
            self.assertEqual(context.counts['cache_hits'],1)
            class Scorer:
                def evaluate(self,s):return {'score':s.score}
            p.structural=Scorer();bank=PageBank(context,time.perf_counter()+20.)
            for i in range(60):
                candidate=replace(state, poses={**state.poses,'piece_03':translation(27+i,27)},score=1.+i*.001)
                bank.consider(candidate,'test')
            self.assertLessEqual(len(bank.states),LIMITS['retained_page_states'])
            self.assertEqual(bank.counts['completed_states_measured'],60)
            from paper_reconstruction_stage4.search.incumbents import Incumbents
            search=type('Search',(),{'problem':p,'config':replace(c,joint_final_check_limit=1)})()
            p.page_context=context
            shared=Incumbents(search,time.perf_counter()+10);route=shared.route('test',time.perf_counter()+10);route.checks=1
            route.consider(state,'after_old_budget')
            self.assertEqual(shared.page_bank.counts['completed_states_measured'],1)

    def test_sealed_sidecar_resume_and_no_truth_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_,_,_=fixture(folder)
            result=solve(p,c);request=request_identity(p,c);out=Path(folder)/'result'
            value=build_result(p,c,result,request)
            with output_session(out,request) as work:publish(out,work,value,p,result.states)
            original=file_hash(out/'result.json')
            # No evaluation path may be opened during assessment or adoption.
            original_open=Path.open
            def guarded(path,*args,**kw):
                if 'evaluation_only' in str(path) or 'reference_geometry' in str(path):raise AssertionError('truth opened')
                return original_open(path,*args,**kw)
            with patch.object(Path,'open',guarded):
                receipt=write_package(p,c,out)
                with output_session(out,request,resume=True) as work:self.assertIsNone(work)
                self.assertEqual(write_package(p,c,out),receipt)
            self.assertEqual(file_hash(out/'result.json'),original)
            side=out.with_name('result.assessment')
            self.assertEqual(verify_package(side),receipt)
            feedback=json.loads((side/'pose_recovery_request.json').read_text())
            validate_recovery(feedback)
            feedback['reference_neighbor']='bad'
            with self.assertRaises(Exception):validate_recovery(feedback)
            with (side/'report.md').open('a') as stream:stream.write('tamper')
            with self.assertRaises(ValueError):verify_package(side)


if __name__=='__main__':unittest.main()
