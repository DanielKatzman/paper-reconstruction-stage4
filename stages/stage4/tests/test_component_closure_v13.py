from dataclasses import replace
from pathlib import Path
import copy
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from paper_reconstruction_stage4.config import SolverConfig,CLOSURE_POLICY
from paper_reconstruction_stage4.component_closure import placement_proposals
from paper_reconstruction_stage4.native_bridge import runs,pair_options,arc_from_indices,contour_arc,install_result_bridges
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import serialize_layout,validate_result,digest
from tests.test_ambiguity_v12 import upgraded
from tests.test_geometry_topk_v11 import partition
from tests.test_page_v10 import ROOT


def setup(folder):
    p,c,_,full=upgraded(folder,2,2)
    c=replace(c,component_closure_policy=CLOSURE_POLICY,closure_seconds=10.,
              joint_finalize_seconds=25.,joint_wall_seconds=40.).validate()
    # Larger physical seams than the tiny legacy toy; no reference data.
    for piece in p.pieces.values():
        piece.mask=np.zeros((128,128),bool);piece.mask[8:117,8:117]=True
        piece.contour['points_xy']*=4.
        with Image.open(piece.image_path) as image:scaled=image.resize((128,128))
        piece.image_path=Path(folder)/(piece.id+'.png');scaled.save(piece.image_path)
    for h in full.poses.values():h[:2,2]*=4.
    for edge in p.edges:edge.transform[:2,2]*=4.
    edges={e.id:e for e in p.edges}
    source=partition(full,edges,[['piece_00','piece_01'],['piece_02','piece_03']])
    # There is deliberately no supplied bridge pose in this input index.
    p.edges=[e for e in p.edges if e.id in source.selected]
    ctx=make_page_context(p,c,ActiveSupportIndex(p,c));p.page_context=ctx
    p.structural=ctx.scorer()
    return p,c,ctx,source,full


class ComponentClosureTests(unittest.TestCase):
    def test_legacy_normalization_and_profile_budget(self):
        for name in ('joint_v12_ambiguity_development.json','joint_v11_geometry_topk_development.json'):
            raw=json.loads((ROOT/'configs'/name).read_text());self.assertEqual(raw,SolverConfig.from_dict(raw).to_dict())
        with self.assertRaises(ValueError):SolverConfig(component_closure_policy=CLOSURE_POLICY).validate()
        with self.assertRaises(ValueError):SolverConfig(closure_source_limit=4).validate()
        with tempfile.TemporaryDirectory() as folder:
            p,c,_,_,_=setup(folder)
            with self.assertRaises(ValueError):replace(c,closure_proposal_limit=513).validate()
            p.constraint_profile=None
            with self.assertRaises(ValueError):make_page_context(p,c,ActiveSupportIndex(p,c))

    def test_circular_contiguous_runs_and_reject_disjoint_support(self):
        self.assertEqual(runs(np.array([1,1,0,0,1,1,1],bool)),[(4,8)])
        with tempfile.TemporaryDirectory() as folder:
            p,_,_,_,_=setup(folder)
            arc=arc_from_indices(p,'piece_00',90,125)
            self.assertEqual(len(arc.intervals),2)
            bad={'piece':'piece_00','intervals':[[0.,60.],[90.,150.]],'length':120.}
            with self.assertRaises(ValueError):contour_arc(p,bad)

    def test_float32_native_contour_wrap_uses_consistent_arc_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            p,_,_,_,_=setup(folder)
            xy=p.pieces['piece_00'].contour['points_xy'].astype(np.float32)
            xy[:,0]+=np.sin(np.arange(len(xy)))*.3
            p.pieces['piece_00'].contour['points_xy']=xy
            # Noninteger segment lengths expose float32 sum/cumsum disagreement
            # on the perimeter, which must not reject a valid wrapping seam.
            arc=arc_from_indices(p,'piece_00',90,125)
            rebuilt=contour_arc(p,{'piece':arc.piece,'intervals':arc.intervals,'length':arc.length})
            self.assertEqual(len(arc.intervals),2)
            np.testing.assert_array_equal(arc.points,rebuilt.points)

    def test_native_seam_checks_distance_normals_occupancy_and_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=setup(folder)
            good=pair_options(p,'piece_00','piece_02',full.poses,c,[])
            self.assertTrue(good)
            self.assertFalse(pair_options(p,'piece_00','piece_02',full.poses,c,list(good[0][0])))
            same=copy.deepcopy(full.poses);same['piece_02']=same['piece_00'].copy()
            self.assertFalse(pair_options(p,'piece_00','piece_02',same,c,[]))
            distant=copy.deepcopy(full.poses);distant['piece_02'][0,2]+=300.
            self.assertFalse(pair_options(p,'piece_00','piece_02',distant,c,[]))
            from paper_reconstruction_stage4.constraints import require_rigid
            for _,proposal,_ in placement_proposals(ctx,source):
                for h in proposal.poses.values():require_rigid(h)
                for group in source.components():
                    a,b=group
                    np.testing.assert_allclose(np.linalg.inv(proposal.poses[a])@proposal.poses[b],
                        np.linalg.inv(source.poses[a])@source.poses[b],atol=1e-8)

    def test_shape_cannot_create_bridge_without_independent_seams(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=setup(folder)
            with patch('paper_reconstruction_stage4.component_closure.pair_options',return_value=[]):
                candidate,verdict=ctx.verify_placement(source,full,'probe',Feasibility(p,c))
            self.assertIsNone(candidate);self.assertEqual(verdict['disposition'],'rejected_native_bridge')
            self.assertEqual(len(p.edges),len(source.selected))

    def test_observed_generation_creates_real_replayable_bridge_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,_=setup(folder)
            original=Path.open
            def guarded(path,*args,**kwargs):
                if 'evaluation_only' in str(path) or 'reference_geometry' in str(path):raise AssertionError('truth read')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):
                result,checks=ctx.prioritize_finalists(p,c,[source],Feasibility(p,c))
            self.assertTrue(result);best=result[0]
            self.assertEqual(len(best.components()),1)
            self.assertGreaterEqual(len(best.selected)-len(source.selected),2)
            self.assertEqual(len(best.tree),len(source.tree)+1)
            layout=serialize_layout(p,best,0,{'valid':True},{'attempted':False})
            bridges=[j for j in layout['selected_joins'] if 'native_bridge' in j]
            self.assertGreaterEqual(len(bridges),2)
            upstream=[e for e in p.edges if 'native_bridge' not in e.raw]
            p.edges=upstream
            install_result_bridges(p,{'layouts':[layout]},c)
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])
            bad=copy.deepcopy(layout);bad['selected_joins'][-1]['native_bridge']['arcs'][0]['length']+=10.
            with self.assertRaises(ValueError):verify_result_support(p,{'layouts':[bad]},c)
            from paper_reconstruction_stage4.search.common import SearchResult
            from paper_reconstruction_stage4.outputs import request_identity,build_result,output_session,publish,read_result
            from paper_reconstruction_stage4.assessment import write_package,verify_package
            solved=SearchResult([best],{'final_feasibility':[{'valid':True}],
                'refinement':[{'attempted':False}],'constraint_profile':p.constraint_profile},'exhausted_candidates')
            request=request_identity(p,c);value=build_result(p,c,solved,request)
            self.assertEqual(value['schema_version'],'stage4.assembly-result.v3')
            output=Path(folder)/'result'
            with output_session(output,request) as work:publish(output,work,value,p,[best])
            self.assertEqual(read_result(output)['run_id'],request['run_id'])
            fresh=replace(p,edges=list(upstream),page_context=None,structural=None)
            receipt=write_package(fresh,c,output)
            self.assertEqual(verify_package(output.with_name('result.assessment')),receipt)
            tampered=copy.deepcopy(value)
            joins=tampered['layouts'][0]['selected_joins']
            joins[:]=[j for j in joins if 'native_bridge' not in j or j is next(j for j in joins if 'native_bridge' in j)]
            with self.assertRaises(ValueError):validate_result(tampered)

    def test_global_proposal_allowance_and_time_abstention(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,_=setup(folder);ctx.finalization_proposal_allowance=2
            ctx.augment_finalists([source],time.perf_counter()+30.)
            self.assertLessEqual(ctx.closure_audit['generated_proposals']+len(ctx.deletion_audit['checks']),2)
            other=make_page_context(p,c,ctx.support)
            other.augment_finalists([source],time.perf_counter()-1.)
            self.assertEqual(other.closure_audit['generated_proposals'],0)
            self.assertTrue(other.closure_audit['budget_abstained'])


if __name__=='__main__':unittest.main()
