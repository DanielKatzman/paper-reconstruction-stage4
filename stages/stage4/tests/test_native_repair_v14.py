from dataclasses import replace
from pathlib import Path
import tempfile,time,json,copy
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from paper_reconstruction_stage4.config import SolverConfig,REPAIR_POLICY
from paper_reconstruction_stage4.model import Piece,Problem
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
from paper_reconstruction_stage4.native_bridge import pair_options,install_result_bridges
from paper_reconstruction_stage4.native_bridge_v2 import validate_record
from paper_reconstruction_stage4.seam_poses import generate
from paper_reconstruction_stage4.repair_search import Node,search
from paper_reconstruction_stage4.search.execution import clone
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.outputs import digest,serialize_layout
from tests.test_component_closure_v13 import setup,ROOT


def upgraded(folder):
    p,c,_,source,full=setup(folder)
    c=replace(c,native_repair_policy=REPAIR_POLICY,repair_seconds=10.,repair_native_checks=64,
              repair_merge_checks=100,joint_finalize_seconds=30.,joint_wall_seconds=45.).validate()
    p.capture_metadata={'rotation_family':'quarter_turn','capture_profile_sha256':p.constraint_profile['capture_profile_sha256']}
    p.structural=None;ctx=make_page_context(p,c,ActiveSupportIndex(p,c));p.page_context=ctx;p.structural=ctx.scorer()
    return p,c,ctx,source,full


def polygon_contour(points):
    xy=np.asarray(points,float);t=np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0)
    t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
    return {'points_xy':xy,'inward_normal_xy':np.column_stack((-t[:,1],t[:,0]))}


def curved_pair(folder):
    height,width=200,240;y=np.arange(height);cut=110.+18*np.sin(y/22.)+9*np.sin(y/11.)
    a=[(x,0.) for x in np.arange(0,cut[0],2.)]+[(cut[j],float(j)) for j in range(height)]+[(x,height-1.) for x in np.arange(cut[-1]-1,0,-2.)]+[(0.,float(j)) for j in range(height-1,0,-1)]
    b=[(x,0.) for x in np.arange(cut[0]+1,width,2.)]+[(width-1.,float(j)) for j in range(height)]+[(x,height-1.) for x in np.arange(width-2,cut[-1]+1,-2.)]+[(cut[j]+1,float(j)) for j in range(height-1,-1,-1)]
    ca,cb=polygon_contour(a),polygon_contour(b)
    yy,xx=np.indices((height,width));ma=xx<=cut[:,None];mb=~ma
    rotation=np.array([[0.,-1.],[1.,0.]])
    cb['points_xy']=cb['points_xy']@rotation.T+[height-1,0]
    cb['inward_normal_xy']=cb['inward_normal_xy']@rotation.T
    mb=np.rot90(mb,-1).copy();pieces={}
    for pid,mask,contour in (('alpha',ma,ca),('beta',mb,cb)):
        image=Path(folder)/(pid+'.png');Image.new('RGB',(mask.shape[1],mask.shape[0]),'white').save(image)
        pieces[pid]=Piece(pid,pid,image,mask,contour)
    p=Problem(pieces,[],{}, {'manifest_sha256':'a'*64}, {},Path(folder),capture_metadata={'rotation_family':'quarter_turn','capture_profile_sha256':'b'*64})
    expected=np.array([[0.,1.,0.],[-1.,0.,height-1.],[0.,0.,1.]])
    return p,SolverConfig(),expected


class NativeRepairTests(unittest.TestCase):
    def test_legacy_normalization_and_repair_envelope(self):
        for name in ('joint_v13_component_closure_development.json','joint_v12_ambiguity_development.json'):
            raw=json.loads((ROOT/'configs'/name).read_text());self.assertEqual(raw,SolverConfig.from_dict(raw).to_dict())
        with self.assertRaises(ValueError):SolverConfig(native_repair_policy=REPAIR_POLICY).validate()
        with self.assertRaises(ValueError):SolverConfig(repair_beam=40).validate()
        with tempfile.TemporaryDirectory() as folder:
            _,c,_,_,_=upgraded(folder)
            for updates in ({'repair_source_limit':5},{'repair_native_checks':1025},{'repair_seconds':91.},{'repair_merge_checks':10001}):
                with self.assertRaises(ValueError):replace(c,**updates).validate()

    def test_native_alignment_recovers_an_unprovided_rotated_curved_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,expected=curved_pair(folder)
            original=Path.open
            def guarded(path,*args,**kwargs):
                if 'evaluation_only' in str(path) or 'reference_geometry' in str(path):raise AssertionError('truth access')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):poses,audit=generate(p,c,time.perf_counter()+15.,maximum_checks=32)
            self.assertTrue(poses);self.assertEqual(audit['pair_rotations'],4)
            self.assertLessEqual(audit['native_checks'],32)
            self.assertTrue(any(np.max(np.abs(np.asarray(v['transform_b_to_a'])-expected))<5 for v in poses))
            self.assertTrue(any(v['conditioned'] for v in poses))
            p.capture_metadata=None
            with self.assertRaises(ValueError):generate(p,c,time.perf_counter()+1.)

    def test_native_budget_and_rotation_disclosure(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_=curved_pair(folder)
            _,audit=generate(p,c,time.perf_counter()+5.,maximum_checks=1)
            self.assertEqual(audit['native_checks'],1);self.assertTrue(audit['budget_abstained'])
            self.assertFalse(audit['pairs'][0]['all_configured_work_completed'])

    def test_merge_budget_abstains_without_changing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=upgraded(folder)
            before={p:h.copy() for p,h in source.poses.items()}
            proposal={'pieces':['piece_00','piece_02'],'score':1.,'proposal_id':'probe',
                'transform_b_to_a':(np.linalg.inv(full.poses['piece_00'])@full.poses['piece_02']).tolist()}
            _,audit=search(ctx,source,[proposal],time.perf_counter()-1.,maximum_checks=1,beam=4)
            self.assertEqual(audit['merge_checks'],0);self.assertTrue(audit['budget_abstained'])
            _,audit=search(ctx,source,[proposal],time.perf_counter()+5.,maximum_checks=1,beam=4)
            self.assertLessEqual(audit['merge_checks'],1)
            for pid,h in before.items():np.testing.assert_array_equal(h,source.poses[pid])

    def test_sources_accept_many_components_and_keep_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,_=upgraded(folder)
            sources,_=ctx.sources([source])
            self.assertTrue(sources)
            self.assertTrue(any(len(s.components())>2 for s,_,_ in sources))
            self.assertEqual(len(source.components()),2)
            self.assertEqual(max(len(removed) for _,removed,_ in sources),2)

    def test_v4_native_graph_provenance_replay_and_straight_cycle_requirement(self):
        from paper_reconstruction_stage4.outputs import request_identity,build_result,output_session,publish,read_result
        from paper_reconstruction_stage4.assessment import write_package,verify_package
        from paper_reconstruction_stage4.search.common import SearchResult
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=upgraded(folder)
            state=clone(source);state.poses=full.poses;state.groups={p:min(full.poses) for p in full.poses}
            contacts=[]
            for i,(a,b) in enumerate((('piece_00','piece_02'),('piece_01','piece_03'))):
                arcs,metrics=pair_options(p,a,b,state.poses,c,[])[0]
                h=np.linalg.inv(state.poses[a])@state.poses[b]
                contacts.append({'pieces':[a,b],'arcs':arcs,'metrics':metrics,'role':'merge' if i==0 else 'cycle',
                    'proposal_id':digest([a,b]),'proposed_transform_b_to_a':h.tolist(),
                    'source_groups':source.components(),'merged_groups':source.components()})
            candidate,verdict=ctx.finalize(source,Node(state,contacts),Feasibility(p,c))
            self.assertIsNotNone(candidate,verdict)
            layout=serialize_layout(p,candidate,0,{'valid':True},{'attempted':False})
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])
            bad=copy.deepcopy(layout);bad['selected_joins']=[j for j in bad['selected_joins'] if j['piece_a']!='piece_01' or j['piece_b']!='piece_03']
            with self.assertRaisesRegex(ValueError,'corroboration'):verify_result_support(p,{'layouts':[bad]},c)
            record=copy.deepcopy(next(j['native_bridge'] for j in layout['selected_joins'] if 'native_bridge' in j))
            record['merged_groups'][0]=['piece_00'];record['bridge_id']=digest({k:v for k,v in record.items() if k!='bridge_id'})
            with self.assertRaises(ValueError):validate_record(record)
            solved=SearchResult([candidate],{'final_feasibility':[{'valid':True}],'refinement':[{'attempted':False}],
                'constraint_profile':p.constraint_profile},'exhausted_candidates')
            request=request_identity(p,c);value=build_result(p,c,solved,request);self.assertEqual(value['schema_version'],'stage4.assembly-result.v4')
            out=Path(folder)/'result'
            with output_session(out,request) as work:publish(out,work,value,p,[candidate])
            self.assertEqual(read_result(out)['run_id'],request['run_id'])
            fresh=replace(p,edges=[e for e in p.edges if 'native_bridge' not in e.raw],structural=None,page_context=None)
            receipt=write_package(fresh,c,out);self.assertEqual(verify_package(out.with_name('result.assessment')),receipt)

    def test_replay_keeps_distinct_native_intervals_for_the_same_pair(self):
        from paper_reconstruction_stage4.native_bridge_v2 import make_record
        from paper_reconstruction_stage4.native_bridge import edge_from_record,arc_value
        from paper_reconstruction_stage4.active_support import slice_arc
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=upgraded(folder)
            arcs,metrics=pair_options(p,'piece_00','piece_02',full.poses,c,[])[0]
            contact={'pieces':['piece_00','piece_02'],'arcs':arcs,'metrics':metrics,'role':'merge',
                'proposal_id':'first','proposed_transform_b_to_a':(np.linalg.inv(full.poses['piece_00'])@full.poses['piece_02']).tolist(),
                'source_groups':source.components(),'merged_groups':source.components()}
            first=make_record(p,full,contact,'source');edge,_=edge_from_record(p,first,c)
            second=copy.deepcopy(first);second['arcs']=[arc_value(slice_arc(p,a,.1,.9)) for a in arcs]
            second['placement_proposal_id']='second';second['bridge_id']=digest({k:v for k,v in second.items() if k!='bridge_id'})
            other,_=edge_from_record(p,second,c);p.edges.extend([edge,other])
            index=ActiveSupportIndex(p,c)
            first_arc=index.arcs[edge.occurrences[0]['occurrence_id']][0]
            other_arc=index.arcs[other.occurrences[0]['occurrence_id']][0]
            self.assertGreater(first_arc.length,other_arc.length)

    def test_unknown_capture_abstains_and_complete_page_is_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=upgraded(folder)
            # With the normal V13 route disabled by a pending deadline, an
            # unknown rotation model must not enter quarter-turn generation.
            p.capture_metadata={'rotation_family':'unknown','capture_profile_sha256':p.constraint_profile['capture_profile_sha256']}
            ctx.finalization_proposal_allowance=100
            ctx.augment_finalists([source],time.perf_counter()+30.)
            self.assertTrue(ctx.repair_audit['protected_supported_completion'])
            other=make_page_context(p,c,ctx.support);other.finalization_proposal_allowance=100
            from paper_reconstruction_stage4.component_closure import ClosureContext
            with patch.object(ClosureContext,'augment_finalists',return_value=[source]):
                other.augment_finalists([source],time.perf_counter()+30.)
            self.assertIn('unavailable_reason',other.repair_audit)


if __name__=='__main__':unittest.main()
