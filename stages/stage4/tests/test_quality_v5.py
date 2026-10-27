import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark.common import read_json
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.geometry import normalize
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.outputs import digest
from paper_reconstruction_stage4.search.component_beam import pose_signature
from paper_reconstruction_stage4.solver import solve
from paper_reconstruction_stage4.structural import perpendicular_spacing,scale_coherence,jacobian_axes
from paper_reconstruction_stage4.support import SupportIndex
from tests.fixtures import build_fixture,translation

ROOT=Path(__file__).resolve().parents[1]


class QualityV5Tests(unittest.TestCase):
    def config(self,**kw):
        v=dict(solver="joint_beam_v3",objective_version="observable-joint.v5-scale",max_seconds=10.,joint_wall_seconds=20.,
               joint_finalize_seconds=2.,beam_width=16,portfolio_size=3)
        v.update(kw);return SolverConfig(**v).validate()

    def test_old_control_configurations_keep_their_exact_normalized_shape(self):
        for path in (ROOT/"controls/stage4_joint_v4_cached_v4/configs").glob("*.json"):
            old=read_json(path)
            # The frozen results contain the complete historical normalization.
            from importlib.util import spec_from_file_location,module_from_spec
            spec=spec_from_file_location("_legacy_quality_config",ROOT/"controls/stage4_joint_v4_cached_v4/src/paper_reconstruction_stage4/config.py")
            module=module_from_spec(spec)
            import sys
            sys.modules[spec.name]=module;spec.loader.exec_module(module)
            self.assertEqual(SolverConfig.from_dict(old).to_dict(),module.SolverConfig.from_dict(old).to_dict())
        with self.assertRaises(ValueError):SolverConfig(scale_coherence_weight=.5).validate()

    def test_perpendicular_spacing_tracks_parallel_lines_under_shear_and_rotation(self):
        for angle in (0.,.4,1.2):
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            matrix=rotation@np.array([[2.,4.],[0.,3.]])
            self.assertAlmostEqual(perpendicular_spacing(matrix[:,0],matrix[:,1]),3.)
            self.assertGreater(np.linalg.norm(matrix[:,1]),perpendicular_spacing(matrix[:,0],matrix[:,1]))
        h=np.array([[1.2,.3,17.],[.1,.9,-2.],[.0002,-.0001,1.]])
        u,v=jacobian_axes(h,(100,140),.2)
        self.assertGreater(perpendicular_spacing(u,v),0.)
        with self.assertRaises(ValueError):perpendicular_spacing(np.zeros(2),np.ones(2))

    def test_soft_pitch_consensus_detects_octave_outliers_and_keeps_component_gauge_free(self):
        c=self.config();same=[(str(i),np.log(18.),.9) for i in range(6)]
        mixed=[(str(i),np.log(9. if i<3 else 18.),.9) for i in range(6)]
        self.assertEqual(scale_coherence(same,c)[0],0.)
        cost,report=scale_coherence(mixed,c)
        self.assertGreater(cost,.4);self.assertLessEqual(cost,1.)
        self.assertFalse(report["free_octave_folding"])
        shifted=[(p,x+3.,w) for p,x,w in mixed]
        self.assertAlmostEqual(scale_coherence(shifted,c)[0],cost)
        heading=[*same,("heading",np.log(36.),.1)]
        self.assertEqual(scale_coherence(heading,c)[0],0.)
        self.assertEqual(scale_coherence(same[:2],c)[0],0.)

    def test_new_scale_term_can_separate_a_stretched_page_with_free_v4_octave_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"input";build_fixture(root,rows=1,cols=3,alternatives=False)
            c=self.config(structural_angle_weight=0.,structural_scale_weight=0.,structural_compactness_weight=0.,
                          structural_boundary_weight=0.,structural_appearance_weight=0.,structural_corroboration_weight=0.)
            p=load_problem(root,c);s=SupportIndex(p,c);scorer=JointScorer(p,c,s)
            for profile in scorer.profiles.values():profile.update(line_pitch_native_px=18.,pitch_confidence=.9,axis_confidence=0.)
            ids=sorted(p.pieces);groups={pid:ids[0] for pid in ids}
            coherent=State({pid:translation(i*27) for i,pid in enumerate(ids)},groups)
            distorted=copy.deepcopy(coherent);distorted.poses[ids[1]]=translation(27)@np.diag([.5,.5,1.])
            self.assertGreater(scorer.evaluate(coherent)["score"],scorer.evaluate(distorted)["score"])
            self.assertIn("scale_coherence_components",scorer.evaluate(distorted))

    def test_component_beam_recovers_a_loop_supported_backbone_without_reference_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"input";build_fixture(root,rows=2,cols=2,alternatives=False)
            c=self.config(objective_version="observable-joint.v4",joint_search_policy="component-beam.v1",
                structural_angle_weight=0.,structural_scale_weight=0.,structural_compactness_weight=0.,
                structural_boundary_weight=0.,structural_appearance_weight=0.,structural_corroboration_weight=.8,
                structural_underconstrained_cost=0.,max_proposals=1000)
            p=load_problem(root,c)
            ab=next(e for e in p.edges if (e.a,e.b)==('piece_00','piece_01'))
            ac=next(e for e in p.edges if (e.a,e.b)==('piece_00','piece_02'))
            bd=next(e for e in p.edges if (e.a,e.b)==('piece_01','piece_03'))
            cd=next(e for e in p.edges if (e.a,e.b)==('piece_02','piece_03'))
            ab.weight,ac.weight,bd.weight,cd.weight=.4,.7,.8,.9
            wrong=copy.deepcopy(bd);wrong.id,wrong.b,wrong.weight,wrong.transform=digest(['false-left']), 'piece_02',.95,translation(-27)
            o=wrong.occurrences[0];o['occurrence_id']=digest(['false-left-occurrence'])
            o['query_key'],o['target_key']='piece_01/left','piece_02/right'
            o['pose']['alignment']['transform']=wrong.transform.tolist();p.edges.append(wrong)
            result=solve(p,c);best=result.states[0]
            self.assertEqual(len(best.components()),1)
            self.assertNotIn(wrong.id,best.selected)
            self.assertTrue(result.diagnostics['final_feasibility'][0]['valid'])
            self.assertFalse(result.diagnostics['evaluation_truth_opened'])
            self.assertLessEqual(result.diagnostics['proposals'],1000)
            h=np.array([[1.2,.2,100.],[.1,.8,-20.],[0.,0.,1.]])
            moved=replace(best,poses={pid:normalize(h@value) for pid,value in best.poses.items()})
            self.assertEqual(pose_signature(best,p),pose_signature(moved,p))
