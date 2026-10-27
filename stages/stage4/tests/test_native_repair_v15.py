from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from paper_reconstruction_stage4.config import SolverConfig,REPAIR_V15_POLICIES
from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
from paper_reconstruction_stage4.page import make_page_context
from paper_reconstruction_stage4.repair_search import extent_hypothesis,fits_extent,search
from paper_reconstruction_stage4.seam_poses import generate as previous_generate
from paper_reconstruction_stage4.seam_poses_v15 import generate
from paper_reconstruction_stage4.outputs import serialize_layout,build_result,request_identity,validate_result
from paper_reconstruction_stage4.search.common import SearchResult
from paper_reconstruction_stage4.feasibility import Feasibility
from tests.test_native_repair_v14 import upgraded,curved_pair,ROOT


def context(folder,policy='native-seam-repair.v2'):
    p,c,_,source,full=upgraded(folder)
    c=replace(c,native_repair_policy=policy).validate()
    ctx=make_page_context(p,c,ActiveSupportIndex(p,c));p.page_context=ctx;p.structural=ctx.scorer()
    return p,c,ctx,source,full


def supplied_modes(expected,critical):
    def prepare(data,a,b,peaks,rotation_cache=None):
        xb,nb,_,_=data[b];rows=[]
        for q in range(4):
            t=q*np.pi/2.;r=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
            hit=(a,b)==critical and q==3
            values=[(expected[:2,2].copy(),100.)] if hit else [(np.array([10000.,10000.]),1.)]
            # Many extra peaks would consume the legacy pair-major budget.
            values.extend((np.array([10000.+100*i,10000.]),.5) for i in range(1,4))
            rows.append({'quarter':q,'r':r,'rb':xb@r.T,'rn':nb@r.T,'general':values,'curved':[]})
        return rows
    return prepare


class NativeRecoveryV15Tests(unittest.TestCase):
    def test_legacy_configuration_and_explicit_ablation_selection(self):
        original=json.loads((ROOT/'configs/joint_v14_native_repair_development.json').read_text())
        self.assertEqual(SolverConfig.from_dict(original).to_dict(),original)
        for policy,(area,fair) in REPAIR_V15_POLICIES.items():
            with tempfile.TemporaryDirectory() as folder:
                p,c,ctx,_,_=context(folder,policy)
                self.assertEqual((ctx.area_bound,ctx.fair_coverage),(area,fair))
                self.assertEqual(c.repair_native_checks,64)
                with self.assertRaises(ValueError):replace(c,component_closure_policy='disabled.v1').validate()

    def test_nonspanning_valid_page_is_admitted_and_bad_area_still_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,source,full=context(folder)
            extent=extent_hypothesis(p)
            self.assertFalse(fits_extent(ctx,full,extent))
            self.assertTrue(ctx.component_fits(full))
            angle=.37;rotation=np.array([[np.cos(angle),-np.sin(angle),500.],[np.sin(angle),np.cos(angle),-30.],[0.,0.,1.]])
            turned=deepcopy(full);turned.poses={pid:rotation@h for pid,h in full.poses.items()}
            self.assertTrue(ctx.component_fits(turned))
            bad=deepcopy(full);bad.poses['piece_03'][0,2]+=800.
            self.assertFalse(ctx.component_fits(bad))
            proposal={'pieces':['piece_00','piece_02'],'score':1.,'proposal_id':'fixture',
                'transform_b_to_a':(np.linalg.inv(full.poses['piece_00'])@full.poses['piece_02']).tolist()}
            old,_=search(ctx,source,[proposal],time.perf_counter()+10.,maximum_checks=16,beam=4)
            new,audit=ctx.merge_search(source,[proposal],time.perf_counter()+10.,16)
            self.assertFalse(old);self.assertTrue(new)
            self.assertFalse(audit['provisional_extent_enforced'])
            candidate,verdict=ctx.finalize(source,new[0][0],Feasibility(p,c))
            self.assertIsNotNone(candidate,verdict)
            layout=serialize_layout(p,candidate,0,{'valid':True},{'attempted':False})
            self.assertTrue(verify_result_support(p,{'layouts':[layout]},c)['valid'])
            solved=SearchResult([candidate],{'final_feasibility':[{'valid':True}],'refinement':[{'attempted':False}],
                'constraint_profile':p.constraint_profile},'exhausted_candidates')
            value=build_result(p,c,solved,request_identity(p,c));validate_result(value)
            self.assertEqual(value['schema_version'],'stage4.assembly-result.v4')

    def test_budget_reaches_a_necessary_late_pair_instead_of_deepening_first_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,expected=curved_pair(folder)
            first,second=p.pieces['alpha'],p.pieces['beta']
            p.pieces={f'p{i}':replace(first if i%2==0 else second,id=f'p{i}',observation=f'o{i}') for i in range(6)}
            critical=('p4','p5');budget=len(list(combinations(p.pieces,2)))
            with patch('paper_reconstruction_stage4.seam_poses_v15.prepare_modes',side_effect=supplied_modes(expected,critical)):
                rows,audit=generate(p,c,time.perf_counter()+15.,maximum_checks=budget,
                    frontiers=[[[f'p{i}' for i in range(3)],[f'p{i}' for i in range(3,6)]]])
            self.assertLessEqual(audit['native_checks'],budget)
            self.assertEqual(audit['untested_pair_count'],0)
            self.assertTrue(all(v['fitted_pairs']==5 for v in audit['per_piece_coverage'].values()))
            self.assertTrue(any(tuple(r['pieces'])==critical and np.max(np.abs(np.asarray(r['transform_b_to_a'])-expected))<5 for r in rows))
            self.assertTrue(audit['budget_abstained']);self.assertIn('native_check_limit',audit['budget_reason_codes'])
            self.assertFalse(audit['all_configured_work_completed'])

    def test_uncapped_measurements_and_curved_alternatives_match_v14(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_=curved_pair(folder)
            old,old_audit=previous_generate(p,c,time.perf_counter()+15.,maximum_checks=128)
            new,audit=generate(p,c,time.perf_counter()+15.,maximum_checks=128)
            self.assertEqual(old,new)
            self.assertEqual(old_audit['native_checks'],audit['native_checks'])
            self.assertTrue(audit['all_configured_work_completed'])
            self.assertFalse(audit['budget_abstained'])
            self.assertEqual(audit['pairs'][0]['tested_rotations'],4)
            self.assertLessEqual(audit['rotation_buffers'],4)

    def test_expired_budget_and_unknown_capture_cannot_claim_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_=curved_pair(folder)
            rows,audit=generate(p,c,time.perf_counter()-1.,maximum_checks=8)
            self.assertFalse(rows);self.assertEqual(audit['native_checks'],0)
            self.assertEqual(audit['untested_pair_count'],1)
            self.assertEqual(audit['pairs'][0]['tested_rotations'],0)
            p.capture_metadata=None
            with self.assertRaises(ValueError):generate(p,c,time.perf_counter()+1.)

    def test_frontier_must_not_invent_or_duplicate_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,_=curved_pair(folder)
            for groups in ([['alpha'],['missing']],[['alpha'],['alpha','beta']]):
                with self.assertRaises(ValueError):generate(p,c,time.perf_counter()+1.,frontiers=[groups])


if __name__=='__main__':unittest.main()
