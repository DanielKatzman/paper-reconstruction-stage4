import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.native_graph_v21 import NativeFrontier
from paper_reconstruction_stage4.seam_poses_rigid import retain

from tests.test_capture_recovery import continuous
from tests.test_native_repair_v17 import native_proposals


def scalar_reference(rows):
    kept=[]
    for row in sorted(rows,key=lambda r:(not r['conditioned'],-r['score'],r['proposal_id'])):
        h=np.asarray(row['transform_b_to_a'])
        if any(np.max(np.abs(h[:2,:2]-np.asarray(old['transform_b_to_a'])[:2,:2]))<1e-6
               and np.linalg.norm(h[:2,2]-np.asarray(old['transform_b_to_a'])[:2,2])<4. for old in kept):continue
        kept.append(row)
    return kept


class NativeModeRetentionTests(unittest.TestCase):
    def test_vectorized_retention_preserves_scalar_ids_and_order_at_boundaries(self):
        rng=np.random.default_rng(804)
        rows=[]
        for i in range(450):
            angle=rng.choice([0.,1e-6,1.00001e-6,.17,np.pi/2])
            h=np.eye(3);h[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
            h[:2,2]=rng.choice([0.,3.999999,4.,4.000001,8.,12.],size=2)
            rows.append({'proposal_id':str(i),'conditioned':bool(i%3),
                'score':float(rng.integers(0,8)),'transform_b_to_a':h.tolist()})
        expected=[r['proposal_id'] for r in scalar_reference(rows)]
        self.assertEqual([r['proposal_id'] for r in retain(rows)],expected)
        self.assertEqual([r['proposal_id'] for r in retain(reversed(rows))],expected)

    def test_pair_cache_matches_full_rebuild_and_recomputes_only_changed_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,ctx,full=continuous(folder)
            basis=singleton_state(p);physical=Feasibility(p,c)
            fast=NativeFrontier(ctx,basis,physical,lambda *a:None,'fast')
            reference=NativeFrontier(ctx,basis,physical,lambda *a:None,'full')
            rows=native_proposals(p,c,full)
            for engine in (fast,reference):engine.add(rows)
            update=deepcopy(rows[0]);update['proposal_id']='higher-ranked-nearby-mode'
            update['score']=rows[0]['score']+10.;update['transform_b_to_a'][0][2]+=1.
            with patch('paper_reconstruction_stage4.seam_poses_rigid.retain',wraps=retain) as measured:
                fast.add([update]);self.assertEqual(measured.call_count,1)
            reference.retained_by_pair.clear()
            reference.add([update])
            for pid in p.pieces:
                self.assertEqual([r['proposal_id'] for r in fast.primary[pid]],
                                 [r['proposal_id'] for r in reference.primary[pid]])
                self.assertEqual([r['proposal_id'] for r in fast.indexed[pid]],
                                 [r['proposal_id'] for r in reference.indexed[pid]])
            self.assertGreater(fast.counts['primary_mode_pair_cache_hits'],0)


if __name__=='__main__':unittest.main()
