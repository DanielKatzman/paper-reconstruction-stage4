from dataclasses import replace
import tempfile
import unittest
import numpy as np
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.page import prioritize_finalists
from paper_reconstruction_stage4.feasibility import Feasibility
from tests.test_page_v10 import fixture
from tests.fixtures import translation


class ClosureSelectionTests(unittest.TestCase):
    def test_normalized_page_features_do_not_assume_dpi_or_format(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,context,state,_=fixture(folder,rows=1,cols=2)
            before=context.features(state)
            from paper_reconstruction_stage4.page import PageContext
            for piece in p.pieces.values():
                piece.mask=np.repeat(np.repeat(piece.mask,2,axis=0),2,axis=1)
                piece.contour['points_xy']=piece.contour['points_xy']*2
            # Scale all observed pixels together; relative piece scale stays 1.
            for arcs in context.support.arcs.values():
                for arc in arcs:
                    arc.points=arc.points*2;arc.intervals=tuple((a*2,b*2) for a,b in arc.intervals);arc.length*=2
            poses={pid:h.copy() for pid,h in state.poses.items()}
            for h in poses.values():h[:2,2]*=2
            scaled=replace(state,poses=poses)
            after=PageContext(p,c,context.support).features(scaled)
            self.assertAlmostEqual(before['fill_fraction'],after['fill_fraction'],delta=.03)
            self.assertFalse(after['aspect_ratio_prior_used'])
            self.assertGreater(after['rectangle_sides_px'][1]/after['rectangle_sides_px'][0],1.8)

    def test_valid_complete_page_precedes_higher_utility_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,context,complete,_=fixture(folder)
            c=replace(c,page_selection_policy='supported-closure-first.v1');context.config=c
            partial=singleton_state(p);partial.score=100.
            p.page_context=context
            states,checks=prioritize_finalists(p,c,[partial,complete],Feasibility(p,c))
            self.assertEqual(len(states[0].components()),1)
            self.assertGreater(context.utility(partial,context.features(partial)),context.utility(complete,context.features(complete)))
            self.assertEqual(context.closure_priority(complete,context.features(complete)),1)
            self.assertEqual(checks['checked'],2)

    def test_bad_complete_page_has_no_closure_preference(self):
        with tempfile.TemporaryDirectory() as folder:
            p,c,context,complete,_=fixture(folder)
            context.config=replace(c,page_selection_policy='supported-closure-first.v1')
            bad=replace(complete,poses={**complete.poses,'piece_03':translation(270,270)})
            self.assertEqual(context.closure_priority(bad,context.features(bad)),0)
            legacy=replace(context.config,page_selection_policy='utility.v1');context.config=legacy
            self.assertEqual(context.closure_priority(complete,context.features(complete)),0)


if __name__=='__main__':unittest.main()
