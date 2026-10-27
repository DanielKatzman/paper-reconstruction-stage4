import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from paper_reconstruction_stage4.active_support import ActiveSupportIndex, arc_record, slice_arc
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.structural import StructuralScorer
from tests.fixtures import build_fixture


class BoundaryReuseTests(unittest.TestCase):
    def test_exact_residuals_across_pose_support_and_sampling_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"input"
            reference = build_fixture(root, rows=2, cols=2)
            config = SolverConfig(solver="joint_beam_v3", objective_version="observable-joint.v4", support_policy="active-subarc.v1")
            problem = load_problem(root, config)
            support = ActiveSupportIndex(problem, config)
            poses = {p: np.asarray(v["native_to_reference"]) for p,v in reference["pieces"].items()}
            ids = tuple(e.id for e in problem.edges)
            assigned, _, _ = support.assign(ids, poses, {e.id:e for e in problem.edges})
            self.assertIsNotNone(assigned)
            state = State(poses, {p:min(poses) for p in poses}, ids, ids, assigned)
            source = Path(__file__).resolve().parents[1]/"controls/stage4_joint_v4_validation_baseline/src/paper_reconstruction_stage4/structural.py"
            spec = importlib.util.spec_from_file_location("paper_reconstruction_stage4._baseline_structure", source)
            baseline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline)
            old, new = baseline.StructuralScorer(problem, config, support), StructuralScorer(problem, config, support)
            for scorer in (old, new):
                for profile in scorer.profiles.values():
                    shape = profile["ink"].shape
                    profile["ink"] = np.arange(np.prod(shape)).reshape(shape)/np.prod(shape)
            def check(candidate, samples=17, appearance=True):
                expected = old.boundary_residual(candidate, samples, appearance)
                actual = new.boundary_residual(candidate, samples, appearance)
                for a,b in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(a,b)
                # A caller mutating its result must not mutate cached residuals.
                if len(actual[0]): actual[0][0] = 12345.
            check(state)
            check(state)
            self.assertGreater(new.boundary_cache_hits, 0)
            altered = copy.deepcopy(state)
            altered.poses[sorted(poses)[1]][0,2] += .37
            check(altered)
            check(state, samples=11)
            check(state, appearance=False)
            # A new source-contained support interval requires a distinct entry.
            eid = ids[0]
            arcs = support.arcs[assigned[eid]]
            altered.active_support[eid] = {"support_id": "different-native-subarc", "arcs": [arc_record(slice_arc(problem,a,.1,.9)) for a in arcs]}
            check(altered)
            check(altered)
            check(state)
            self.assertGreater(new.boundary_cache_misses, len(ids))
            # Full objective parity also covers native area/probe reuse.
            for candidate in (state, altered):
                self.assertEqual(old.evaluate(candidate), new.evaluate(candidate))
            for i in range(1, 11):
                candidate = copy.deepcopy(state)
                h = candidate.poses[sorted(poses)[1]]
                h[0,1] += i*.001
                h[2,0] += i*.000001
                self.assertEqual(old.evaluate(candidate), new.evaluate(candidate))
