from dataclasses import replace
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from paper_reconstruction_contracts import ContractError, map_points
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.geometry import normalize, relative, warp_check
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import singleton_state
from paper_reconstruction_stage4.outputs import (atomic_json, build_result, digest, output_session, publish,
                                                 file_hash, read_result, request_identity, validate_result)
from paper_reconstruction_stage4.search.partial_beam import PartialSearch
from paper_reconstruction_stage4.solver import solve
from paper_reconstruction_stage4.support import Arc, SupportIndex, native_arc, overlaps
from tests.fixtures import build_fixture, translation

ROOT = Path(__file__).resolve().parents[1]


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage4_offline_evaluator", ROOT / "evaluation" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryTests(unittest.TestCase):
    def test_vendored_contract_sdk_matches_recorded_upstream_bytes(self):
        origin = json.loads((ROOT / "vendor" / "ORIGIN.json").read_text())
        for name, expected in origin["sdk_files"].items():
            self.assertEqual(file_hash(ROOT / "vendor" / "contracts-sdk" / name), expected, name)

    def test_projective_composition_division_and_inverse(self):
        h = np.array([[1., .1, 5.], [0., 1., 2.], [.001, -.0001, 1.]])
        k = translation(4, 3)
        points = np.array([[1., 2.], [20., 15.]])
        np.testing.assert_allclose(map_points(points, h @ k), map_points(map_points(points, k), h))
        np.testing.assert_allclose(map_points(map_points(points, h), np.linalg.inv(h)), points, atol=1e-10)
        self.assertGreater(abs(map_points(points, h)[1, 0] - (points @ h[:2, :2].T + h[:2, 2])[1, 0]), .1)

    def test_reflection_pole_and_anisotropy_rejected(self):
        config = SolverConfig()
        matrices = [np.diag([-1., 1., 1.]), np.array([[1., 0., 0.], [0., 1., 0.], [-.08, 0., 1.]]), np.diag([5., .2, 1.])]
        for h in matrices:
            self.assertFalse(warp_check(h, (32, 32), (32, 32), config)["valid"])
        self.assertTrue(warp_check(translation(27), (32, 32), (32, 32), config)["valid"])

    def test_relative_map_is_invariant_to_common_projective_gauge(self):
        poses = {"a": translation(0), "b": translation(27, 27)}
        common = np.array([[1., .1, 5.], [0., 1., 2.], [.001, .0001, 1.]])
        np.testing.assert_allclose(relative(poses, "a", "b"), relative({p: common @ h for p, h in poses.items()}, "a", "b"), atol=1e-10)

    def test_config_rejects_unknown_nonfinite_and_wrong_types(self):
        for data in ({"beam_width": True}, {"join_cost": float("nan")}, {"unrecognized": 1}, {"solver": "magic"}, {"max_seconds": 0.}):
            with self.assertRaises(ValueError):
                SolverConfig.from_dict(data)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = SolverConfig(max_seconds=20.)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name="input", **kwargs):
        reference = build_fixture(self.root / name, **kwargs)
        return load_problem(self.root / name, self.config), reference


class ContractTests(FixtureCase):
    def test_fixture_independent_loader_and_zero_candidate_inventory(self):
        problem, _ = self.fixture(rows=1, cols=2, isolated=True)
        self.assertEqual(len(problem.pieces), 3)
        self.assertIn("piece_isolated", problem.pieces)
        self.assertEqual(problem.input_summary["archive_passes"], 1)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("paper_reconstruction_stage3", sys.modules)

    def test_missing_barrier_and_tampered_assets_fail(self):
        with self.assertRaises(ContractError):
            load_problem(self.root, self.config)
        problem, _ = self.fixture()
        mask_name = problem.manifest["pieces"]["piece_00"]["mask"]
        with (problem.root / mask_name).open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(ContractError):
            load_problem(problem.root, self.config)

    def test_input_and_occurrence_budgets_fail_explicitly(self):
        problem, _ = self.fixture()
        with self.assertRaisesRegex(ContractError, "byte budget"):
            load_problem(problem.root, replace(self.config, max_input_bytes=1))
        with self.assertRaisesRegex(ContractError, "occurrence indexing budget"):
            load_problem(problem.root, replace(self.config, max_working_occurrences=1))

    def test_closed_truth_field_is_rejected(self):
        problem, _ = self.fixture()
        manifest = copy.deepcopy(problem.manifest)
        manifest["provenance"]["teacher_transform"] = np.eye(3).tolist()
        manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        atomic_json(problem.root / "bundle.json", manifest)
        with self.assertRaisesRegex(ContractError, "Closed evaluation"):
            load_problem(problem.root, self.config)

    def test_unknown_contract_and_closed_truth_fail(self):
        problem, _ = self.fixture()
        manifest = copy.deepcopy(problem.manifest)
        manifest["schema_version"] = "stage3.assembly-evidence.v999"
        atomic_json(problem.root / "bundle.json", manifest)
        with self.assertRaises(ContractError):
            load_problem(problem.root, self.config)


class SupportTests(FixtureCase):
    def test_wrapped_and_window_fraction_map_to_native_arc(self):
        problem, _ = self.fixture(rows=1, cols=2)
        full = native_arc(problem, "piece_01/left", None, (0., 1.))
        cropped = native_arc(problem, "piece_01/left", {"start_px": 3., "end_px": 23.}, (.25, .75))
        self.assertAlmostEqual(full.length, 27.)
        self.assertAlmostEqual(cropped.length, 10.)
        self.assertAlmostEqual(cropped.points[0, 1], 21.)
        problem.candidates["piece_01/wrap"] = {"piece_id": "piece_01", "candidate": {"start_index": 100, "end_index": 5, "wraps": True}}
        crossing = native_arc(problem, "piece_01/wrap", None, (0., 1.))
        self.assertEqual(len(crossing.intervals), 2)

    def test_endpoint_tolerance_does_not_allow_substantial_reuse(self):
        points = np.zeros((2, 2))
        first = Arc("a", ((0., 20.),), points, 20.)
        self.assertFalse(overlaps(first, Arc("a", ((19., 30.),), points, 11.), self.config))
        self.assertTrue(overlaps(first, Arc("a", ((10., 30.),), points, 20.), self.config))
        self.assertFalse(overlaps(first, Arc("b", ((0., 20.),), points, 20.), self.config))

    def test_reverse_query_direction_preserves_physical_support(self):
        problem, _ = self.fixture(rows=1, cols=2)
        original = problem.edges[0].occurrences[0]
        reverse = copy.deepcopy(original)
        for q, t in (("query_key", "target_key"), ("query_piece", "target_piece"), ("query_observation", "target_observation")):
            reverse[q], reverse[t] = reverse[t], reverse[q]
        reverse["occurrence_id"] = "reversed"
        reverse["pose"]["alignment"]["transform"] = np.linalg.inv(problem.edges[0].transform).tolist()
        problem.edges[0].occurrences.append(reverse)
        support = SupportIndex(problem, self.config)
        by_piece = {arc.piece: arc for arc in support.arcs[original["occurrence_id"]]}
        for arc in support.arcs["reversed"]:
            self.assertEqual(arc.intervals, by_piece[arc.piece].intervals)

    def test_assignment_backtracks_to_nonrepresentative_occurrence(self):
        problem, _ = self.fixture(rows=1, cols=3)
        first, second = problem.edges
        first.occurrences = [{"occurrence_id": "a_bad"}, {"occurrence_id": "b_good"}]
        second.occurrences = [{"occurrence_id": "c_only"}]
        support = object.__new__(SupportIndex)
        support.config = self.config
        points = np.zeros((2, 2))
        support.arcs = {"a_bad": (Arc("shared", ((0., 20.),), points, 20.),),
                        "b_good": (Arc("shared", ((30., 50.),), points, 20.),),
                        "c_only": (Arc("shared", ((0., 20.),), points, 20.),)}
        with patch("paper_reconstruction_stage4.support.contact_error", return_value=0.):
            assigned, reason, _ = support.assign((first.id, second.id), {}, {e.id: e for e in problem.edges})
        self.assertEqual(reason, "assigned")
        self.assertEqual(assigned[first.id], "b_good")


class SolverTests(FixtureCase):
    def test_known_chain_and_isolated_piece_preserved(self):
        problem, _ = self.fixture(rows=1, cols=3, isolated=True)
        result = solve(problem, self.config)
        self.assertEqual([len(c) for c in result.states[0].components()], [3, 1])
        np.testing.assert_allclose(result.states[0].poses["piece_02"], translation(54))
        self.assertEqual(len(result.states[0].assignments), 2)

    def test_complete_square_loop_and_alternative_rejection(self):
        problem, _ = self.fixture(alternatives=True)
        result = solve(problem, replace(self.config, loop_proposals=True, repair_rounds=1))
        self.assertEqual(len(result.states[0].components()), 1)
        self.assertEqual(len(result.states[0].selected), 4)
        self.assertGreater(result.diagnostics["loop_search"]["accepted_seeds"], 0)
        self.assertGreater(result.rejections.get("no_valid_contact_support", 0), 0)
        self.assertTrue(result.diagnostics["final_feasibility"][0]["valid"])

    def test_weak_join_abstains_while_legacy_control_joins(self):
        problem, _ = self.fixture(rows=1, cols=2, weight=.3)
        improved = solve(problem, self.config)
        legacy = solve(problem, replace(self.config, solver="legacy_forest_control_v1"))
        self.assertFalse(improved.states[0].selected)
        self.assertEqual(len(legacy.states[0].selected), 1)

    def test_all_singletons_supported_without_any_edges(self):
        problem, _ = self.fixture(rows=1, cols=1, isolated=True)
        result = solve(problem, self.config)
        self.assertEqual(len(result.states[0].components()), 2)
        self.assertEqual(result.diagnostics["proposals"], 0)

    def test_collision_and_reflection_reject_proposals(self):
        problem, _ = self.fixture(rows=1, cols=2)
        edge = problem.edges[0]
        edge.transform = np.eye(3)
        result = solve(problem, self.config)
        self.assertFalse(result.states[0].selected)
        self.assertGreater(result.rejections.get("interior_collision", 0), 0)
        edge.transform = np.diag([-1., 1., 1.])
        result = solve(problem, self.config)
        self.assertFalse(result.states[0].selected)
        self.assertGreater(result.rejections.get("orientation_reversal", 0), 0)

    def test_final_collision_sampling_can_reject_coarse_acceptance(self):
        problem, _ = self.fixture(rows=1, cols=2)
        checks = Feasibility(problem, replace(self.config, collision_grid=2, maximum_assembly_collision=.3))
        coarse = checks.pair("piece_00", "piece_01", translation(10, 10), final=False)
        final = checks.pair("piece_00", "piece_01", translation(10, 10), final=True)
        self.assertTrue(coarse["valid"])
        self.assertFalse(final["valid"])
        self.assertGreater(final["overlap"], coarse["overlap"])

    def test_budget_and_determinism(self):
        problem, _ = self.fixture()
        config = replace(self.config, max_proposals=1)
        one, two = solve(problem, config), solve(problem, config)
        self.assertEqual(one.termination, "proposal_budget")
        self.assertEqual(one.states[0].selected, two.states[0].selected)
        self.assertEqual(one.diagnostics["proposals"], 1)
        for p in problem.pieces:
            np.testing.assert_array_equal(one.states[0].poses[p], two.states[0].poses[p])

    def test_contradictory_selected_cycle_is_rejected(self):
        problem, _ = self.fixture()
        support, feasibility = SupportIndex(problem, self.config), Feasibility(problem, self.config)
        search = PartialSearch(problem, self.config, feasibility, support)
        # First construct a known path around three sides of the square.
        by_pair = {(e.a, e.b): e for e in problem.edges}
        state = singleton_state(problem)
        for pair in (("piece_00", "piece_01"), ("piece_00", "piece_02"), ("piece_01", "piece_03")):
            state = search.attempt(state, by_pair[pair])
            self.assertIsNotNone(state)
        closing = by_pair["piece_02", "piece_03"]
        closing.transform = translation(60)
        self.assertIsNone(search.attempt(state, closing))
        self.assertEqual(search.rejections["selected_cycle_inconsistent"], 1)

    def test_refinement_uses_redundancy_and_preserves_feasibility(self):
        problem, _ = self.fixture(drift=3.)
        initial = solve(problem, self.config)
        refined = solve(problem, replace(self.config, refine=True, refine_max_nfev=40))
        self.assertEqual(len(refined.states[0].selected), 4)
        self.assertTrue(refined.diagnostics["refinement"][0]["attempted"])
        self.assertTrue(refined.diagnostics["refinement"][0]["accepted"])
        self.assertGreater(refined.states[0].score, initial.states[0].score)
        self.assertTrue(refined.diagnostics["final_feasibility"][0]["valid"])

    def test_tree_only_refinement_reports_no_redundancy(self):
        problem, _ = self.fixture(rows=1, cols=3)
        result = solve(problem, replace(self.config, refine=True))
        self.assertEqual(result.diagnostics["refinement"][0]["reason"], "no_redundant_selected_contacts")


class PublicationTests(FixtureCase):
    def publish_fixture(self, **kwargs):
        problem, reference = self.fixture(**kwargs)
        result = solve(problem, self.config)
        request = request_identity(problem, self.config)
        value = build_result(problem, self.config, result, request)
        output = self.root / "result"
        with output_session(output, request) as work:
            publish(output, work, value, problem, result.states)
        return problem, reference, output, value, request

    def test_atomic_publication_preview_and_immutable_resume(self):
        problem, reference, output, value, request = self.publish_fixture()
        loaded = read_result(output)
        self.assertEqual(loaded["status"], "complete_candidate")
        self.assertTrue((output / "preview.png").is_file())
        with self.assertRaises(ValueError):
            with output_session(output, request):
                pass
        with output_session(output, request, resume=True) as work:
            self.assertIsNone(work)
        with self.assertRaises(ValueError):
            with output_session(output, {**request, "run_id": "f" * 64}, resume=True):
                pass

    def test_result_tampering_and_duplicate_piece_fail(self):
        _, _, output, value, _ = self.publish_fixture()
        bad = copy.deepcopy(value)
        bad["layouts"][0]["pieces"][1]["piece_id"] = bad["layouts"][0]["pieces"][0]["piece_id"]
        with self.assertRaises(ValueError):
            validate_result(bad)
        with (output / "result.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaises(ValueError):
            read_result(output)

    def test_failure_has_no_ready_manifest_and_can_replay_matching_work(self):
        problem, _ = self.fixture()
        request = request_identity(problem, self.config)
        output = self.root / "failure"
        with self.assertRaises(RuntimeError):
            with output_session(output, request):
                raise RuntimeError("fixture failure")
        self.assertFalse((output / "manifest.json").exists())
        self.assertTrue((output.with_name(output.name + ".work") / "failure.json").exists())
        with output_session(output, request, resume=True) as work:
            self.assertIsNotNone(work)

    def test_cli_round_trip_and_separate_evaluation(self):
        _, reference, output, value, _ = self.publish_fixture()
        done = subprocess.run([sys.executable, "-m", "paper_reconstruction_stage4", "inspect", "--result", str(output)],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["selected_joins"], 4)
        report = load_evaluator().evaluate(value, reference)
        self.assertTrue(report["strict_whole_page_pass"])
        self.assertEqual(report["passing_implied_contacts"], 4)
        self.assertEqual(report["unresolved_ordered_pairs"], 0)

    def test_unresolved_pieces_are_in_evaluation_denominator(self):
        _, reference, _, value, _ = self.publish_fixture(rows=1, cols=3, weight=.3)
        report = load_evaluator().evaluate(value, reference)
        self.assertFalse(report["strict_whole_page_pass"])
        self.assertEqual(report["expected_contacts"], 2)
        self.assertEqual(report["passing_implied_contacts"], 0)
        self.assertEqual(report["unresolved_ordered_pairs"], 6)
        self.assertIsNone(report["maximum_pairwise_p95_native_px"])

    def test_evaluator_cannot_write_inside_prediction_using_file_argument(self):
        _, reference, output, value, _ = self.publish_fixture()
        reference_file = self.root / "reference.json"
        atomic_json(reference_file, reference)
        done = subprocess.run([sys.executable, str(ROOT / "evaluation" / "evaluate.py"), "--result", str(output / "result.json"),
                               "--reference", str(reference_file), "--output", str(output / "forbidden_evaluation.json")],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertFalse((output / "forbidden_evaluation.json").exists())

    def test_cli_refuses_to_write_inside_input(self):
        problem, _ = self.fixture()
        output = problem.root / "forbidden_output"
        done = subprocess.run([sys.executable, "-m", "paper_reconstruction_stage4", "solve", "--bundle", str(problem.root),
                               "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertFalse(output.exists())

    def test_cli_invalid_input_does_not_publish_prediction(self):
        output = self.root / "bad_output"
        done = subprocess.run([sys.executable, "-m", "paper_reconstruction_stage4", "solve", "--bundle", str(self.root / "missing"),
                               "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
