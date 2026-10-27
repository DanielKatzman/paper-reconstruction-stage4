from __future__ import annotations

import unittest

import numpy as np

from paper_reconstruction_stage2.fragments import (
    FragmentProposalConfig,
    _endpoint_adjust,
    best_fragment_matches,
    circular_runs,
    fragment_recall,
    propose_fragments,
)


class FragmentTests(unittest.TestCase):
    def test_circular_run_wraps_without_breaking(self) -> None:
        runs = circular_runs(np.array([True, True, False, False, True]))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].tolist(), [4, 0, 1])

    def test_hysteresis_proposes_and_recovers_seam(self) -> None:
        count = 80
        roles = np.zeros((count, 3), dtype=np.float32)
        roles[:, 1] = 0.92
        roles[:, 0] = 0.04
        roles[:, 2] = 0.04
        seam_indices = np.arange(20, 49)
        roles[seam_indices] = (0.90, 0.05, 0.05)
        transition = np.full(count, 0.05, dtype=np.float32)
        transition[[20, 48]] = 0.95
        proposals = propose_fragments(
            roles,
            transition,
            FragmentProposalConfig(
                smoothing_sigma_samples=0.0,
                minimum_fragment_points=5,
                endpoint_search_radius=2,
            ),
        )
        self.assertGreaterEqual(len(proposals), 1)
        matched, truth_count, best = fragment_recall(
            proposals, [seam_indices], count, top_k=5, iou_threshold=0.80
        )
        self.assertEqual(truth_count, 1)
        self.assertEqual(matched, 1)
        self.assertGreaterEqual(best[0], 0.80)
        matches = best_fragment_matches(
            proposals, [seam_indices], count, top_k=5
        )
        self.assertEqual(matches[0]["candidate_rank"], 1)
        self.assertGreaterEqual(matches[0]["best_iou"], 0.80)

    def test_internal_transition_can_split_a_merged_run(self) -> None:
        count = 100
        roles = np.zeros((count, 3), dtype=np.float32)
        roles[:, 1] = 0.95
        roles[20:80, 0] = 0.90
        roles[20:80, 1] = 0.05
        transition = np.zeros(count, dtype=np.float32)
        transition[[20, 50, 79]] = 0.95
        proposals = propose_fragments(
            roles,
            transition,
            FragmentProposalConfig(
                smoothing_sigma_samples=0.0,
                minimum_fragment_points=8,
                endpoint_search_radius=1,
                transition_split_threshold=0.80,
            ),
        )
        self.assertTrue(any(item["end_index"] == 50 for item in proposals))
        self.assertTrue(any(item["start_index"] == 50 for item in proposals))
        self.assertTrue(
            any(
                item["start_index"] == 20
                and item["end_index"] == 79
                and item["hypothesis"].startswith("parent")
                for item in proposals
            )
        )

    def test_endpoint_adjustment_cannot_collapse_nearly_complete_run(self) -> None:
        count = 100
        run = np.concatenate(
            (
                np.arange(50, count, dtype=np.int64),
                np.arange(0, 40, dtype=np.int64),
            )
        )
        transition = np.zeros(count, dtype=np.float32)
        # This point appears in both endpoint search windows. The old
        # adjustment selected it for both ends and collapsed 90 points to one.
        transition[44] = 0.99
        adjusted = _endpoint_adjust(
            run,
            transition,
            radius=6,
            contour_length=count,
            minimum_points=8,
        )
        self.assertEqual(adjusted.tolist(), run.tolist())

    def test_public_proposals_never_violate_minimum_after_adjustment(self) -> None:
        count = 100
        roles = np.zeros((count, 3), dtype=np.float32)
        roles[:, 0] = 0.95
        roles[:, 1] = 0.03
        roles[:, 2] = 0.02
        roles[40:50] = (0.03, 0.95, 0.02)
        transition = np.zeros(count, dtype=np.float32)
        transition[44] = 0.99
        proposals = propose_fragments(
            roles,
            transition,
            FragmentProposalConfig(
                smoothing_sigma_samples=0.0,
                high_threshold=0.8,
                low_threshold=0.25,
                relaxed_high_threshold=0.7,
                relaxed_low_threshold=0.20,
                strict_high_threshold=0.9,
                strict_low_threshold=0.5,
                minimum_fragment_points=8,
                endpoint_search_radius=6,
            ),
        )
        self.assertTrue(proposals)
        self.assertTrue(all(item["length_points"] >= 8 for item in proposals))
        self.assertTrue(any(item["length_points"] == 90 for item in proposals))

    def test_relaxed_split_is_an_additional_hypothesis(self) -> None:
        count = 100
        roles = np.zeros((count, 3), dtype=np.float32)
        roles[:, 1] = 0.95
        roles[20:80, 0] = 0.90
        roles[20:80, 1] = 0.05
        transition = np.zeros(count, dtype=np.float32)
        transition[[20, 79]] = 0.95
        transition[50] = 0.35
        proposals = propose_fragments(
            roles,
            transition,
            FragmentProposalConfig(
                smoothing_sigma_samples=0.0,
                minimum_fragment_points=8,
                endpoint_search_radius=0,
                transition_split_threshold=0.50,
                relaxed_transition_split_threshold=0.25,
            ),
        )
        self.assertTrue(
            any(
                item["start_index"] == 20
                and item["end_index"] == 79
                and item["hypothesis"] == "parent_raw"
                for item in proposals
            )
        )
        self.assertTrue(
            any(
                item["end_index"] == 50
                and item["hypothesis"] == "split_relaxed"
                for item in proposals
            )
        )
        self.assertTrue(
            any(
                item["start_index"] == 50
                and item["hypothesis"] == "split_relaxed"
                for item in proposals
            )
        )
        keys = {
            (item["start_index"], item["end_index"], item["length_points"])
            for item in proposals
        }
        self.assertEqual(len(keys), len(proposals))
        relevant_ranks = [
            item["rank"]
            for item in proposals
            if item["hypothesis"] in {"parent_raw", "split_relaxed"}
        ]
        self.assertTrue(relevant_ranks)
        self.assertLessEqual(max(relevant_ranks), 5)


if __name__ == "__main__":
    unittest.main()
