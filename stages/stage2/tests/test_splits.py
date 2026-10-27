from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from paper_reconstruction_stage2.config import DEFAULT_SPLIT_RATIOS
from paper_reconstruction_stage2.splits import assign_source_splits, exact_schedule


class SplitTests(unittest.TestCase):
    def test_exact_schedule_uses_requested_counts(self) -> None:
        schedule = exact_schedule(100, DEFAULT_SPLIT_RATIOS, seed=12)
        self.assertEqual(
            Counter(schedule),
            Counter(train=70, validation=10, calibration=10, holdout=10),
        )
        self.assertEqual(
            schedule, exact_schedule(100, DEFAULT_SPLIT_RATIOS, seed=12)
        )

    def test_sources_are_assigned_once_as_whole_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index in range(8):
                path = root / f"source_{index:02d}.png"
                path.write_bytes(f"source-{index}".encode("ascii"))
                sources.append(path)
            assignments, records = assign_source_splits(
                sources, root, DEFAULT_SPLIT_RATIOS, seed=91
            )
            self.assertEqual(set(assignments), set(sources))
            self.assertEqual(len(records), len(sources))
            self.assertEqual(len({item["source_relpath"] for item in records}), 8)
            self.assertEqual(
                set(item["split"] for item in records),
                set(DEFAULT_SPLIT_RATIOS),
            )
            self.assertTrue(
                all(len(item["source_sha256"]) == 64 for item in records)
            )


if __name__ == "__main__":
    unittest.main()
