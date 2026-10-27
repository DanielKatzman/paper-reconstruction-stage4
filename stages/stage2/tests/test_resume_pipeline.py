from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_reconstruction_stage2.boundary import BoundaryConfig
from paper_reconstruction_stage2.config import GeneratorConfig
from paper_reconstruction_stage2.generator import DatasetGenerator
from paper_reconstruction_stage2.observation import ObservationCompiler, ObservationConfig
from paper_reconstruction_stage2.pipeline import PipelinePaths, Stage2Pipeline


def _generator_config(output: Path, *, resume: bool = False) -> GeneratorConfig:
    return GeneratorConfig(
        output_dir=output,
        puzzle_count=2,
        seed=41,
        min_pieces=2,
        max_pieces=3,
        page_width=220,
        page_height=300,
        min_piece_area_fraction=0.01,
        min_piece_dimension_px=4,
        preview_max_side=320,
        resume=resume,
    )


class PuzzleResumeTests(unittest.TestCase):
    def test_canonical_resume_reuses_committed_puzzle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "canonical"
            first = DatasetGenerator(_generator_config(output)).generate()
            first_puzzle = output / first["puzzles"][0]["puzzle_path"]
            first_mtime = first_puzzle.stat().st_mtime_ns
            second_dir = (output / first["puzzles"][1]["puzzle_path"]).parent
            second_dir.rename(second_dir.with_name(f"{second_dir.name}.partial"))
            (output / "dataset_manifest.json").unlink()

            resumed = DatasetGenerator(
                _generator_config(output, resume=True)
            ).generate()
            self.assertEqual(
                resumed["dataset_fingerprint"], first["dataset_fingerprint"]
            )
            self.assertEqual(first_puzzle.stat().st_mtime_ns, first_mtime)

    def test_observation_resume_reuses_committed_puzzle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_dir = root / "canonical"
            canonical = DatasetGenerator(_generator_config(canonical_dir)).generate()
            output = root / "observations"
            base_config = ObservationConfig(
                canonical_dataset_dir=canonical_dir,
                output_dir=output,
                seed=43,
                views_per_piece=2,
                preview_max_side=320,
            )
            first = ObservationCompiler(base_config).compile()
            first_manifest = output / first["puzzles"][0]["observation_manifest_path"]
            first_mtime = first_manifest.stat().st_mtime_ns
            second_dir = (
                output / first["puzzles"][1]["observation_manifest_path"]
            ).parent
            second_dir.rename(second_dir.with_name(f"{second_dir.name}.partial"))
            (output / "observation_dataset_manifest.json").unlink()

            resumed = ObservationCompiler(
                ObservationConfig(
                    canonical_dataset_dir=canonical_dir,
                    output_dir=output,
                    seed=43,
                    views_per_piece=2,
                    preview_max_side=320,
                    resume=True,
                )
            ).compile()
            self.assertEqual(
                resumed["observation_dataset_fingerprint"],
                first["observation_dataset_fingerprint"],
            )
            self.assertEqual(first_manifest.stat().st_mtime_ns, first_mtime)
            self.assertEqual(len(canonical["puzzles"]), len(resumed["puzzles"]))


class UnifiedPipelineTests(unittest.TestCase):
    def _pipeline(self, root: Path, *, resume: bool) -> Stage2Pipeline:
        paths = PipelinePaths.from_root(root)
        canonical = _generator_config(paths.canonical, resume=resume)
        observations = ObservationConfig(
            canonical_dataset_dir=paths.canonical,
            output_dir=paths.observations,
            seed=43,
            views_per_piece=2,
            preview_max_side=320,
            resume=resume,
        )
        boundaries = BoundaryConfig(
            canonical_dataset_dir=paths.canonical,
            observation_dataset_dir=paths.observations,
            output_dir=paths.boundaries,
            preview_max_side=400,
            resume=resume,
        )
        return Stage2Pipeline(
            paths=paths,
            canonical_config=canonical,
            observation_config=observations,
            boundary_config=boundaries,
            resume=resume,
        )

    def test_one_public_pipeline_runs_verifies_and_reuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pipeline"
            first = self._pipeline(root, resume=False).run()
            self.assertEqual(first["status"], "complete")
            state = json.loads(
                (root / "pipeline_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["stages"]["canonical"]["completed_puzzles"], 2)
            second_pipeline = self._pipeline(root, resume=True)
            second = second_pipeline.run()
            self.assertEqual(second["status"], "complete")
            self.assertTrue(second_pipeline.verify()["valid"])


if __name__ == "__main__":
    unittest.main()
