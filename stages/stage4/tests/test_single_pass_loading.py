from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paper_reconstruction_contracts import load_bundle, ContractError
from paper_reconstruction_contracts.reader import AssemblyInput
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.outputs import atomic_json, canonical, digest, file_hash
from tests.fixtures import build_fixture


class SinglePassLoadingTests(unittest.TestCase):
    def test_one_complete_sdk_archive_validation_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"input"
            build_fixture(root, rows=2, cols=2)
            calls = []
            original = AssemblyInput.hypotheses
            def tracked(bundle):
                calls.append("started")
                yield from original(bundle)
                calls.append("exhausted")
            with patch.object(AssemblyInput, "hypotheses", tracked):
                problem = load_problem(root, SolverConfig())
            self.assertEqual(calls, ["started", "exhausted"])
            self.assertTrue(problem.input_summary["full_archive_validated"])
            self.assertEqual(problem.input_summary["archive_passes"], 1)
            self.assertEqual(len(problem.edges), 4)

    def test_inventory_check_at_archive_exhaustion_still_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"input"
            build_fixture(root)
            path = root/"bundle.json"
            manifest = json.loads(path.read_text())
            manifest["counts"]["geometries"] += 1
            manifest["manifest_sha256"] = digest({k:v for k,v in manifest.items() if k != "manifest_sha256"})
            atomic_json(path, manifest)
            load_bundle(root, full=False)  # Metadata alone cannot detect this.
            with self.assertRaisesRegex(ContractError, "inventory"):
                load_problem(root, SolverConfig())

    def test_unused_malformed_tail_is_validated_for_both_evidence_policies(self):
        for policy in ("working_pool", "working_plus_diverse_archive.v1"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as d:
                root = Path(d)/"input"
                build_fixture(root, rows=2, cols=2)
                manifest = json.loads((root/"bundle.json").read_text())
                pool = json.loads((root/"working_pool.json").read_text())
                groups = [json.loads(line) for line in gzip.decompress((root/"hypotheses.jsonl.gz").read_bytes()).splitlines()]
                last = groups[-1]
                pool["edges"] = [e for e in pool["edges"] if e["hypothesis_id"] != last["hypothesis_id"]]
                pool["pairs"] = [r for r in pool["pairs"] if r["pieces"] != [last["piece_a"], last["piece_b"]]]
                atomic_json(root/"working_pool.json", pool)
                last["occurrences"][0]["pose"]["alignment"]["transform"][0][2] += 123.
                (root/"hypotheses.jsonl.gz").write_bytes(gzip.compress(b"".join(canonical(g)+b"\n" for g in groups), mtime=0))
                manifest["counts"]["working_edges"] = len(pool["edges"])
                for key in ("archive", "working_pool"):
                    target = root/manifest[key]["path"]
                    manifest[key].update(size=target.stat().st_size, sha256=file_hash(target))
                manifest["manifest_sha256"] = digest({k:v for k,v in manifest.items() if k != "manifest_sha256"})
                atomic_json(root/"bundle.json", manifest)
                load_bundle(root, full=False)
                with self.assertRaisesRegex(ContractError, "direction"):
                    load_problem(root, replace(SolverConfig(), evidence_policy=policy))
