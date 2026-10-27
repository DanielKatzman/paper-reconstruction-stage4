import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark.common import atomic_json
from benchmark.cpu_qualification import profile_command,cpu_profile,compare_rankings,diagnose as diagnose_cpu
from benchmark.quality_experiments import diagnose as diagnose_quality
from benchmark.quality_experiments import validate_variant
from benchmark.snapshots import freeze,verify_snapshot
from tests.fixtures import build_fixture
from paper_reconstruction_contracts import load_bundle


class ExperimentControllerTests(unittest.TestCase):
    def test_variant_names_bind_objectives_and_cannot_change_search_budgets(self):
        baseline={"objective_version":"observable-joint.v4","max_proposals":100}
        config={**baseline,"joint_search_policy":"component-beam.v1"}
        validate_variant("search",config,baseline)
        with self.assertRaisesRegex(ValueError,"binding differs"):validate_variant("scale",config,baseline)
        with self.assertRaisesRegex(ValueError,"budgets"):
            validate_variant("search",{**config,"max_proposals":1000},baseline)

    def test_backend_rankings_report_numeric_and_topk_changes_without_assuming_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            from benchmark.common import digest
            root=Path(folder)
            rows=[{"query":0,"target":i,"scores":{"ranker":.9-i*.1}} for i in range(6)]
            changed=copy.deepcopy(rows);changed[-1]["scores"]["ranker"]=.95
            atomic_json(root/"cuda.json",{"rows":rows,"payload_sha256":digest(rows)})
            atomic_json(root/"cpu.json",{"rows":changed,"payload_sha256":digest(changed)})
            report=compare_rankings(root/"cpu.json",root/"cuda.json")
            self.assertFalse(report["bitwise_equality_assumed"])
            self.assertAlmostEqual(report["models"]["ranker"]["mean_top5_target_overlap"],.8)
            self.assertGreater(report["models"]["ranker"]["maximum_probability_difference"],.5)
            atomic_json(root/"cpu.json",{"rows":rows,"payload_sha256":digest(changed)})
            with self.assertRaisesRegex(ValueError,"payload checksum"):
                compare_rankings(root/"cpu.json",root/"cuda.json")

    def test_cpu_commands_bind_their_backend_and_comparison_baseline_explicitly(self):
        case={"case_id":"example","stage1":"observations","stage2":"boundaries"}
        v1=profile_command(case,"v1","runtime1","output1")
        v3=profile_command(case,"v3","runtime3","output3","output1/evidence")
        self.assertEqual(v1[v1.index("--device")+1],"cpu")
        self.assertNotIn("--baseline",v1)
        self.assertEqual(v3[v3.index("--baseline")+1],"output1/evidence")
        self.assertTrue(any("profile_stage3_visual.py" in arg for arg in v3))
        self.assertNotIn("--preset",v1)

    def test_cpu_profile_cannot_relabel_a_cuda_profile_or_compare_to_a_cuda_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);build_fixture(root/"cpu/evidence",rows=1,cols=2)
            data=load_bundle(root/"cpu/evidence");m={"variant":"v3"}
            case={"case_id":data.manifest["puzzle_id"],"stage1":str(root/"stage1"),"stage2":str(root/"stage2")}
            data.manifest["provenance"]["inference_signature"]="sig"
            from benchmark.common import digest
            data.manifest["manifest_sha256"]=digest({k:v for k,v in data.manifest.items() if k!="manifest_sha256"})
            atomic_json(root/"cpu/evidence/bundle.json",data.manifest)
            p={"status":"profile_complete","scope":"full","sources_verified_unchanged":True,"source_validation_status":"passed",
                "ready_bundle_published":True,"training_performed":False,"evaluation_truth_opened":False,"runtime_policy_changed":False,
                "inference_result":{"device":"cpu"},
                "request":{**case,"puzzle_id":case["case_id"],"device":"cpu","scope":"full","cprofile":False,
                    "candidate_runtime":m,"output":str(root/"cpu"),"baseline":str(root/"cpu_v1/evidence"),"expected_inference_signature":"sig"},
                "comparison":{"status":"passed","equal":True,"archive_equal":True,"archive":{"different_groups":0}}}
            atomic_json(root/"cpu/profile.json",p)
            cpu_profile(root/"cpu",case,"v3",m,root/"cpu_v1/evidence")
            with self.assertRaisesRegex(ValueError,"against CPU v1"):
                cpu_profile(root/"cpu",case,"v3",m,root/"cuda_v1/evidence")
            p["request"]["device"]="cuda";atomic_json(root/"cpu/profile.json",p)
            with self.assertRaisesRegex(ValueError,"device/runtime"):
                cpu_profile(root/"cpu",case,"v3",m,root/"cpu_v1/evidence")

    def test_cpu_and_quality_reports_require_the_complete_serving_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/"request.json",{});atomic_json(root/"progress.json",{"phase":"predictions"})
            for function in (diagnose_cpu,diagnose_quality):
                with self.subTest(function=function.__module__),self.assertRaisesRegex(ValueError,"barrier"):
                    function(["--run",str(root)])

    def test_snapshot_is_immutable_and_configuration_changes_require_a_new_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/"src/paper_reconstruction_stage4";source.mkdir(parents=True)
            (source/"__init__.py").write_text("sample source")
            config=root/"configs/experiment.json";config.parent.mkdir();atomic_json(config,{"option":1})
            with patch("benchmark.snapshots.WORKSPACE",root),patch("benchmark.snapshots.implementation_identity",return_value={"sha256":"test"}):
                output=root/"controls/experiment";freeze(output,[config]);freeze(output,[config])
                self.assertEqual(verify_snapshot(output)["implementation"]["sha256"],"test")
                atomic_json(config,{"option":2})
                with self.assertRaisesRegex(ValueError,"configuration differs"):freeze(output,[config])
                with self.assertRaisesRegex(ValueError,"inside this workspace"):freeze(root/"outside",[config])
                (output/"src/paper_reconstruction_stage4/__init__.py").write_text("changed snapshot")
                with self.assertRaisesRegex(ValueError,"snapshot changed"):verify_snapshot(output)
