import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from benchmark.common import atomic_json,digest,file_hash,read_json
from benchmark.qualified_v3 import compare_evidence,validate_profile,select_cases
from benchmark.v3_integration import diagnose,verify_snapshot
from tests.fixtures import build_fixture


class QualifiedV3Tests(unittest.TestCase):
    def test_different_execution_manifests_require_complete_payload_equivalence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);build_fixture(root/"old",rows=1,cols=2)
            shutil.copytree(root/"old",root/"new")
            m = read_json(root/"new/bundle.json")
            m["identity"] = digest(["new execution"]);m["provenance"] = {"execution":"new"}
            m["manifest_sha256"] = digest({k:v for k,v in m.items() if k != "manifest_sha256"})
            atomic_json(root/"new/bundle.json",m)
            receipt = compare_evidence(root/"new",root/"old")
            self.assertNotEqual(receipt["left_manifest_sha256"],receipt["right_manifest_sha256"])
            poolpath = root/"new"/m["working_pool"]["path"]
            pool = read_json(poolpath);pool["edges"][0]["weight"] -= .01;atomic_json(poolpath,pool)
            m["working_pool"].update(size=poolpath.stat().st_size,sha256=file_hash(poolpath))
            m["manifest_sha256"] = digest({k:v for k,v in m.items() if k != "manifest_sha256"});atomic_json(root/"new/bundle.json",m)
            with self.assertRaisesRegex(ValueError,"evidence differs"):
                compare_evidence(root/"new",root/"old")

    def profile(self,root):
        case={"case_id":"case","baseline":str(root/"baseline"),"stage1":str(root/"stage1"),"stage2":str(root/"stage2")}
        manifest={"variant":"visual-reuse-v3"}
        p={"status":"profile_complete","scope":"full","sources_verified_unchanged":True,"source_validation_status":"passed",
           "ready_bundle_published":True,"evaluation_truth_opened":False,"training_performed":False,"runtime_policy_changed":False,
           "feature_image_cache":{},"visual_reuse_counters":{},"visual_preparation_pairs":[],
           "request":{**case,"puzzle_id":"case","candidate_runtime":manifest,"output":str(root/"original_profile"),
                      "scope":"full","device":"cuda","cprofile":False,"fresh_inference_caches":True,"model_policy_sha256":"policy"},
           "comparison":{"status":"passed","equal":True,"archive_equal":True,"archive":{"equal":True,"different_groups":0},
                         "working_pool_payload_equal":True,"assets_equal":True,"excluded_manifest_metadata":["identity","manifest_sha256","provenance"]}}
        return case,manifest,p

    def test_qualification_import_keeps_actual_output_baseline_and_device_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);case,m,p=self.profile(root)
            validate_profile(p,root/"original_profile",case,m,"policy")
            for field,value in (("device","cpu"),("baseline",str(root/"other")),("cprofile",True),("scope","smoke")):
                altered=copy.deepcopy(p);altered["request"][field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):
                    validate_profile(altered,root/"original_profile",case,m,"policy")
            with self.assertRaisesRegex(ValueError,"output binding"):
                validate_profile(p,root/"copied_and_relabelled",case,m,"policy")
            p["comparison"]["archive"]["different_groups"]=1
            with self.assertRaises(ValueError):validate_profile(p,root/"original_profile",case,m,"policy")

    def test_diagnosis_cannot_read_references_before_prediction_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/"request.json",{});atomic_json(root/"progress.json",{"phase":"predictions"})
            with self.assertRaisesRegex(ValueError,"barrier"):diagnose(["--run",str(root)])

    def test_case_selection_does_not_fall_back_or_repeat(self):
        cases=[{"case_id":"puzzle_000174_a"},{"case_id":"puzzle_000341_b"}]
        self.assertEqual(select_cases(cases,["341"]),[cases[1]])
        for wanted in (["999"],["174","174"]):
            with self.assertRaises(ValueError):select_cases(cases,wanted)

    def test_snapshot_escape_cannot_be_used_as_a_verified_consumer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"snapshot").mkdir();(root/"outside").write_text("wrong")
            atomic_json(root/"snapshot/receipt.json",{"files":{"../outside":file_hash(root/"outside")}})
            with self.assertRaisesRegex(ValueError,"snapshot changed"):verify_snapshot(root/"snapshot")
