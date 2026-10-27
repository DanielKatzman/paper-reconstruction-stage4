from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark.common import atomic_json, file_hash, read_json, WORKSPACE
from benchmark.candidate import CASE_ID
from benchmark.experiment import main, make_plan, materialize


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self):
        source, cohort = self.root / "source", self.root / "cohort"
        case = {"case_id": CASE_ID, "path": "cases/"+CASE_ID}
        stage1 = cohort / case["path"] / "stage1"
        stage1.mkdir(parents=True)
        (stage1 / "segmentation_manifest.jsonl").write_text("fixture")
        atomic_json(source / "request.json", {"case_ids":[CASE_ID],"benchmark_sha256":"cohort"})
        atomic_json(source / "progress.json", {"phase":"finished","cases":{CASE_ID:{"status":"predicted"}}})
        output = source / "cases" / CASE_ID
        atomic_json(output / "stage3/bundle.json", {"manifest_sha256":"evidence"})
        atomic_json(output / "stage2/edge_hypotheses_manifest.json", {"status":"complete","truth_free":True,
            "stage1_manifest_sha256":file_hash(stage1 / "segmentation_manifest.jsonl")})
        result = {"input":{"manifest_sha256":"evidence"}, "run_id":"source", "layouts":[{"layout_sha256":"layout"}]}
        return source, cohort, case, result

    def test_consumer_plan_never_invokes_upstream_and_control_is_explicit(self):
        source, cohort, case, result = self.source()
        output = self.root / "run"
        with patch("benchmark.experiment.fixed_case", return_value=(cohort,{"benchmark_sha256":"cohort"},case)), patch("benchmark.experiment.read_result", return_value=result):
            _, _, request = make_plan(source,output,"structural",True)
            self.assertEqual(request["configuration"]["evidence_policy"],"working_plus_diverse_archive.v1")
            self.assertEqual(list(request["commands"]),["stage4"])
            self.assertTrue(request["stage2_reused"] and request["stage3_reused"])
            self.assertFalse(request["fresh_full_pipeline_measurement"])
            _, _, control = make_plan(source,output,"control",False)
            self.assertIn(str(WORKSPACE/"scripts/run_frozen_stage4_control.py"),control["commands"]["stage4"])
            with self.assertRaises(ValueError): make_plan(source,output,"control",True)
            with self.assertRaises(ValueError): make_plan(source,source/"forbidden","structural",False)
        self.assertFalse(output.exists())

    def test_materialization_keeps_original_input_bytes_and_ranking_receipt(self):
        source, cohort, case, result = self.source()
        output = self.root / "run"
        with patch("benchmark.experiment.fixed_case", return_value=(cohort,{"benchmark_sha256":"cohort"},case)), patch("benchmark.experiment.read_result", return_value=result):
            _,_,request=make_plan(source,output,"structural",False)
        ranking=source/"cases"/CASE_ID/"stage3.work/rankings.json"
        atomic_json(ranking,{"signature":"fixture"})
        before=file_hash(source/"cases"/CASE_ID/"stage3/bundle.json")
        receipts=materialize(request,output)
        self.assertEqual(before,file_hash(output/"cases"/CASE_ID/"stage3/bundle.json"))
        self.assertEqual(before,file_hash(source/"cases"/CASE_ID/"stage3/bundle.json"))
        self.assertEqual(file_hash(ranking),file_hash(output/"cases"/CASE_ID/"stage3.work/rankings.json"))
        self.assertGreater(receipts["stage2"]["copied_bytes"],0)

    def request(self):
        return {"schema_version":"stage4.consumer-experiment.v1","case_ids":[CASE_ID],"stage4_config_source":str(WORKSPACE/"configs/structural_development.json"),
                "stage4_config_sha256":file_hash(WORKSPACE/"configs/structural_development.json"),
                "commands":{"stage4":["fake_consumer"]}}

    def test_failed_child_stops_before_diagnostic_phase_is_opened(self):
        output=self.root/"run"
        calls=[]
        def child(command,log,**kwargs):
            calls.append(command)
            if len(calls)==1: return {"returncode":2,"status":"failed"}
            self.assertEqual(read_json(output/"progress.json")["phase"],"diagnostics")
            return {"returncode":0,"status":"complete"}
        with patch("benchmark.experiment.make_plan",return_value=(self.root,{},self.request())), patch("benchmark.experiment.materialize",return_value={}), patch("benchmark.experiment.execute",side_effect=child), patch("benchmark.experiment.summarize"):
            self.assertEqual(main(["--output",str(output)]),1)
        self.assertEqual(len(calls),2)
        self.assertFalse((output/"run.lock").exists())
        self.assertEqual(read_json(output/"progress.json")["cases"][CASE_ID]["status"],"failed")

    def test_interruption_never_starts_offline_evaluation(self):
        output=self.root/"run"
        with patch("benchmark.experiment.make_plan",return_value=(self.root,{},self.request())), patch("benchmark.experiment.materialize",return_value={}), patch("benchmark.experiment.execute",side_effect=KeyboardInterrupt) as child:
            self.assertEqual(main(["--output",str(output)]),130)
        self.assertEqual(child.call_count,1)
        self.assertEqual(read_json(output/"progress.json")["phase"],"interrupted")
        self.assertFalse((output/"run.lock").exists())
