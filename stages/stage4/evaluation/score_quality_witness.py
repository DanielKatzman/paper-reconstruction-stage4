"""Offline only: rescore a published primary and an existing truth-guided witness."""
from pathlib import Path
import numpy as np
from benchmark.common import file_hash,read_json
from benchmark.qualified_v3 import require
from benchmark.quality_experiments import checked_integration
from paper_reconstruction_stage4.config import SolverConfig
from paper_reconstruction_stage4.inputs import load_problem
from paper_reconstruction_stage4.model import State
from paper_reconstruction_stage4.outputs import implementation_identity,read_result
from paper_reconstruction_stage4.feasibility import Feasibility
from paper_reconstruction_stage4.joint_score import JointScorer
from paper_reconstruction_stage4.active_support import ActiveSupportIndex


def layout_state(layout):
    joins=layout["selected_joins"]
    return State({p["piece_id"]:np.asarray(p["native_to_component"],float) for p in layout["pieces"]},
                 {p["piece_id"]:p["component_id"] for p in layout["pieces"]},
                 tuple(sorted(j["geometry_id"] for j in joins)),tuple(j["geometry_id"] for j in joins if j["tree_join"]),
                 {j["geometry_id"]:j["occurrence_id"] for j in joins},layout["objective_score"],
                 {j["geometry_id"]:j["active_support"] for j in joins if j.get("active_support")})


def compare(integration,case_id,coverage,configs):
    root=Path(integration).resolve();request=checked_integration(root)
    case=next(c for c in request["import"]["cases"] if c["case_id"]==case_id)
    diagnostic=read_json(coverage)
    require(diagnostic["schema_version"]=="stage4.offline-pose-coverage.v1" and diagnostic["evaluation_truth_opened"] is True,
            "An explicitly offline reference-guided coverage diagnostic is required")
    original=read_result(case["controls"]["robust"]["path"])
    require(diagnostic["case_id"]==case_id and diagnostic["source_run_id"]==original["run_id"] and
            diagnostic["source_result_sha256"]==file_hash(Path(case["controls"]["robust"]["path"])/"result.json"),"Witness source binding differs")
    witness=diagnostic["families"]["working"]["oracle"]["best_reference_ranked_witness"]
    require(witness is not None,"No existing working-pool witness")
    primary=read_result(root/"cases"/case_id/"coarse/stage4")
    rows=[]
    for path in configs:
        config=SolverConfig.from_dict(read_json(path));problem=load_problem(root/"cases"/case_id/"stage3",config)
        support=ActiveSupportIndex(problem,config);feasibility=Feasibility(problem,config);scorer=JointScorer(problem,config,support)
        states=[layout_state(primary["layouts"][0]),layout_state(witness["layout"])]
        terms=[]
        for state in states:
            require(all(e in scorer.edges for e in state.selected),"Witness references a geometry absent from imported evidence")
            assignments,reason,nodes=support.assign(state.selected,state.poses,scorer.edges)
            require(assignments is not None,"Diagnostic state support incompatibility: "+reason)
            state.assignments=assignments;support.bind(state)
            require(feasibility.state(state,final=True)["valid"],"Diagnostic state failed physical guards")
            terms.append(scorer.evaluate(state))
        rows.append({"configuration":str(path),"configuration_sha256":file_hash(Path(path)),"objective":config.objective_version,
                     "primary":terms[0],"reference_guided_existing_witness":terms[1],"witness_preferred":terms[1]["score"]>terms[0]["score"]})
    return {"schema_version":"stage4.offline-witness-score.v1","prediction":False,"evaluation_truth_opened":True,
            "case_id":case_id,"diagnostic_sha256":file_hash(Path(coverage)),"consumer_implementation":implementation_identity(),
            "serving_seed_created":False,"new_pose_search_performed":False,"states_reassigned_to_valid_supports":True,"comparisons":rows}
