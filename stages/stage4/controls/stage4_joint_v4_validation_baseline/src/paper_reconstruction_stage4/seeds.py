"""Import an authenticated serving primary, never an offline counterfactual."""
from pathlib import Path
import copy
import numpy as np
from .model import State
from .outputs import read_result, file_hash


def attach_initial_result(problem, path):
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    result = read_result(root)
    if result["input"]["manifest_sha256"] != problem.manifest["manifest_sha256"]:
        raise ValueError("Initial serving result belongs to a different evidence bundle")
    layout = result["layouts"][0]
    pieces = layout["pieces"]
    if {p["piece_id"] for p in pieces} != set(problem.pieces):
        raise ValueError("Initial serving inventory differs")
    state = State({p["piece_id"]: np.asarray(p["native_to_component"], float) for p in pieces},
        {p["piece_id"]: p["component_id"] for p in pieces},
        tuple(sorted(j["geometry_id"] for j in layout["selected_joins"])),
        tuple(j["geometry_id"] for j in layout["selected_joins"] if j["tree_join"]),
        {j["geometry_id"]: j["occurrence_id"] for j in layout["selected_joins"]}, layout["objective_score"],
        {j["geometry_id"]: copy.deepcopy(j["active_support"]) for j in layout["selected_joins"] if j.get("active_support")})
    if not set(state.selected) <= {e.id for e in problem.edges}:
        raise ValueError("Initial result selected a geometry absent from incoming evidence")
    problem.initial_states = [state]
    problem.input_summary["initial_result"] = {"run_id": result["run_id"], "result_sha256": file_hash(root/"result.json"),
        "manifest_sha256": file_hash(root/"manifest.json"), "layout_sha256": layout["layout_sha256"]}
