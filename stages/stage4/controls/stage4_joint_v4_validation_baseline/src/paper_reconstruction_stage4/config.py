"""Explicit, versioned development policies. Scores are not probabilities."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class SolverConfig:
    solver: str = "partial_beam_v1"
    objective_version: str = "unique-contact-minus-join-cost.v1"
    beam_width: int = 32
    max_proposals: int = 100_000
    max_seconds: float = 60.0
    max_input_bytes: int = 1_073_741_824
    max_working_occurrences: int = 150_000
    seed: int = 9001
    join_cost: float = 0.55
    collision_grid: int = 24
    maximum_assembly_collision: float = 0.08
    final_collision_samples: int = 50_000
    edge_erosion_px: int = 1
    cycle_tolerance_px: float = 8.0
    contact_tolerance_px: float = 12.0
    support_endpoint_tolerance_px: float = 2.0
    support_overlap_fraction: float = 0.05
    max_assignment_nodes: int = 10_000
    minimum_denominator_ratio: float = 0.15
    minimum_local_scale: float = 0.15
    maximum_local_scale: float = 7.0
    maximum_local_anisotropy: float = 8.0
    maximum_cache_entries: int = 8192
    portfolio_size: int = 3
    loop_proposals: bool = False
    loop_alternatives_per_pair: int = 3
    max_loop_checks: int = 2000
    max_loop_seeds: int = 16
    repair_rounds: int = 0
    max_repair_checks: int = 1000
    refine: bool = False
    refine_max_nfev: int = 50
    refine_translation_px: float = 4.0
    refine_rotation_degrees: float = 1.0
    refine_log_scale: float = 0.01
    refine_regularization: float = 0.1
    refine_seconds: float = 20.0
    evidence_policy: str = "working_pool"
    archive_extra_per_pair: int = 8
    archive_candidates_per_pair: int = 128
    archive_score_margin: float = 0.2
    structural_angle_weight: float = 0.1
    structural_scale_weight: float = 0.12
    structural_compactness_weight: float = 0.15
    structural_corroboration_weight: float = 0.12
    structural_boundary_weight: float = 0.12
    structural_appearance_weight: float = 0.04
    structural_underconstrained_cost: float = 0.03
    structural_angle_sigma_degrees: float = 12.0
    structural_scale_sigma: float = 0.15
    structural_corroboration_fraction: float = 0.025
    structural_alternatives_per_pair: int = 12
    structural_repair_removals: int = 2
    structural_repair_width: int = 4
    structural_refine_samples: int = 33
    support_policy: str = "raw-occurrence.v1"
    joint_alternatives_per_pair: int = 24
    joint_trials: int = 32
    joint_region_joins: int = 6
    joint_rebuild_width: int = 6
    joint_wall_seconds: float = 120.0
    joint_finalize_seconds: float = 10.0
    active_distance_px: float = 6.0
    active_minimum_length_px: float = 32.0
    active_minimum_fraction: float = 0.2
    active_max_options: int = 4
    active_trim_weight: float = 0.04
    joint_refinement: bool = False
    joint_refinement_rounds: int = 2

    def validate(self):
        if not isinstance(self.solver, str) or self.solver not in {"partial_beam_v1", "legacy_forest_control_v1", "structural_beam_v2", "joint_beam_v3"}:
            raise ValueError("Unknown solver")
        if self.objective_version not in {"unique-contact-minus-join-cost.v1", "observable-structure.v2", "observable-joint.v3", "observable-joint.v4"}:
            raise ValueError("Unknown objective version")
        structural = self.solver in {"structural_beam_v2", "joint_beam_v3"}
        if structural != (self.objective_version in {"observable-structure.v2", "observable-joint.v3", "observable-joint.v4"}):
            raise ValueError("Structural solver and objective must be selected together")
        if self.evidence_policy not in {"working_pool", "working_plus_diverse_archive.v1"}:
            raise ValueError("Unknown consumer evidence policy")
        if self.evidence_policy != "working_pool" and not structural:
            raise ValueError("Archive expansion is opt-in for the structural solver")
        if self.support_policy not in {"raw-occurrence.v1", "active-subarc.v1"}:
            raise ValueError("Unknown support policy")
        if self.objective_version in {"observable-joint.v3", "observable-joint.v4"} and self.solver != "joint_beam_v3":
            raise ValueError("Joint objective requires joint solver")
        if self.support_policy != "raw-occurrence.v1" and (self.solver != "joint_beam_v3" or self.objective_version not in {"observable-joint.v3", "observable-joint.v4"}):
            raise ValueError("Active support requires the versioned joint objective and solver")
        if self.joint_refinement and self.solver != "joint_beam_v3":
            raise ValueError("Joint refinement requires joint solver")
        if self.refine and self.support_policy == "active-subarc.v1" and not self.joint_refinement:
            raise ValueError("Active support refinement requires the joint optimizer")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(item.default, bool):
                if type(value) is not bool:
                    raise ValueError(f"{item.name} must be boolean")
            elif isinstance(item.default, int):
                if type(value) is not int or value < (0 if item.name in {"seed", "edge_erosion_px", "repair_rounds"} else 1):
                    raise ValueError(f"Invalid integer {item.name}")
            elif isinstance(item.default, float):
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid numeric {item.name}")
        if not 0 < self.max_seconds or not 0 < self.refine_seconds:
            raise ValueError("Time budgets must be positive")
        if self.cycle_tolerance_px <= 0 or self.contact_tolerance_px <= 0:
            raise ValueError("Pose/contact tolerances must be positive")
        if not 0 <= self.maximum_assembly_collision <= 1 or not 0 <= self.support_overlap_fraction <= 1:
            raise ValueError("Fractions must be in [0,1]")
        if not 0 < self.minimum_denominator_ratio <= 1 or not 0 < self.minimum_local_scale <= self.maximum_local_scale:
            raise ValueError("Invalid warp limits")
        if self.collision_grid < 2 or self.maximum_local_anisotropy < 1 or self.final_collision_samples < 4:
            raise ValueError("Invalid feasibility resolution")
        if self.portfolio_size > self.beam_width:
            raise ValueError("portfolio_size exceeds beam_width")
        if not 0 < self.structural_angle_sigma_degrees <= 90 or self.structural_scale_sigma <= 0 or self.structural_corroboration_fraction <= 0:
            raise ValueError("Structural uncertainty scales must be positive and bounded")
        if self.structural_repair_removals > 2 or not 3 <= self.structural_refine_samples <= 65:
            raise ValueError("Structural repair/refinement resource limits exceeded")
        if not 0 < self.active_minimum_fraction <= 1 or not 0 < self.active_distance_px <= self.contact_tolerance_px:
            raise ValueError("Invalid active support distance/coverage")
        if self.active_minimum_length_px <= 0 or self.joint_wall_seconds <= self.joint_finalize_seconds or self.joint_finalize_seconds <= 0:
            raise ValueError("Invalid joint serving budget or active support length")
        if self.joint_region_joins > 12 or self.joint_alternatives_per_pair > 64 or self.active_max_options > 16 or self.joint_refinement_rounds > 4:
            raise ValueError("Joint resource limits exceeded")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**value).validate()
