"""Explicit, versioned development policies. Scores are not probabilities."""
from dataclasses import asdict, dataclass, fields
import math

CONTINUATION_OBJECTIVES = frozenset({'observable-joint.v7-continuation', 'observable-joint.v8-continuation'})
JOINT_OBJECTIVES = frozenset({"observable-joint.v3", "observable-joint.v4", "observable-joint.v5-perpendicular", "observable-joint.v5-scale", "observable-joint.v6-attachment", *CONTINUATION_OBJECTIVES})
MEAN_COMPONENT_OBJECTIVES = JOINT_OBJECTIVES - {"observable-joint.v3"}
PERPENDICULAR_OBJECTIVES = frozenset({"observable-joint.v5-perpendicular", "observable-joint.v5-scale", "observable-joint.v6-attachment", *CONTINUATION_OBJECTIVES})
SCALE_OBJECTIVES = frozenset({"observable-joint.v5-scale", "observable-joint.v6-attachment", *CONTINUATION_OBJECTIVES})
ATTACHMENT_OBJECTIVES = frozenset({"observable-joint.v6-attachment", *CONTINUATION_OBJECTIVES})
LOOKAHEAD_POLICY = 'component-beam.v3-lookahead'
LOOKAHEAD_DEFAULTS = {'joint_ordering_occurrences':3, 'joint_component_pool_factor':3,
                      'joint_lookahead_checks':2, 'joint_lookahead_proposals':1200}
RETAINED_POLICY = 'cold-portfolio.v2-retained'
RETENTION_DEFAULTS = {'joint_retention_route_proposals':20000,'joint_finalist_policy':'late.v1',
    'joint_incumbent_limit':6,'joint_final_check_limit':128,'joint_early_pair_checks':256}
GEOMETRY_SELECTION = 'geometry-gated-topk.v1'
AMBIGUITY_SELECTION = 'ambiguity-aware-geometry-topk.v1'
GEOMETRY_SELECTIONS = frozenset({GEOMETRY_SELECTION, AMBIGUITY_SELECTION})
GEOMETRY_DEFAULTS = {'page_top_k':50, 'page_geometry_k':50, 'page_diversity_k':12,
                     'page_final_candidate_limit':128, 'page_component_area_slack':.10}
AMBIGUITY_DEFAULTS = {'ambiguity_risk_weight':.10, 'ambiguity_deletion_checks':24,
                      'ambiguity_partial_slots':16, 'ambiguity_measurement_limit':12}
CLOSURE_POLICY = 'native-component-closure.v1'
CLOSURE_DEFAULTS = {'closure_source_limit':2, 'closure_proposal_limit':256,
                    'closure_fine_limit':16, 'closure_seconds':12.0}
REPAIR_POLICY = 'native-seam-repair.v1'
REPAIR_V16_POLICY = 'native-seam-repair.portfolio.v3'
REPAIR_V17_POLICY = 'native-seam-repair.native-graph.v4'
REPAIR_V18_POLICY = 'native-seam-repair.geometry-until-valid.v5'
STRAIGHT_DEFERRED_GROWTH = 'near-straight-deferred.v1'
REPAIR_V15_POLICIES = {
    'native-seam-repair.area-bound.v2': (True, False),
    'native-seam-repair.fair-coverage.v2': (False, True),
    'native-seam-repair.v2': (True, True),
}
REPAIR_DEFAULTS = {'repair_source_limit':3,'repair_native_checks':768,'repair_merge_checks':8000,
    'repair_beam':24,'repair_seconds':45.0,'repair_complete_checks':12}
COLD_POLICIES = frozenset({'cold-portfolio.v1',RETAINED_POLICY})


def support_aware_search(config):
    return (config.joint_search_policy == LOOKAHEAD_POLICY or
            config.joint_search_policy in COLD_POLICIES and config.joint_portfolio_component_policy == LOOKAHEAD_POLICY)


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
    joint_search_policy: str = "diverse-constructions.v1"
    scale_coherence_weight: float = 0.12
    scale_coherence_sigma: float = 0.30
    scale_coherence_min_confidence: float = 0.40
    joint_portfolio_fraction: float = 0.45
    joint_portfolio_component_policy: str = "component-beam.v2"
    attachment_uncertainty_cost: float = 0.10
    attachment_axis_weight: float = 0.12
    attachment_corroboration_discount: float = 0.75
    attachment_deletion_checks: int = 24
    joint_execution_policy: str = "baseline.v1"
    continuation_weight: float = 1.25
    continuation_min_coverage: float = 0.60
    continuation_samples: int = 65
    continuation_image_side: int = 768
    joint_ordering_occurrences: int = 3
    joint_component_pool_factor: int = 3
    joint_lookahead_checks: int = 2
    joint_lookahead_proposals: int = 1200
    joint_retention_route_proposals: int = 20000
    joint_finalist_policy: str = 'late.v1'
    joint_incumbent_limit: int = 6
    joint_final_check_limit: int = 128
    joint_early_pair_checks: int = 256
    page_policy: str = 'disabled.v1'
    page_selection_policy: str = 'utility.v1'
    recovery_region_limit: int = 6
    recovery_poses_per_region: int = 32
    page_top_k: int = 50
    page_geometry_k: int = 50
    page_diversity_k: int = 12
    page_final_candidate_limit: int = 128
    page_component_area_slack: float = .10
    ambiguity_risk_weight: float = .10
    ambiguity_deletion_checks: int = 24
    ambiguity_partial_slots: int = 16
    ambiguity_measurement_limit: int = 12
    component_closure_policy: str = 'disabled.v1'
    closure_source_limit: int = 2
    closure_proposal_limit: int = 256
    closure_fine_limit: int = 16
    closure_seconds: float = 12.0
    native_repair_policy: str = 'disabled.v1'
    native_growth_policy: str = 'all-modes.v1'
    native_expand_marginal_geometry: bool = False
    native_cut_repair: bool = False
    repair_source_limit: int = 3
    repair_native_checks: int = 768
    repair_merge_checks: int = 8000
    repair_beam: int = 24
    repair_seconds: float = 45.0
    repair_complete_checks: int = 12

    def validate(self):
        if self.native_cut_repair and (self.native_repair_policy != REPAIR_V18_POLICY or self.native_growth_policy != STRAIGHT_DEFERRED_GROWTH):
            raise ValueError('Resumable cut repair requires deferred-straight geometry recovery')
        if self.native_growth_policy not in {'all-modes.v1', STRAIGHT_DEFERRED_GROWTH}:
            raise ValueError('Unknown native growth policy')
        if self.native_growth_policy != 'all-modes.v1' and self.native_repair_policy != REPAIR_V18_POLICY:
            raise ValueError('Deferred straight growth requires geometry-driven native recovery')
        if self.native_expand_marginal_geometry and self.native_growth_policy != STRAIGHT_DEFERRED_GROWTH:
            raise ValueError('Geometry expansion requires deferred-straight native growth')
        if self.native_repair_policy not in {'disabled.v1',REPAIR_POLICY,REPAIR_V16_POLICY,REPAIR_V17_POLICY,REPAIR_V18_POLICY,*REPAIR_V15_POLICIES}:
            raise ValueError('Unknown native repair policy')
        if self.native_repair_policy == REPAIR_V17_POLICY and self.repair_source_limit < 2:
            raise ValueError('Native graph recovery reserves a separate independent source')
        if self.native_repair_policy!='disabled.v1' and self.component_closure_policy!=CLOSURE_POLICY:
            raise ValueError('Native seam repair requires the versioned component closure policy')
        if self.native_repair_policy=='disabled.v1' and any(getattr(self,k)!=v for k,v in REPAIR_DEFAULTS.items()):
            raise ValueError('Native repair limits require the repair policy')
        if not 1<=self.repair_source_limit<=4 or not 1<=self.repair_native_checks<=1024 or not 1<=self.repair_merge_checks<=10000 or not 4<=self.repair_beam<=64 or not 0<self.repair_seconds<=90 or not 1<=self.repair_complete_checks<=24:
            raise ValueError('Invalid bounded native repair envelope')
        if self.component_closure_policy not in {'disabled.v1', CLOSURE_POLICY}:
            raise ValueError('Unknown component closure policy')
        if self.component_closure_policy == CLOSURE_POLICY and (self.page_selection_policy != AMBIGUITY_SELECTION or self.support_policy != 'active-subarc.v1'):
            raise ValueError('Native component closure requires ambiguity selection and active native support')
        if self.component_closure_policy == 'disabled.v1' and any(getattr(self,k)!=v for k,v in CLOSURE_DEFAULTS.items()):
            raise ValueError('Closure limits require the versioned native component closure policy')
        if not 1 <= self.closure_source_limit <= 4 or not 1 <= self.closure_proposal_limit <= 512 or not 1 <= self.closure_fine_limit <= 32 or not 0 < self.closure_seconds <= 30:
            raise ValueError('Invalid bounded component closure envelope')
        if self.page_policy not in {'disabled.v1','shared-scale-complete-rectangle.v1'}:
            raise ValueError('Unknown page policy')
        if self.page_selection_policy not in {'utility.v1','supported-closure-first.v1',*GEOMETRY_SELECTIONS}:
            raise ValueError('Unknown page selection policy')
        if self.page_policy == 'disabled.v1' and self.page_selection_policy != 'utility.v1':
            raise ValueError('Page selection requires the explicit page profile')
        if self.page_policy != 'disabled.v1' and (self.solver != 'joint_beam_v3' or self.refine or self.joint_refinement):
            raise ValueError('Complete-page policy requires joint solver with refinement disabled')
        if not 1 <= self.recovery_region_limit <= 6 or not 1 <= self.recovery_poses_per_region <= 32:
            raise ValueError('Recovery request exceeds one-round, six-region/32-pose envelope')
        if self.page_policy == 'disabled.v1' and (self.recovery_region_limit != 6 or self.recovery_poses_per_region != 32):
            raise ValueError('Recovery limits require the page policy')
        if self.page_selection_policy not in GEOMETRY_SELECTIONS and any(getattr(self,k)!=v for k,v in GEOMETRY_DEFAULTS.items()):
            raise ValueError('Geometry candidate limits require the versioned geometry selection')
        if self.page_selection_policy != AMBIGUITY_SELECTION and any(getattr(self,k)!=v for k,v in AMBIGUITY_DEFAULTS.items()):
            raise ValueError('Ambiguity parameters require the versioned ambiguity selection')
        if not 0 < self.ambiguity_risk_weight <= .30 or not 1 <= self.ambiguity_deletion_checks <= 64:
            raise ValueError('Invalid ambiguity risk or deletion budget')
        if not 8 <= self.ambiguity_partial_slots <= 32 or not 1 <= self.ambiguity_measurement_limit <= 32:
            raise ValueError('Invalid ambiguity partial/measurement budget')
        for key,default in GEOMETRY_DEFAULTS.items():
            value=getattr(self,key)
            if isinstance(default,int) and type(value) is not int:
                raise ValueError('Invalid geometry integer '+key)
            if isinstance(default,float) and (type(value) not in (int,float) or not math.isfinite(value)):
                raise ValueError('Invalid geometry numeric '+key)
        if not 1 <= self.page_top_k <= 100 or not 1 <= self.page_geometry_k <= 100 or not 1 <= self.page_diversity_k <= 32:
            raise ValueError('Geometry candidate bank limits exceeded')
        partial_slots = self.ambiguity_partial_slots if self.page_selection_policy == AMBIGUITY_SELECTION else 8
        if not self.page_top_k+self.page_geometry_k+self.page_diversity_k+partial_slots <= self.page_final_candidate_limit <= 256:
            raise ValueError('Final geometry check cap must cover both top-k banks, diversity and partial slots')
        if not .02 <= self.page_component_area_slack <= .20:
            raise ValueError('Invalid complete-page area tolerance')
        if not isinstance(self.solver, str) or self.solver not in {"partial_beam_v1", "legacy_forest_control_v1", "structural_beam_v2", "joint_beam_v3"}:
            raise ValueError("Unknown solver")
        if self.objective_version not in {"unique-contact-minus-join-cost.v1", "observable-structure.v2", *JOINT_OBJECTIVES}:
            raise ValueError("Unknown objective version")
        structural = self.solver in {"structural_beam_v2", "joint_beam_v3"}
        if structural != (self.objective_version in {"observable-structure.v2", *JOINT_OBJECTIVES}):
            raise ValueError("Structural solver and objective must be selected together")
        if self.evidence_policy not in {"working_pool", "working_plus_diverse_archive.v1"}:
            raise ValueError("Unknown consumer evidence policy")
        if self.evidence_policy != "working_pool" and not structural:
            raise ValueError("Archive expansion is opt-in for the structural solver")
        if self.support_policy not in {"raw-occurrence.v1", "active-subarc.v1"}:
            raise ValueError("Unknown support policy")
        if self.objective_version in JOINT_OBJECTIVES and self.solver != "joint_beam_v3":
            raise ValueError("Joint objective requires joint solver")
        if self.support_policy != "raw-occurrence.v1" and (self.solver != "joint_beam_v3" or self.objective_version not in JOINT_OBJECTIVES):
            raise ValueError("Active support requires the versioned joint objective and solver")
        if self.joint_refinement and self.solver != "joint_beam_v3":
            raise ValueError("Joint refinement requires joint solver")
        if self.joint_search_policy not in {"diverse-constructions.v1", "component-beam.v1", "component-beam.v2", LOOKAHEAD_POLICY, *COLD_POLICIES}:
            raise ValueError("Unknown joint search policy")
        if self.joint_search_policy != "diverse-constructions.v1" and self.solver != "joint_beam_v3":
            raise ValueError("Component beam requires the joint solver")
        if self.joint_execution_policy not in {"baseline.v1", "exact-reuse.v1"}:
            raise ValueError("Unknown joint execution policy")
        if self.joint_execution_policy != "baseline.v1" and self.solver != "joint_beam_v3":
            raise ValueError("Execution reuse requires the joint solver")
        if self.joint_portfolio_component_policy not in {"component-beam.v1", "component-beam.v2", LOOKAHEAD_POLICY} or not .1 <= self.joint_portfolio_fraction <= .8:
            raise ValueError("Invalid cold portfolio allocation/policy")
        if self.joint_search_policy not in COLD_POLICIES and (self.joint_portfolio_fraction != .45 or self.joint_portfolio_component_policy != "component-beam.v2"):
            raise ValueError("Portfolio parameters require the cold portfolio policy")
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
        if self.scale_coherence_sigma <= 0 or not 0 < self.scale_coherence_min_confidence <= 1 or not 0 <= self.scale_coherence_weight <= 1:
            raise ValueError("Invalid scale coherence policy")
        if self.objective_version not in SCALE_OBJECTIVES and any(getattr(self,k) != v for k,v in {
                "scale_coherence_weight":.12,"scale_coherence_sigma":.30,"scale_coherence_min_confidence":.40}.items()):
            raise ValueError("Scale coherence parameters require the versioned scale objective")
        if not 0 <= self.attachment_uncertainty_cost <= 1 or not 0 <= self.attachment_axis_weight <= 1 or not 0 <= self.attachment_corroboration_discount <= 1 or self.attachment_deletion_checks > 128:
            raise ValueError("Invalid attachment uncertainty policy")
        if self.objective_version not in ATTACHMENT_OBJECTIVES and any(getattr(self,k) != v for k,v in {
                "attachment_uncertainty_cost":.10,"attachment_axis_weight":.12,"attachment_corroboration_discount":.75,"attachment_deletion_checks":24}.items()):
            raise ValueError("Attachment parameters require the versioned attachment objective")
        if not 0 <= self.continuation_weight <= 2 or not 0 < self.continuation_min_coverage <= 1:
            raise ValueError("Invalid continuation weight/coverage")
        if not 17 <= self.continuation_samples <= 129 or not 128 <= self.continuation_image_side <= 1024:
            raise ValueError("Continuation sampling/image resource limits exceeded")
        if self.objective_version not in CONTINUATION_OBJECTIVES and any(getattr(self,k) != v for k,v in {
                'continuation_weight':1.25,'continuation_min_coverage':.60,'continuation_samples':65,'continuation_image_side':768}.items()):
            raise ValueError("Continuation parameters require the versioned continuation objective")
        if self.joint_ordering_occurrences > 8 or self.joint_component_pool_factor > 6 or self.joint_lookahead_checks > 8 or self.joint_lookahead_proposals > 4000:
            raise ValueError('Support-aware search resource limits exceeded')
        if not support_aware_search(self) and any(getattr(self,k) != v for k,v in LOOKAHEAD_DEFAULTS.items()):
            raise ValueError('Lookahead parameters require the versioned support-aware search')
        if self.joint_finalist_policy not in {'late.v1','checked-incumbents.v1'}:
            raise ValueError('Unknown finalist policy')
        if self.joint_retention_route_proposals > 100000 or self.joint_incumbent_limit > 16 or self.joint_final_check_limit > 512 or self.joint_early_pair_checks > 2048:
            raise ValueError('Retention resource limits exceeded')
        if self.joint_search_policy != RETAINED_POLICY and any(getattr(self,k) != v for k,v in RETENTION_DEFAULTS.items()):
            raise ValueError('Retention parameters require the retained portfolio')
        if self.joint_search_policy == RETAINED_POLICY and (self.objective_version != 'observable-joint.v8-continuation' or self.joint_portfolio_component_policy != LOOKAHEAD_POLICY or self.joint_execution_policy != 'exact-reuse.v1'):
            raise ValueError('Retained portfolio requires v8 scoring, lookahead and exact reuse')
        return self

    def to_dict(self):
        value = asdict(self)
        if self.native_growth_policy == 'all-modes.v1':
            value.pop('native_growth_policy')
        if not self.native_expand_marginal_geometry:
            value.pop('native_expand_marginal_geometry')
        if not self.native_cut_repair:
            value.pop('native_cut_repair')
        if self.native_repair_policy=='disabled.v1':
            value.pop('native_repair_policy')
            for key in REPAIR_DEFAULTS:value.pop(key)
        if self.component_closure_policy == 'disabled.v1':
            value.pop('component_closure_policy')
            for key in CLOSURE_DEFAULTS:value.pop(key)
        if self.page_selection_policy not in GEOMETRY_SELECTIONS:
            for key in GEOMETRY_DEFAULTS:
                value.pop(key)
        if self.page_selection_policy != AMBIGUITY_SELECTION:
            for key in AMBIGUITY_DEFAULTS:
                value.pop(key)
        if self.page_selection_policy == 'utility.v1':
            value.pop('page_selection_policy')
        if self.page_policy == 'disabled.v1':
            value.pop('page_policy')
            value.pop('recovery_region_limit')
            value.pop('recovery_poses_per_region')
        if self.joint_execution_policy == "baseline.v1":
            value.pop("joint_execution_policy")
        # Preserve the exact normalized configuration of historical controls.
        if self.joint_search_policy == "diverse-constructions.v1":
            value.pop("joint_search_policy")
        if self.objective_version not in SCALE_OBJECTIVES:
            for key in ("scale_coherence_weight","scale_coherence_sigma","scale_coherence_min_confidence"):
                value.pop(key)
        if self.joint_search_policy not in COLD_POLICIES:
            for key in ("joint_portfolio_fraction", "joint_portfolio_component_policy"):
                value.pop(key)
        if self.objective_version not in ATTACHMENT_OBJECTIVES:
            for key in ("attachment_uncertainty_cost", "attachment_axis_weight", "attachment_corroboration_discount", "attachment_deletion_checks"):
                value.pop(key)
        if self.objective_version not in CONTINUATION_OBJECTIVES:
            for key in ('continuation_weight','continuation_min_coverage','continuation_samples','continuation_image_side'):
                value.pop(key)
        if not support_aware_search(self):
            for key in LOOKAHEAD_DEFAULTS:
                value.pop(key)
        if self.joint_search_policy != RETAINED_POLICY:
            for key in RETENTION_DEFAULTS:
                value.pop(key)
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**value).validate()
