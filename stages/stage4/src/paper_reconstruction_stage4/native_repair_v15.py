"""Opt-in recovery ablations; original V14 behavior remains available."""
from .config import REPAIR_V15_POLICIES
from .native_repair import RepairContext as PreviousRepairContext
from .repair_search import search
from .seam_poses_v15 import generate


class RepairContext(PreviousRepairContext):
    def __init__(self,problem,config,support):
        super().__init__(problem,config,support)
        self.area_bound,self.fair_coverage=REPAIR_V15_POLICIES[config.native_repair_policy]
        self.repair_audit.update(
            component_bound_policy='observed-total-area.v1' if self.area_bound else 'spanning-piece-extent.v1',
            generation_schedule='pair-rotation-strata.v1' if self.fair_coverage else 'pair-major.v1',
            source_selection_changed=False,fit_measurement_changed=False,
            page_format_or_dpi_prior_used=False)

    def component_fits(self,state):
        # A true subset can be enclosed by the complete page's rectangle.
        # The observed total foreground area supplies its area, with the same
        # existing slack. No side length or aspect ratio is assumed.
        return self.component_bounds(state)['valid']

    def generate_proposals(self,deadline,maximum_checks,sources):
        if not self.fair_coverage:
            return super().generate_proposals(deadline,maximum_checks,sources)
        return generate(self.problem,self.config,deadline,maximum_checks=maximum_checks,
                        frontiers=[source.components() for source,_,_ in sources])

    def merge_search(self,source,proposals,deadline,maximum_checks):
        complete,audit=search(self,source,proposals,deadline,maximum_checks=maximum_checks,
            beam=self.config.repair_beam,component_bound=self.component_fits if self.area_bound else None)
        audit.update(policy='native-seam-component-beam.v2',
            component_bound_policy=self.repair_audit['component_bound_policy'],
            provisional_extent_enforced=not self.area_bound,
            full_page_and_native_verification_required=True)
        return complete,audit

    def augment_finalists(self,states,deadline):
        result=super().augment_finalists(states,deadline)
        generation=self.repair_audit.get('native_generation') or {}
        merge_exhausted=any(row['budget_abstained'] for row in self.repair_audit['merge_searches'])
        self.repair_audit.update(
            generation_budget_abstained=bool(generation.get('budget_abstained',False)),
            merge_budget_abstained=merge_exhausted,
            route_budget_abstained=self.repair_audit['budget_abstained'])
        # Keep component flags explicit and aggregate them for the summary.
        self.repair_audit['budget_abstained']=bool(self.repair_audit['budget_abstained'] or
            generation.get('budget_abstained',False) or merge_exhausted or
            self.repair_audit.get('scheduled_sources_not_attempted',0))
        return result
