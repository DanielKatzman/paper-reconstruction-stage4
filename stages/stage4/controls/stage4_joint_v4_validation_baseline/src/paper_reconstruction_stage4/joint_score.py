"""The v2 observable score with an explicit active-support coverage cost."""
from .structural import StructuralScorer


class JointScorer(StructuralScorer):
    def evaluate(self, state):
        base = super().evaluate(state)
        if self.config.objective_version not in {"observable-joint.v3", "observable-joint.v4"}:
            return base
        trimming = self.config.active_trim_weight*sum(1.-min(a["coverage"]) for a in state.active_support.values())
        return {**base, "score": base["score"]-trimming, "penalties": {**base["penalties"], "active_trimming": trimming},
                "active_support_joins": len(state.active_support)}
