"""The v2 observable score with an explicit active-support coverage cost."""
from .structural import StructuralScorer
from .config import JOINT_OBJECTIVES,ATTACHMENT_OBJECTIVES,CONTINUATION_OBJECTIVES


class JointScorer(StructuralScorer):
    def evaluate(self, state):
        base = super().evaluate(state)
        if self.config.objective_version not in JOINT_OBJECTIVES:
            return base
        trimming = self.config.active_trim_weight*sum(1.-min(a["coverage"]) for a in state.active_support.values())
        result = {**base, "score": base["score"]-trimming, "penalties": {**base["penalties"], "active_trimming": trimming},
                  "active_support_joins": len(state.active_support)}
        if self.config.objective_version in ATTACHMENT_OBJECTIVES:
            from .attachments import evaluate
            attachment = evaluate(self,state,base)
            result = {**result,"score":result['score']-attachment['cost'],
                      "penalties":{**result['penalties'],'attachment_uncertainty':attachment['cost']},
                      "attachment_evidence":attachment}
        if self.config.objective_version in CONTINUATION_OBJECTIVES:
            from .continuation import evaluate
            continuation=evaluate(self,state,result['attachment_evidence'])
            result={**result,'score':result['score']-continuation['cost'],
                'penalties':{**result['penalties'],'local_continuation':continuation['cost']},
                'continuation_evidence':continuation}
        return result
