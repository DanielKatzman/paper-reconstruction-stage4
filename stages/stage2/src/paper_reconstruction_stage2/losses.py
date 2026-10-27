from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import torch
    import torch.nn.functional as functional
    from torch import Tensor, nn
except ImportError:
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None
    functional = None


@dataclass(slots=True)
class BoundaryRoleLossConfig:
    transition_weight: float = 0.30
    smoothness_weight: float = 0.05
    transition_focal_gamma: float = 2.0
    transition_positive_alpha: float = 0.75
    confidence_floor: float = 0.25
    center_emphasis: float = 2.0


if nn is not None:

    def transition_targets(role_targets: Tensor) -> Tensor:
        if role_targets.ndim != 3 or role_targets.shape[-1] != 3:
            raise ValueError("role_targets must have shape [batch, sequence, 3]")
        uncertainty = role_targets[..., 2]
        difference = 0.5 * torch.abs(
            role_targets[:, 1:, :] - role_targets[:, :-1, :]
        ).sum(dim=-1)
        transition = uncertainty.clone()
        transition[:, 1:] = torch.maximum(transition[:, 1:], difference)
        transition[:, :-1] = torch.maximum(transition[:, :-1], difference)
        return transition.clamp(0.0, 1.0)


    class BoundaryRoleLoss(nn.Module):
        def __init__(self, config: BoundaryRoleLossConfig | None = None):
            super().__init__()
            self.config = config or BoundaryRoleLossConfig()

        def forward(
            self,
            outputs: dict[str, Tensor],
            role_targets: Tensor,
            label_confidence: Tensor,
        ) -> dict[str, Tensor]:
            role_logits = outputs["role_logits"]
            transition_logits = outputs["transition_logits"]
            if role_logits.shape != role_targets.shape:
                raise ValueError("role logits and targets must have identical shapes")
            sequence_length = role_targets.shape[1]
            positions = torch.linspace(
                -1.0,
                1.0,
                sequence_length,
                device=role_targets.device,
                dtype=role_targets.dtype,
            )
            center_weights = 1.0 + self.config.center_emphasis * torch.exp(
                -0.5 * torch.square(positions / 0.28)
            )
            center_weights = center_weights / center_weights.mean()
            confidence_weights = self.config.confidence_floor + (
                1.0 - self.config.confidence_floor
            ) * label_confidence
            point_weights = confidence_weights * center_weights.unsqueeze(0)

            log_probabilities = functional.log_softmax(role_logits, dim=-1)
            role_per_point = -(role_targets * log_probabilities).sum(dim=-1)
            role_loss = (role_per_point * point_weights).sum() / point_weights.sum().clamp_min(
                1e-6
            )

            transitions = transition_targets(role_targets)
            probabilities = torch.sigmoid(transition_logits)
            binary_ce = functional.binary_cross_entropy_with_logits(
                transition_logits, transitions, reduction="none"
            )
            probability_true = probabilities * transitions + (1.0 - probabilities) * (
                1.0 - transitions
            )
            alpha = self.config.transition_positive_alpha * transitions + (
                1.0 - self.config.transition_positive_alpha
            ) * (1.0 - transitions)
            focal = alpha * torch.pow(1.0 - probability_true, self.config.transition_focal_gamma)
            transition_loss = (binary_ce * focal * point_weights).sum() / point_weights.sum().clamp_min(
                1e-6
            )

            role_probabilities = torch.softmax(role_logits, dim=-1)
            adjacent_difference = torch.square(
                role_probabilities[:, 1:, :] - role_probabilities[:, :-1, :]
            ).sum(dim=-1)
            stable_weight = 1.0 - torch.maximum(
                transitions[:, 1:], transitions[:, :-1]
            )
            smoothness_loss = (
                adjacent_difference * stable_weight
            ).sum() / stable_weight.sum().clamp_min(1e-6)

            total = (
                role_loss
                + self.config.transition_weight * transition_loss
                + self.config.smoothness_weight * smoothness_loss
            )
            return {
                "total": total,
                "role": role_loss.detach(),
                "transition": transition_loss.detach(),
                "smoothness": smoothness_loss.detach(),
            }


else:

    def transition_targets(role_targets: Any) -> Any:
        del role_targets
        raise RuntimeError("transition_targets requires PyTorch")


    class BoundaryRoleLoss:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            raise RuntimeError("BoundaryRoleLoss requires PyTorch")
