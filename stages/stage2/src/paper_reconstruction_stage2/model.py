from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


try:
    import torch
    from torch import Tensor, nn
except ImportError:  # Model use is optional until the training extra is installed.
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None


@dataclass(slots=True)
class BoundaryRoleModelConfig:
    geometry_features: int = 7
    raster_channels: int = 4
    branch_channels: int = 64
    fusion_channels: int = 128
    dropout: float = 0.10
    use_geometry: bool = True
    use_raster: bool = True
    architecture: str = "tcn"
    attention_layers: int = 0
    attention_heads: int = 4
    attention_expansion: float = 2.0
    use_global_context: bool = False
    global_geometry_features: int = 9
    global_context_points: int = 256
    global_mask_size: int = 64
    global_attention_layers: int = 2

    def validate(self) -> None:
        if self.geometry_features < 1 or self.raster_channels < 1:
            raise ValueError("input channel counts must be positive")
        if self.branch_channels < 8 or self.fusion_channels < 8:
            raise ValueError("model channel counts must be at least 8")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.use_geometry and not self.use_raster:
            raise ValueError("at least one model branch must be enabled")
        if self.architecture not in {
            "tcn",
            "hybrid_attention",
            "global_context_hybrid",
        }:
            raise ValueError(
                "architecture must be tcn, hybrid_attention, or global_context_hybrid"
            )
        if self.attention_layers < 0 or self.attention_heads < 1:
            raise ValueError("attention layer count must be non-negative and heads positive")
        if (
            self.architecture in {"hybrid_attention", "global_context_hybrid"}
            and self.attention_layers < 1
        ):
            raise ValueError("hybrid architectures require at least one attention layer")
        if (
            self.architecture in {"hybrid_attention", "global_context_hybrid"}
            and self.fusion_channels % self.attention_heads != 0
        ):
            raise ValueError("fusion_channels must be divisible by attention_heads")
        if self.attention_expansion < 1.0:
            raise ValueError("attention_expansion must be at least 1.0")
        if self.architecture == "global_context_hybrid" and not self.use_global_context:
            raise ValueError("global_context_hybrid requires use_global_context")
        if self.use_global_context:
            if self.global_geometry_features < 3:
                raise ValueError("global geometry must include at least three features")
            if self.global_context_points < 16 or self.global_mask_size < 16:
                raise ValueError("global contour and mask sizes must be at least 16")
            if self.global_attention_layers < 1:
                raise ValueError("global context requires at least one attention layer")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


if nn is not None:

    def _group_count(channels: int) -> int:
        for groups in (16, 8, 4, 2):
            if channels % groups == 0:
                return groups
        return 1


    class ConvNormAct2d(nn.Sequential):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            *,
            stride: tuple[int, int] = (1, 1),
        ):
            super().__init__(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(output_channels), output_channels),
                nn.SiLU(inplace=True),
            )


    class TemporalResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(channels), channels),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(channels), channels),
            )
            self.activation = nn.SiLU(inplace=True)

        def forward(self, values: Tensor) -> Tensor:
            return self.activation(values + self.network(values))


    class ContourAttentionBlock(nn.Module):
        """Small pre-normalized self-attention block over ordered contour points."""

        def __init__(
            self,
            channels: int,
            heads: int,
            expansion: float,
            dropout: float,
        ):
            super().__init__()
            hidden = int(round(channels * expansion))
            self.attention_norm = nn.LayerNorm(channels)
            self.attention = nn.MultiheadAttention(
                channels,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.feed_forward_norm = nn.LayerNorm(channels)
            self.feed_forward = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, channels),
                nn.Dropout(dropout),
            )

        def forward(self, values: Tensor) -> Tensor:
            normalized = self.attention_norm(values)
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            values = values + attended
            return values + self.feed_forward(self.feed_forward_norm(values))


    class CrossContourAttentionBlock(nn.Module):
        """Inject complete-piece context into each local contour token."""

        def __init__(
            self,
            channels: int,
            heads: int,
            expansion: float,
            dropout: float,
        ):
            super().__init__()
            hidden = int(round(channels * expansion))
            self.query_norm = nn.LayerNorm(channels)
            self.context_norm = nn.LayerNorm(channels)
            self.attention = nn.MultiheadAttention(
                channels,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.feed_forward_norm = nn.LayerNorm(channels)
            self.feed_forward = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, channels),
                nn.Dropout(dropout),
            )

        def forward(self, local: Tensor, context: Tensor) -> Tensor:
            normalized_context = self.context_norm(context)
            attended, _ = self.attention(
                self.query_norm(local),
                normalized_context,
                normalized_context,
                need_weights=False,
            )
            local = local + attended
            return local + self.feed_forward(self.feed_forward_norm(local))


    class BoundaryRoleNetV1(nn.Module):
        """Dual-stream contour model that preserves one output per input point."""

        def __init__(
            self,
            config: BoundaryRoleModelConfig | None = None,
            *,
            geometry_mean: list[float] | None = None,
            geometry_std: list[float] | None = None,
        ):
            super().__init__()
            self.config = config or BoundaryRoleModelConfig()
            self.config.validate()
            branch = self.config.branch_channels
            if geometry_mean is None:
                geometry_mean = [0.0] * self.config.geometry_features
            if geometry_std is None:
                geometry_std = [1.0] * self.config.geometry_features
            if len(geometry_mean) != self.config.geometry_features or len(
                geometry_std
            ) != self.config.geometry_features:
                raise ValueError("geometry normalization length does not match model")
            self.register_buffer(
                "geometry_mean",
                torch.tensor(geometry_mean, dtype=torch.float32).view(1, 1, -1),
            )
            self.register_buffer(
                "geometry_std",
                torch.tensor(geometry_std, dtype=torch.float32)
                .clamp_min(1e-6)
                .view(1, 1, -1),
            )

            if self.config.use_raster:
                self.raster_branch = nn.Sequential(
                    ConvNormAct2d(self.config.raster_channels, 32, stride=(1, 2)),
                    ConvNormAct2d(32, branch, stride=(1, 2)),
                    ConvNormAct2d(branch, branch, stride=(1, 2)),
                )
            else:
                self.raster_branch = None

            if self.config.use_geometry:
                self.geometry_projection = nn.Sequential(
                    nn.Conv1d(
                        self.config.geometry_features,
                        branch,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(branch), branch),
                    nn.SiLU(inplace=True),
                )
                self.geometry_temporal = nn.Sequential(
                    *[
                        TemporalResidualBlock(branch, dilation, self.config.dropout)
                        for dilation in (1, 2, 4)
                    ]
                )
            else:
                self.geometry_projection = None
                self.geometry_temporal = None

            fusion_input = branch * int(self.config.use_raster) + branch * int(
                self.config.use_geometry
            )
            self.fusion_projection = nn.Sequential(
                nn.Conv1d(
                    fusion_input,
                    self.config.fusion_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(self.config.fusion_channels),
                    self.config.fusion_channels,
                ),
                nn.SiLU(inplace=True),
            )
            self.fusion_temporal = nn.Sequential(
                *[
                    TemporalResidualBlock(
                        self.config.fusion_channels,
                        dilation,
                        self.config.dropout,
                    )
                    for dilation in (1, 2, 4, 8)
                ]
            )
            if self.config.architecture in {
                "hybrid_attention",
                "global_context_hybrid",
            }:
                self.attention_blocks = nn.Sequential(
                    *[
                        ContourAttentionBlock(
                            self.config.fusion_channels,
                            self.config.attention_heads,
                            self.config.attention_expansion,
                            self.config.dropout,
                        )
                        for _ in range(self.config.attention_layers)
                    ]
                )
            else:
                # An empty Sequential adds no checkpoint keys, so legacy TCN
                # checkpoints remain state-dict compatible.
                self.attention_blocks = nn.Sequential()

            if self.config.use_global_context:
                global_mean = [0.0, 0.0, *geometry_mean]
                global_std = [1.0, 1.0, *geometry_std]
                if len(global_mean) != self.config.global_geometry_features:
                    raise ValueError(
                        "global geometry feature count must equal normalized xy plus "
                        "the local geometry features"
                    )
                self.register_buffer(
                    "global_geometry_mean",
                    torch.tensor(global_mean, dtype=torch.float32).view(1, 1, -1),
                )
                self.register_buffer(
                    "global_geometry_std",
                    torch.tensor(global_std, dtype=torch.float32)
                    .clamp_min(1e-6)
                    .view(1, 1, -1),
                )
                self.global_geometry_projection = nn.Sequential(
                    nn.Conv1d(
                        self.config.global_geometry_features,
                        self.config.fusion_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        _group_count(self.config.fusion_channels),
                        self.config.fusion_channels,
                    ),
                    nn.SiLU(inplace=True),
                )
                self.global_temporal = nn.Sequential(
                    *[
                        TemporalResidualBlock(
                            self.config.fusion_channels,
                            dilation,
                            self.config.dropout,
                        )
                        for dilation in (1, 2, 4)
                    ]
                )
                self.global_attention = nn.Sequential(
                    *[
                        ContourAttentionBlock(
                            self.config.fusion_channels,
                            self.config.attention_heads,
                            self.config.attention_expansion,
                            self.config.dropout,
                        )
                        for _ in range(self.config.global_attention_layers)
                    ]
                )
                self.global_mask_branch = nn.Sequential(
                    ConvNormAct2d(1, 32, stride=(2, 2)),
                    ConvNormAct2d(32, 64, stride=(2, 2)),
                    ConvNormAct2d(64, branch, stride=(2, 2)),
                    ConvNormAct2d(
                        branch,
                        self.config.fusion_channels,
                        stride=(2, 2),
                    ),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.cross_attention = CrossContourAttentionBlock(
                    self.config.fusion_channels,
                    self.config.attention_heads,
                    self.config.attention_expansion,
                    self.config.dropout,
                )
            else:
                self.global_geometry_projection = None
                self.global_temporal = None
                self.global_attention = None
                self.global_mask_branch = None
                self.cross_attention = None
            self.role_head = nn.Conv1d(self.config.fusion_channels, 3, kernel_size=1)
            self.transition_head = nn.Conv1d(
                self.config.fusion_channels, 1, kernel_size=1
            )

        def forward(
            self,
            geometry: Tensor,
            raster: Tensor | None = None,
            global_geometry: Tensor | None = None,
            global_mask: Tensor | None = None,
        ) -> dict[str, Tensor]:
            branches: list[Tensor] = []
            if self.config.use_geometry:
                if geometry.ndim != 3 or geometry.shape[-1] != self.config.geometry_features:
                    raise ValueError(
                        "geometry must have shape [batch, sequence, geometry_features]"
                    )
                normalized = (geometry - self.geometry_mean) / self.geometry_std
                projected = self.geometry_projection(normalized.transpose(1, 2))
                branches.append(self.geometry_temporal(projected))
            if self.config.use_raster:
                if raster is None or raster.ndim != 4:
                    raise ValueError("raster must have shape [batch, channels, sequence, width]")
                raster_features = self.raster_branch(raster).mean(dim=3)
                branches.append(raster_features)
            fused = self.fusion_projection(torch.cat(branches, dim=1))
            fused = self.fusion_temporal(fused)
            if self.config.architecture in {
                "hybrid_attention",
                "global_context_hybrid",
            }:
                fused = self.attention_blocks(fused.transpose(1, 2)).transpose(1, 2)
            if self.config.use_global_context:
                if (
                    global_geometry is None
                    or global_geometry.ndim != 3
                    or global_geometry.shape[-1]
                    != self.config.global_geometry_features
                ):
                    raise ValueError(
                        "global_geometry must have shape [batch, global_points, features]"
                    )
                if global_mask is None or global_mask.ndim != 4:
                    raise ValueError(
                        "global_mask must have shape [batch, 1, height, width]"
                    )
                normalized_global = (
                    global_geometry - self.global_geometry_mean
                ) / self.global_geometry_std
                global_tokens = self.global_geometry_projection(
                    normalized_global.transpose(1, 2)
                )
                global_tokens = self.global_temporal(global_tokens).transpose(1, 2)
                global_tokens = self.global_attention(global_tokens)
                mask_token = self.global_mask_branch(global_mask).flatten(2).transpose(1, 2)
                context = torch.cat((global_tokens, mask_token), dim=1)
                fused = self.cross_attention(fused.transpose(1, 2), context).transpose(
                    1, 2
                )
            return {
                "role_logits": self.role_head(fused).transpose(1, 2),
                "transition_logits": self.transition_head(fused).squeeze(1),
            }

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())


else:

    class BoundaryRoleNetV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            raise RuntimeError(
                "BoundaryRoleNetV1 requires PyTorch. Install the CUDA or CPU "
                "build appropriate for this machine before training."
            )
