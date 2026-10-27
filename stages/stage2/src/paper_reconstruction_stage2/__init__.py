"""Stage 2 boundary-role learning and production edge-hypothesis inference."""

__version__ = "1.0.2"

from .boundary import BoundaryConfig, BoundaryDatasetCompiler
from .config import GeneratorConfig
from .generator import DatasetGenerator
from .observation import ObservationCompiler, ObservationConfig
from .pipeline import PipelinePaths, Stage2Pipeline
from .model import BoundaryRoleModelConfig, BoundaryRoleNetV1
from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
    TorchBoundaryWindowDataset,
)
from .production import (
    BoundaryGeometryConfig,
    ProductionBoundarySampler,
    ProductionInferenceConfig,
    Stage2ProductionRunner,
    load_stage1_output,
)
from .verification import verify_stage2_output
from .onnx_runtime import OnnxModelRelease, OnnxContourPredictor

__all__ = [
    "BoundaryConfig",
    "BoundaryDatasetCompiler",
    "DatasetGenerator",
    "GeneratorConfig",
    "ObservationCompiler",
    "ObservationConfig",
    "PipelinePaths",
    "Stage2Pipeline",
    "BoundaryTrainingIndex",
    "BoundaryRoleModelConfig",
    "BoundaryRoleNetV1",
    "BoundaryWindowConfig",
    "BoundaryWindowSampler",
    "TorchBoundaryWindowDataset",
    "BoundaryGeometryConfig",
    "ProductionBoundarySampler",
    "ProductionInferenceConfig",
    "Stage2ProductionRunner",
    "load_stage1_output",
    "verify_stage2_output",
    "OnnxModelRelease",
    "OnnxContourPredictor",
    "__version__",
]
