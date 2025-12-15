"""EvalHub SDK Models - Standard request/response models for framework adapters."""

from .api import (
    BenchmarkInfo,
    ErrorResponse,
    EvaluationJob,
    # Core API models
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    HealthResponse,
    # Status and metadata
    JobStatus,
    ModelConfig,
)

__all__ = [
    # API models
    "EvaluationRequest",
    "EvaluationResponse",
    "EvaluationJob",
    "EvaluationResult",
    "BenchmarkInfo",
    "ModelConfig",
    "FrameworkInfo",
    "JobStatus",
    "EvaluationStatus",
    "ErrorResponse",
    "HealthResponse",
]
