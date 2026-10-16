"""
StreamGuard Concept Drift Detection and Adaptive Telemetry Package (Milestone 4).

Provides unsupervised, label-free distribution shift monitoring over streaming risk scores
and continuous feature distributions under delayed ground-truth verification.

Exports:
  - DriftStatus: Categorical stability tiers (STABLE, WARNING, DRIFT_DETECTED).
  - DEFAULT_HYBRID_BIN_EDGES: Custom quantile bin edges optimized for zero-skewed fraud scores.
  - PSIResult: Diagnostic dataclass for Population Stability Index calculations.
  - calculate_psi: Fast domain-adaptive hybrid binning PSI calculation.
  - compute_hybrid_psi: Detailed PSI computation with per-bin diagnostics.
  - classify_psi_status: Helper classifying PSI scalar into DriftStatus tiers.
  - WassersteinResult: Diagnostic dataclass for 1D Wasserstein Earth Mover's Distance.
  - calculate_wasserstein_distance: 1D Wasserstein distance between sample distributions.
  - compute_wasserstein_result: Structured Wasserstein calculation with threshold alert.
  - compute_feature_wasserstein_distances: Pairwise feature-level Wasserstein metrics.
  - DriftTelemetry: Standardized telemetry contract conforming to PROJECT.md and REST specs.
  - StreamingDriftMonitor: Sliding-window streaming monitor with automatic threshold recalibration.
  - default_recalibration_strategy: Default dynamic boundary adjustment policy on drift alert.
  - RecalibrationHook: Type alias for custom recalibration callback hooks.
"""

from app.drift.psi import (
    DEFAULT_HYBRID_BIN_EDGES,
    DriftStatus,
    PSIResult,
    calculate_psi,
    classify_psi_status,
    compute_hybrid_psi,
)
from app.drift.wasserstein import (
    WassersteinResult,
    calculate_wasserstein_distance,
    compute_feature_wasserstein_distances,
    compute_wasserstein_result,
)
from app.drift.monitor import (
    DriftTelemetry,
    RecalibrationHook,
    StreamingDriftMonitor,
    default_recalibration_strategy,
)

__all__ = [
    "DriftStatus",
    "DEFAULT_HYBRID_BIN_EDGES",
    "PSIResult",
    "calculate_psi",
    "compute_hybrid_psi",
    "classify_psi_status",
    "WassersteinResult",
    "calculate_wasserstein_distance",
    "compute_wasserstein_result",
    "compute_feature_wasserstein_distances",
    "DriftTelemetry",
    "StreamingDriftMonitor",
    "default_recalibration_strategy",
    "RecalibrationHook",
]
