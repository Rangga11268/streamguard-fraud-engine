"""
Domain-Adaptive Population Stability Index (PSI) Module for StreamGuard.

Calculates unsupervised distribution shift metrics on continuous prediction risk scores
and feature streams. In financial fraud detection, risk scores are heavily zero-skewed
(>95% of scores < 0.05), causing standard equal-frequency deciles to collapse into single bins.
This module uses a domain-adaptive hybrid binning strategy:
  [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
with additive Laplace/Dirichlet epsilon smoothing (1e-4) to guarantee numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from app.config import (
    EPSILON,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
)

# Domain-adaptive hybrid bin edges for zero-skewed fraud risk scores
DEFAULT_HYBRID_BIN_EDGES: np.ndarray = np.array(
    [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
    dtype=np.float64,
)


class DriftStatus(str, Enum):
    """Categorical classification of distribution stability based on PSI boundaries."""
    STABLE = "STABLE"
    WARNING = "WARNING"
    DRIFT_DETECTED = "DRIFT_DETECTED"


@dataclass(slots=True)
class PSIResult:
    """Detailed mathematical breakdown of a Population Stability Index calculation."""
    psi: float
    status: DriftStatus
    is_drift: bool
    is_warning: bool
    expected_pct: np.ndarray
    actual_pct: np.ndarray
    bin_edges: np.ndarray
    bin_contributions: np.ndarray

    def to_dict(self) -> dict:
        return {
            "psi": round(float(self.psi), 6),
            "status": self.status.value,
            "is_drift": self.is_drift,
            "is_warning": self.is_warning,
            "bin_edges": [round(float(b), 6) for b in self.bin_edges],
            "bin_contributions": [round(float(c), 6) for c in self.bin_contributions],
        }


def classify_psi_status(
    psi: float,
    warning_threshold: float = PSI_WARNING_THRESHOLD,
    alert_threshold: float = PSI_ALERT_THRESHOLD,
) -> DriftStatus:
    """Classify PSI scalar value into standard operational alert tiers.

    Thresholds:
      PSI < 0.10:           STABLE (No significant distribution shift)
      0.10 <= PSI < 0.25:   WARNING (Moderate shift, monitor closely)
      PSI >= 0.25:          DRIFT_DETECTED (Severe concept/distribution shift)
    """
    if psi >= alert_threshold:
        return DriftStatus.DRIFT_DETECTED
    elif psi >= warning_threshold:
        return DriftStatus.WARNING
    return DriftStatus.STABLE


def _sanitize_score_array(
    scores: Union[np.ndarray, Sequence[float]],
    name: str = "scores",
) -> np.ndarray:
    """Validate and sanitize an input score array, stripping NaNs and Infs."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"Input '{name}' array must be non-empty")
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        arr = arr[finite_mask]
        if arr.size == 0:
            raise ValueError(f"Input '{name}' array must contain at least one finite numeric value")
    return arr


def calculate_psi(
    expected_scores: Union[np.ndarray, Sequence[float]],
    actual_scores: Union[np.ndarray, Sequence[float]],
    num_bins: Optional[int] = None,
    epsilon: float = EPSILON,
    bin_edges: Optional[Union[np.ndarray, Sequence[float]]] = None,
    use_hybrid_bins: bool = True,
    expected_pct: Optional[np.ndarray] = None,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and evaluation scores.

    Supports both domain-adaptive hybrid binning (default for zero-skewed risk scores)
    and quantile-based binning.

    Args:
        expected_scores: Baseline / reference distribution samples.
        actual_scores: Streaming / evaluation window distribution samples.
        num_bins: Optional number of quantile bins. If provided and not using hybrid bins,
                  generates quantile edges from expected_scores.
        epsilon: Additive smoothing constant to prevent division by zero or log(0).
        bin_edges: Explicit custom bin edges. If provided, overrides default hybrid edges.
        use_hybrid_bins: When True and bin_edges is None (and num_bins is None or 10),
                         uses DEFAULT_HYBRID_BIN_EDGES.
        expected_pct: Optional precomputed expected baseline proportions.

    Returns:
        float: Computed non-negative PSI scalar.
    """
    res = compute_hybrid_psi(
        expected_scores=expected_scores,
        actual_scores=actual_scores,
        num_bins=num_bins,
        epsilon=epsilon,
        bin_edges=bin_edges,
        use_hybrid_bins=use_hybrid_bins,
        expected_pct=expected_pct,
    )
    return res.psi


def compute_hybrid_psi(
    expected_scores: Union[np.ndarray, Sequence[float]],
    actual_scores: Union[np.ndarray, Sequence[float]],
    num_bins: Optional[int] = None,
    epsilon: float = EPSILON,
    bin_edges: Optional[Union[np.ndarray, Sequence[float]]] = None,
    use_hybrid_bins: bool = True,
    warning_threshold: float = PSI_WARNING_THRESHOLD,
    alert_threshold: float = PSI_ALERT_THRESHOLD,
    expected_pct: Optional[np.ndarray] = None,
) -> PSIResult:
    """Compute detailed Population Stability Index with full bin-level diagnostics.

    Args:
        expected_scores: Baseline / reference samples.
        actual_scores: Streaming evaluation samples.
        num_bins: Number of quantile bins if quantile binning is requested.
        epsilon: Additive Laplace/Dirichlet smoothing constant.
        bin_edges: Custom bin edges array.
        use_hybrid_bins: Use default hybrid bin edges when applicable.
        warning_threshold: Threshold for WARNING status.
        alert_threshold: Threshold for DRIFT_DETECTED status.
        expected_pct: Optional precomputed expected baseline proportions.

    Returns:
        PSIResult: Complete diagnostic dataclass.
    """
    act_arr = _sanitize_score_array(actual_scores, name="actual_scores")

    # Determine bin edges
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=np.float64).copy()
    elif use_hybrid_bins and (num_bins is None or num_bins == 10):
        edges = DEFAULT_HYBRID_BIN_EDGES.copy()
    else:
        exp_arr = _sanitize_score_array(expected_scores, name="expected_scores")
        k_bins = num_bins if num_bins is not None else 10
        percentiles = np.linspace(0, 100, k_bins + 1)
        edges = np.percentile(exp_arr, percentiles)
        # Ensure monotonic uniqueness
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-5

    # Expand outer boundaries so edge values are captured without clipping
    e_bins = edges.copy()
    e_bins[0] = -1e9
    e_bins[-1] = 1e9

    k_bins = len(edges) - 1

    # Frequency binning for actual scores
    act_counts, _ = np.histogram(act_arr, bins=e_bins)
    n_act = np.sum(act_counts)
    act_pct = (act_counts + epsilon) / (n_act + epsilon * k_bins)

    # Use precomputed expected_pct or compute from expected_scores
    if expected_pct is not None and len(expected_pct) == k_bins:
        exp_pct = expected_pct
    else:
        exp_arr = _sanitize_score_array(expected_scores, name="expected_scores")
        exp_counts, _ = np.histogram(exp_arr, bins=e_bins)
        n_exp = np.sum(exp_counts)
        exp_pct = (exp_counts + epsilon) / (n_exp + epsilon * k_bins)

    # Bin-level PSI contribution: (Actual% - Expected%) * ln(Actual% / Expected%)
    bin_diff = act_pct - exp_pct
    bin_log_ratio = np.log(act_pct / exp_pct)
    bin_contributions = bin_diff * bin_log_ratio

    raw_psi = float(np.sum(bin_contributions))
    # Safeguard against negative float precision noise
    psi_val = float(max(0.0, raw_psi))

    status = classify_psi_status(
        psi_val,
        warning_threshold=warning_threshold,
        alert_threshold=alert_threshold,
    )
    is_drift = (status == DriftStatus.DRIFT_DETECTED)
    is_warning = (status == DriftStatus.WARNING)

    return PSIResult(
        psi=psi_val,
        status=status,
        is_drift=is_drift,
        is_warning=is_warning,
        expected_pct=exp_pct,
        actual_pct=act_pct,
        bin_edges=edges,
        bin_contributions=bin_contributions,
    )
