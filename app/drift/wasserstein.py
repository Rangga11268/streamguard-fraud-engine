"""
1D Wasserstein Distance (Earth Mover's Distance) Module for StreamGuard.

Computes continuous geometrical distribution differences between baseline reference
distributions and real-time streaming windows for:
  1. Prediction fraud risk scores [0.0, 1.0].
  2. Continuous velocity and transaction amount feature streams.

The 1D Wasserstein distance measures the minimal work needed to transform one empirical
distribution into another:
    W_1(u, v) = integral_0^1 |F_u^{-1}(t) - F_v^{-1}(t)| dt
              = integral_{-inf}^{inf} |F_u(x) - F_v(x)| dx
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.stats import wasserstein_distance as scipy_wasserstein
    HAS_SCIPY: bool = True
except ImportError:
    HAS_SCIPY = False

from app.config import WASSERSTEIN_ALERT_THRESHOLD
from app.drift.psi import DriftStatus


@dataclass(slots=True)
class WassersteinResult:
    """Diagnostic result of a 1D Wasserstein Earth Mover's Distance calculation."""
    distance: float
    status: DriftStatus
    is_drift: bool
    alert_threshold: float
    sample_sizes: Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "distance": round(float(self.distance), 6),
            "status": self.status.value,
            "is_drift": self.is_drift,
            "alert_threshold": self.alert_threshold,
            "sample_sizes": self.sample_sizes,
        }


def _sanitize_1d_array(
    values: Union[np.ndarray, Sequence[float]],
    name: str = "values",
) -> np.ndarray:
    """Validate and sanitize a 1D continuous array, filtering non-finite values."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"Input '{name}' array must be non-empty")
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        arr = arr[finite_mask]
        if arr.size == 0:
            raise ValueError(f"Input '{name}' array must contain at least one finite numeric value")
    return arr


def _wasserstein_1d_numpy(u: np.ndarray, v: np.ndarray, u_sorted: Optional[np.ndarray] = None) -> float:
    """Vectorized NumPy implementation of 1D Wasserstein Earth Mover's Distance.

    Computes the exact integral of the absolute difference between two empirical CDFs.
    Identical to scipy.stats.wasserstein_distance down to float machine precision.
    """
    if len(u) == len(v) and np.array_equal(u, v):
        return 0.0

    u_s = u_sorted if u_sorted is not None else np.sort(u)
    v_sorted = np.sort(v)

    all_vals = np.concatenate([u_s, v_sorted])
    all_vals.sort()

    # Empirical CDF evaluation on all sorted support points
    u_cdf = np.searchsorted(u_s, all_vals[:-1], side="right") / len(u_s)
    v_cdf = np.searchsorted(v_sorted, all_vals[:-1], side="right") / len(v_sorted)

    deltas = np.diff(all_vals)
    dist = np.sum(np.abs(u_cdf - v_cdf) * deltas)
    return float(max(0.0, dist))


def calculate_wasserstein_distance(
    u_values: Union[np.ndarray, Sequence[float]],
    v_values: Union[np.ndarray, Sequence[float]],
    u_sorted: Optional[np.ndarray] = None,
) -> float:
    """Calculate the 1D Wasserstein distance between two empirical sample sets.

    Args:
        u_values: First distribution samples (e.g. baseline reference).
        v_values: Second distribution samples (e.g. streaming window).
        u_sorted: Optional precomputed sorted array of u_values.

    Returns:
        float: Computed non-negative Wasserstein-1 distance.
    """
    # Fast path if u_sorted is already sanitized and sorted numpy array
    if u_sorted is not None:
        v_arr = _sanitize_1d_array(v_values, name="v_values")
        return _wasserstein_1d_numpy(u_sorted, v_arr, u_sorted=u_sorted)

    u_arr = _sanitize_1d_array(u_values, name="u_values")
    v_arr = _sanitize_1d_array(v_values, name="v_values")

    # Fast path for identical inputs
    if len(u_arr) == len(v_arr) and np.array_equal(u_arr, v_arr):
        return 0.0

    if HAS_SCIPY:
        return float(scipy_wasserstein(u_arr, v_arr))
    return _wasserstein_1d_numpy(u_arr, v_arr)


def compute_wasserstein_result(
    u_values: Union[np.ndarray, Sequence[float]],
    v_values: Union[np.ndarray, Sequence[float]],
    alert_threshold: float = WASSERSTEIN_ALERT_THRESHOLD,
) -> WassersteinResult:
    """Compute 1D Wasserstein distance and return structured alert diagnostics."""
    u_arr = _sanitize_1d_array(u_values, name="u_values")
    v_arr = _sanitize_1d_array(v_values, name="v_values")

    dist = calculate_wasserstein_distance(u_arr, v_arr)
    is_drift = bool(dist >= alert_threshold)
    status = DriftStatus.DRIFT_DETECTED if is_drift else DriftStatus.STABLE

    return WassersteinResult(
        distance=dist,
        status=status,
        is_drift=is_drift,
        alert_threshold=alert_threshold,
        sample_sizes=(len(u_arr), len(v_arr)),
    )


def compute_feature_wasserstein_distances(
    reference_features: Union[np.ndarray, Dict[str, Union[np.ndarray, Sequence[float]]]],
    current_features: Union[np.ndarray, Dict[str, Union[np.ndarray, Sequence[float]]]],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute pairwise 1D Wasserstein distances across multiple continuous feature streams.

    Args:
        reference_features: 2D array (N x D) or mapping of feature_name -> 1D array.
        current_features: 2D array (M x D) or mapping of feature_name -> 1D array.
        feature_names: Optional names corresponding to columns if 2D array is passed.

    Returns:
        Dict[str, float]: Mapping of feature_name to computed Wasserstein distance.
    """
    results: Dict[str, float] = {}

    if isinstance(reference_features, dict) and isinstance(current_features, dict):
        common_keys = set(reference_features.keys()).intersection(current_features.keys())
        for key in sorted(common_keys):
            u = reference_features[key]
            v = current_features[key]
            if len(u) > 0 and len(v) > 0:
                results[key] = calculate_wasserstein_distance(u, v)
        return results

    # Array path
    ref_mat = np.asarray(reference_features, dtype=np.float64)
    cur_mat = np.asarray(current_features, dtype=np.float64)

    if ref_mat.ndim == 1:
        ref_mat = ref_mat.reshape(-1, 1)
    if cur_mat.ndim == 1:
        cur_mat = cur_mat.reshape(-1, 1)

    n_feats = min(ref_mat.shape[1], cur_mat.shape[1])
    for j in range(n_feats):
        f_name = feature_names[j] if feature_names and j < len(feature_names) else f"feature_{j}"
        u_col = ref_mat[:, j]
        v_col = cur_mat[:, j]
        results[f_name] = calculate_wasserstein_distance(u_col, v_col)

    return results
