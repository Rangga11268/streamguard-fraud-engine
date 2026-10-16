"""
Streaming Drift Telemetry Monitor and Dynamic Threshold Recalibration Module for StreamGuard.

Implements unsupervised, label-free concept and covariate drift detection over streaming
prediction scores and continuous feature distributions without requiring delayed ground-truth
labels (14-30 day chargeback horizon).

Features:
  1. Rolling score window (150-200 observations) against baseline reference scores (1000 observations).
  2. Sub-millisecond dual-metric calculation: Domain-Adaptive PSI and 1D Wasserstein distance.
  3. Rapid early warning: flags distribution shift within <= 200 samples of drift onset.
  4. Automatic threshold recalibration hook: triggers dynamic adjustment of decision boundaries
     (tau_challenge, tau_block) upon drift alert.
  5. Strict adherence to the DriftTelemetry interface contract defined in PROJECT.md and REST API specs.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from app.config import (
    DEFAULT_TAU_BLOCK,
    DEFAULT_TAU_CHALLENGE,
    DRIFT_WINDOW_SIZE,
    EPSILON,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    REFERENCE_WINDOW_SIZE,
    WASSERSTEIN_ALERT_THRESHOLD,
)
from app.drift.psi import (
    DEFAULT_HYBRID_BIN_EDGES,
    DriftStatus,
    calculate_psi,
    classify_psi_status,
)
from app.drift.wasserstein import (
    calculate_wasserstein_distance,
    compute_feature_wasserstein_distances,
)


def _format_timestamp(ts: Optional[Union[float, str]]) -> str:
    """Format simulation float epoch or current time into ISO-8601 UTC string."""
    if ts is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts, str):
        return ts
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class DriftTelemetry:
    """Standardized drift telemetry contract conforming to PROJECT.md and REST endpoints."""
    window_samples: int
    score_psi: float
    score_wasserstein: float
    feature_psi: Dict[str, float]
    drift_status: str  # "STABLE" | "WARNING" | "DRIFT_DETECTED"
    drift_alert: bool
    sample_count_since_drift_onset: int
    last_recalibration_timestamp: Optional[str]
    current_thresholds: Dict[str, float]
    feature_wasserstein: Dict[str, float] = field(default_factory=dict)
    recalibration_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry instance into JSON-compliant dictionary."""
        return {
            "window_samples": self.window_samples,
            "score_psi": round(float(self.score_psi), 4),
            "score_wasserstein": round(float(self.score_wasserstein), 4),
            "feature_psi": {k: round(float(v), 4) for k, v in self.feature_psi.items()},
            "feature_wasserstein": {k: round(float(v), 4) for k, v in self.feature_wasserstein.items()},
            "drift_status": self.drift_status,
            "drift_alert": self.drift_alert,
            "sample_count_since_drift_onset": self.sample_count_since_drift_onset,
            "last_recalibration_timestamp": self.last_recalibration_timestamp,
            "current_thresholds": {k: round(float(v), 4) for k, v in self.current_thresholds.items()},
            "recalibration_triggered": self.recalibration_triggered,
        }


RecalibrationHook = Callable[[DriftTelemetry, Dict[str, float]], Dict[str, float]]


def default_recalibration_strategy(
    telemetry: DriftTelemetry,
    current_thresholds: Dict[str, float],
) -> Dict[str, float]:
    """Default dynamic adjustment of decision boundaries under drift alert.

    Under concept drift or fraud burst:
      - Tighten tau_challenge to increase step-up verification on ambiguous scores.
      - Adapt tau_block to bound fraud loss while preventing customer churn.
    """
    psi = telemetry.score_psi
    shift_factor = min(2.0, max(0.1, psi / 0.25))

    # Dynamic tightening of challenge boundary
    new_tau_challenge = max(
        0.05,
        min(0.35, current_thresholds["tau_challenge"] / (1.0 + 0.25 * shift_factor)),
    )
    # Adapt block boundary slightly
    new_tau_block = max(
        0.40,
        min(0.85, current_thresholds["tau_block"] * (1.0 + 0.05 * (shift_factor - 1.0))),
    )

    return {
        "tau_challenge": round(float(new_tau_challenge), 4),
        "tau_block": round(float(new_tau_block), 4),
    }


class StreamingDriftMonitor:
    """Stateful, unsupervised streaming drift monitor with sliding window telemetry."""

    def __init__(
        self,
        reference_scores: Optional[Union[np.ndarray, Sequence[float]]] = None,
        reference_features: Optional[Union[np.ndarray, Dict[str, Union[np.ndarray, Sequence[float]]]]] = None,
        feature_names: Optional[List[str]] = None,
        window_size: int = DRIFT_WINDOW_SIZE,
        reference_size: int = REFERENCE_WINDOW_SIZE,
        psi_warning_threshold: float = PSI_WARNING_THRESHOLD,
        psi_alert_threshold: float = PSI_ALERT_THRESHOLD,
        wasserstein_alert_threshold: float = WASSERSTEIN_ALERT_THRESHOLD,
        initial_tau_challenge: float = DEFAULT_TAU_CHALLENGE,
        initial_tau_block: float = DEFAULT_TAU_BLOCK,
        auto_recalibrate: bool = True,
        recalibration_strategy: Optional[RecalibrationHook] = None,
        min_evaluation_samples: Optional[int] = None,
    ) -> None:
        self.window_size: int = window_size
        self.reference_size: int = reference_size
        self.psi_warning_threshold: float = psi_warning_threshold
        self.psi_alert_threshold: float = psi_alert_threshold
        self.wasserstein_alert_threshold: float = wasserstein_alert_threshold
        self.initial_tau_challenge: float = initial_tau_challenge
        self.initial_tau_block: float = initial_tau_block
        self.auto_recalibrate: bool = auto_recalibrate
        self.recalibration_strategy: RecalibrationHook = (
            recalibration_strategy or default_recalibration_strategy
        )
        self.min_evaluation_samples: int = min_evaluation_samples or window_size

        # Monitored feature names
        self.feature_names: List[str] = feature_names or ["amount", "velocity_5m_count", "velocity_1h_count"]

        # Sliding evaluation windows
        self._score_window: collections.deque = collections.deque(maxlen=window_size)
        self._feature_windows: Dict[str, collections.deque] = {
            name: collections.deque(maxlen=window_size) for name in self.feature_names
        }

        # Reference baseline stores
        self._reference_scores: collections.deque = collections.deque(maxlen=reference_size)
        self._reference_features: Dict[str, collections.deque] = {
            name: collections.deque(maxlen=reference_size) for name in self.feature_names
        }

        # Operational state
        self._current_thresholds: Dict[str, float] = {
            "tau_challenge": initial_tau_challenge,
            "tau_block": initial_tau_block,
        }
        self._total_samples_observed: int = 0
        self._drift_onset_index: Optional[int] = None
        self._sample_count_since_drift_onset: int = 0
        self._last_recalibration_timestamp: Optional[str] = None
        self._last_telemetry: Optional[DriftTelemetry] = None
        self._custom_hooks: List[RecalibrationHook] = []

        # Baseline acceleration caches
        self._ref_scores_arr: Optional[np.ndarray] = None
        self._ref_scores_sorted: Optional[np.ndarray] = None
        self._ref_score_exp_pct: Optional[np.ndarray] = None
        self._ref_feat_arrs: Dict[str, np.ndarray] = {}
        self._ref_feat_sorted: Dict[str, np.ndarray] = {}
        self._ref_feat_edges: Dict[str, np.ndarray] = {}
        self._ref_feat_exp_pct: Dict[str, np.ndarray] = {}

        # Populate reference baseline if provided
        if reference_scores is not None:
            self.set_baseline_scores(reference_scores)
        if reference_features is not None:
            self.set_baseline_features(reference_features)

    def _update_score_baseline_cache(self) -> None:
        """Precompute sorted score array and hybrid bin proportions for sub-ms execution."""
        if len(self._reference_scores) == 0:
            self._ref_scores_arr = None
            self._ref_scores_sorted = None
            self._ref_score_exp_pct = None
            return

        self._ref_scores_arr = np.array(self._reference_scores, dtype=np.float64)
        self._ref_scores_sorted = np.sort(self._ref_scores_arr)

        edges = DEFAULT_HYBRID_BIN_EDGES.copy()
        edges[0] = -1e9
        edges[-1] = 1e9
        k_bins = len(DEFAULT_HYBRID_BIN_EDGES) - 1

        exp_counts, _ = np.histogram(self._ref_scores_arr, bins=edges)
        n_exp = np.sum(exp_counts)
        self._ref_score_exp_pct = (exp_counts + EPSILON) / (n_exp + EPSILON * k_bins)

    def _update_feature_baseline_cache(self, feature_name: Optional[str] = None) -> None:
        """Precompute sorted feature arrays and quantile edges for sub-ms execution."""
        targets = [feature_name] if feature_name else list(self._reference_features.keys())
        for k in targets:
            vals = self._reference_features.get(k)
            if vals and len(vals) >= 10:
                f_arr = np.array(vals, dtype=np.float64)
                self._ref_feat_arrs[k] = f_arr
                self._ref_feat_sorted[k] = np.sort(f_arr)
                edges = np.percentile(f_arr, np.linspace(0, 100, 11))
                for i in range(1, len(edges)):
                    if edges[i] <= edges[i - 1]:
                        edges[i] = edges[i - 1] + 1e-5
                edges[0] = -1e9
                edges[-1] = 1e9
                self._ref_feat_edges[k] = edges
                c, _ = np.histogram(f_arr, bins=edges)
                self._ref_feat_exp_pct[k] = (c + EPSILON) / (np.sum(c) + EPSILON * 10)

    def set_baseline_scores(self, scores: Union[np.ndarray, Sequence[float]]) -> None:
        """Establish the baseline reference score distribution."""
        arr = np.asarray(scores, dtype=np.float64).ravel()
        finite_arr = arr[np.isfinite(arr)]
        if finite_arr.size == 0:
            raise ValueError("Baseline scores must contain at least one finite numeric value")

        self._reference_scores.clear()
        # Retain up to reference_size
        subset = finite_arr[-self.reference_size:] if len(finite_arr) > self.reference_size else finite_arr
        self._reference_scores.extend(subset.tolist())
        self._update_score_baseline_cache()

    def set_baseline_features(
        self,
        features: Union[np.ndarray, Dict[str, Union[np.ndarray, Sequence[float]]]],
    ) -> None:
        """Establish baseline reference feature distributions."""
        if isinstance(features, dict):
            for k, vals in features.items():
                if k not in self._reference_features:
                    self._reference_features[k] = collections.deque(maxlen=self.reference_size)
                    self._feature_windows[k] = collections.deque(maxlen=self.window_size)
                arr = np.asarray(vals, dtype=np.float64).ravel()
                finite_arr = arr[np.isfinite(arr)]
                self._reference_features[k].clear()
                subset = finite_arr[-self.reference_size:] if len(finite_arr) > self.reference_size else finite_arr
                self._reference_features[k].extend(subset.tolist())
        else:
            mat = np.asarray(features, dtype=np.float64)
            if mat.ndim == 1:
                mat = mat.reshape(-1, 1)
            for j in range(min(mat.shape[1], len(self.feature_names))):
                name = self.feature_names[j]
                col = mat[:, j]
                finite_col = col[np.isfinite(col)]
                self._reference_features[name].clear()
                subset = finite_col[-self.reference_size:] if len(finite_col) > self.reference_size else finite_col
                self._reference_features[name].extend(subset.tolist())
        self._update_feature_baseline_cache()

    def register_recalibration_hook(self, hook: RecalibrationHook) -> None:
        """Register a custom callback invoked when a drift alert triggers threshold recalibration."""
        self._custom_hooks.append(hook)

    def record_prediction(
        self,
        score: float,
        features: Optional[Union[Dict[str, float], Sequence[float], np.ndarray]] = None,
        timestamp: Optional[Union[float, str]] = None,
    ) -> DriftTelemetry:
        """Ingest a single transaction risk score and feature observation into streaming telemetry.

        Args:
            score: Predicted fraud probability in [0.0, 1.0].
            features: Optional mapping or vector of feature values.
            timestamp: Simulation epoch seconds or ISO timestamp.

        Returns:
            DriftTelemetry: Updated snapshot of drift metrics and alert states.
        """
        self._total_samples_observed += 1

        # Auto-bootstrap baseline if unpopulated
        if len(self._reference_scores) < self.reference_size and self._total_samples_observed <= self.reference_size:
            self._reference_scores.append(float(score))

        # Push to sliding score window
        self._score_window.append(float(score))

        # Record feature observations if provided
        if features is not None:
            self._ingest_features(features)

        # Compute telemetry
        telemetry = self._evaluate_drift(timestamp=timestamp)
        self._last_telemetry = telemetry
        return telemetry

    def record_batch(
        self,
        scores: Sequence[float],
        features: Optional[Union[Sequence[Dict[str, float]], np.ndarray]] = None,
        timestamps: Optional[Sequence[Union[float, str]]] = None,
    ) -> DriftTelemetry:
        """Ingest a batch of sequential scoring events, returning the final telemetry snapshot."""
        n = len(scores)
        last_tel: Optional[DriftTelemetry] = None
        for i in range(n):
            s = scores[i]
            feat = features[i] if features is not None else None
            ts = timestamps[i] if timestamps is not None else None
            last_tel = self.record_prediction(score=s, features=feat, timestamp=ts)
        return last_tel if last_tel is not None else self.get_telemetry()

    def _ingest_features(
        self,
        features: Union[Dict[str, float], Sequence[float], np.ndarray],
    ) -> None:
        """Helper to append feature values to sliding windows."""
        if isinstance(features, dict):
            for k, val in features.items():
                if k in self._feature_windows and np.isfinite(val):
                    self._feature_windows[k].append(float(val))
                    if len(self._reference_features.get(k, [])) < self.reference_size:
                        self._reference_features[k].append(float(val))
        elif isinstance(features, (list, tuple, np.ndarray)):
            arr = np.asarray(features, dtype=np.float64).ravel()
            for j in range(min(len(arr), len(self.feature_names))):
                name = self.feature_names[j]
                val = arr[j]
                if np.isfinite(val):
                    self._feature_windows[name].append(float(val))
                    if len(self._reference_features.get(name, [])) < self.reference_size:
                        self._reference_features[name].append(float(val))

    def _evaluate_drift(self, timestamp: Optional[Union[float, str]] = None) -> DriftTelemetry:
        """Evaluate PSI and Wasserstein metrics between reference baseline and sliding window."""
        window_len = len(self._score_window)

        # If not enough evaluation samples have accumulated, return stable baseline telemetry
        if window_len < self.min_evaluation_samples or len(self._reference_scores) < 10:
            return DriftTelemetry(
                window_samples=window_len,
                score_psi=0.0,
                score_wasserstein=0.0,
                feature_psi={name: 0.0 for name in self.feature_names if len(self._feature_windows[name]) > 0},
                drift_status=DriftStatus.STABLE.value,
                drift_alert=False,
                sample_count_since_drift_onset=0,
                last_recalibration_timestamp=self._last_recalibration_timestamp,
                current_thresholds=dict(self._current_thresholds),
                feature_wasserstein={},
                recalibration_triggered=False,
            )

        if self._ref_scores_arr is None:
            self._update_score_baseline_cache()

        ref_scores = self._ref_scores_arr if self._ref_scores_arr is not None else np.array(self._reference_scores, dtype=np.float64)
        cur_scores = np.array(self._score_window, dtype=np.float64)

        # 1. Compute Prediction Score Metrics using cached expected proportions and sorted arrays
        score_psi = calculate_psi(
            expected_scores=ref_scores,
            actual_scores=cur_scores,
            epsilon=EPSILON,
            use_hybrid_bins=True,
            expected_pct=self._ref_score_exp_pct,
        )
        score_w1 = calculate_wasserstein_distance(
            ref_scores,
            cur_scores,
            u_sorted=self._ref_scores_sorted,
        )

        # 2. Compute Feature Stream Metrics using cached edges and baseline distributions
        feat_psi: Dict[str, float] = {}
        feat_w1: Dict[str, float] = {}
        for name in self.feature_names:
            cur_f = self._feature_windows.get(name)
            if cur_f and len(cur_f) >= 10:
                if name not in self._ref_feat_edges:
                    self._update_feature_baseline_cache(name)

                ref_f_arr = self._ref_feat_arrs.get(name)
                cached_edges = self._ref_feat_edges.get(name)
                cached_exp_pct = self._ref_feat_exp_pct.get(name)
                cached_sorted = self._ref_feat_sorted.get(name)

                if ref_f_arr is not None and cached_edges is not None:
                    cur_f_arr = np.array(cur_f, dtype=np.float64)
                    feat_psi[name] = calculate_psi(
                        ref_f_arr,
                        cur_f_arr,
                        bin_edges=cached_edges,
                        expected_pct=cached_exp_pct,
                    )
                    feat_w1[name] = calculate_wasserstein_distance(
                        ref_f_arr,
                        cur_f_arr,
                        u_sorted=cached_sorted,
                    )

        # 3. Determine Drift Alert Status
        is_drift = (score_psi >= self.psi_alert_threshold) or (score_w1 >= self.wasserstein_alert_threshold)
        is_warning = (score_psi >= self.psi_warning_threshold) and not is_drift

        if is_drift:
            status = DriftStatus.DRIFT_DETECTED
            if self._drift_onset_index is None:
                self._drift_onset_index = self._total_samples_observed
            self._sample_count_since_drift_onset = (
                self._total_samples_observed - self._drift_onset_index + 1
            )
        elif is_warning:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.STABLE
            self._drift_onset_index = None
            self._sample_count_since_drift_onset = 0

        drift_alert = bool(is_drift)
        recalibration_triggered = False

        # Construct candidate telemetry
        telemetry = DriftTelemetry(
            window_samples=window_len,
            score_psi=score_psi,
            score_wasserstein=score_w1,
            feature_psi=feat_psi,
            feature_wasserstein=feat_w1,
            drift_status=status.value,
            drift_alert=drift_alert,
            sample_count_since_drift_onset=self._sample_count_since_drift_onset,
            last_recalibration_timestamp=self._last_recalibration_timestamp,
            current_thresholds=dict(self._current_thresholds),
            recalibration_triggered=False,
        )

        # 4. Automatic Threshold Recalibration Hook
        if drift_alert and self.auto_recalibrate:
            recalibration_triggered = True
            iso_ts = _format_timestamp(timestamp)
            self._last_recalibration_timestamp = iso_ts

            # Primary recalibration strategy
            new_thresholds = self.recalibration_strategy(telemetry, self._current_thresholds)
            self._current_thresholds.update(new_thresholds)

            # Invoke custom registered hooks
            for hook in self._custom_hooks:
                hook_thresholds = hook(telemetry, self._current_thresholds)
                if hook_thresholds and isinstance(hook_thresholds, dict):
                    self._current_thresholds.update(hook_thresholds)

            telemetry.current_thresholds = dict(self._current_thresholds)
            telemetry.last_recalibration_timestamp = self._last_recalibration_timestamp
            telemetry.recalibration_triggered = True

        return telemetry

    def get_telemetry(self) -> DriftTelemetry:
        """Retrieve the latest cached or computed drift telemetry."""
        if self._last_telemetry is not None:
            return self._last_telemetry
        return self._evaluate_drift()

    def reset_thresholds(self) -> None:
        """Reset decision boundaries to default baseline values."""
        self._current_thresholds = {
            "tau_challenge": self.initial_tau_challenge,
            "tau_block": self.initial_tau_block,
        }

    def reset(self) -> None:
        """Reset all stateful windows, counters, and thresholds."""
        self._score_window.clear()
        for w in self._feature_windows.values():
            w.clear()
        self._reference_scores.clear()
        for w in self._reference_features.values():
            w.clear()
        self._total_samples_observed = 0
        self._drift_onset_index = None
        self._sample_count_since_drift_onset = 0
        self._last_recalibration_timestamp = None
        self._last_telemetry = None
        self.reset_thresholds()
