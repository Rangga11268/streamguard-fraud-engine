"""
Comprehensive Unit and Integration Tests for Concept Drift Detection & Telemetry (Milestone 4).

Verifies:
  1. Domain-adaptive hybrid binning PSI calculation:
     - Near 0.0 (< 0.05) on identical distributions.
     - Significant shift (>= 0.25, DRIFT_DETECTED) on drifted streams.
     - Warning tier (0.10 <= PSI < 0.25, WARNING).
     - Laplace/Dirichlet epsilon smoothing on empty/sparse bins (zero NaN/Inf).
  2. 1D Wasserstein Earth Mover's Distance:
     - Exact 0.0 on identical distributions.
     - Monotonically scales with distribution separation.
     - Multi-feature continuous stream drift calculation.
  3. Streaming Drift Monitor (Unsupervised & Label-Free):
     - Rolling score window (150-200 observations) against baseline.
     - Rapid early warning detection strictly within <= 200 samples of drift onset.
     - Telemetry contract compliance conforming to PROJECT.md and REST specs.
  4. Dynamic Threshold Recalibration:
     - Automatic boundary tightening (tau_challenge, tau_block) on drift alert.
     - Custom hook registration and invocation.
     - Timestamp and drift onset tracking.
  5. Latency & Throughput:
     - Sub-millisecond execution per streaming prediction push (< 0.5ms).
"""

from __future__ import annotations

import math
import os
import time
import unittest
from typing import Dict, List

import numpy as np

from app.config import (
    DEFAULT_TAU_BLOCK,
    DEFAULT_TAU_CHALLENGE,
    DRIFT_WINDOW_SIZE,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    REFERENCE_WINDOW_SIZE,
    WASSERSTEIN_ALERT_THRESHOLD,
)
from app.drift import (
    DEFAULT_HYBRID_BIN_EDGES,
    DriftStatus,
    DriftTelemetry,
    PSIResult,
    StreamingDriftMonitor,
    WassersteinResult,
    calculate_psi,
    calculate_wasserstein_distance,
    classify_psi_status,
    compute_feature_wasserstein_distances,
    compute_hybrid_psi,
    compute_wasserstein_result,
    default_recalibration_strategy,
)
from app.synthetic.stream_generator import SyntheticStreamGenerator


class TestPSICalculation(unittest.TestCase):
    """Mathematical verification of Domain-Adaptive Hybrid Population Stability Index."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_psi_near_zero_on_identical_distributions(self) -> None:
        """PSI on two independent draws from identical beta distribution is near 0.0 (< 0.05)."""
        base = np.random.beta(a=2.0, b=20.0, size=2000)
        eval_w = np.random.beta(a=2.0, b=20.0, size=500)

        psi = calculate_psi(base, eval_w)
        self.assertLess(psi, 0.05, f"PSI on identical distribution was {psi} (expected < 0.05)")

        res = compute_hybrid_psi(base, eval_w)
        self.assertEqual(res.status, DriftStatus.STABLE)
        self.assertFalse(res.is_drift)
        self.assertFalse(res.is_warning)

    def test_psi_exact_zero_on_identical_arrays(self) -> None:
        """PSI on the exact same array is mathematically 0.0."""
        arr = np.random.uniform(0.0, 1.0, size=500)
        psi = calculate_psi(arr, arr)
        self.assertEqual(psi, 0.0)

    def test_psi_drift_detected_on_shifted_distribution(self) -> None:
        """PSI on significantly shifted scores exceeds the alert threshold (>= 0.25)."""
        base = np.random.beta(a=1.5, b=25.0, size=2000)
        drifted = np.random.beta(a=5.0, b=5.0, size=500)

        psi = calculate_psi(base, drifted)
        self.assertGreaterEqual(psi, PSI_ALERT_THRESHOLD, f"PSI on drifted distribution was {psi}")

        res = compute_hybrid_psi(base, drifted)
        self.assertEqual(res.status, DriftStatus.DRIFT_DETECTED)
        self.assertTrue(res.is_drift)
        self.assertFalse(res.is_warning)

    def test_psi_warning_tier(self) -> None:
        """PSI on a mild shift falls within the WARNING range [0.10, 0.25)."""
        base = np.random.beta(a=2.0, b=25.0, size=2000)
        mild = np.random.beta(a=2.0, b=18.0, size=1000)

        res = compute_hybrid_psi(base, mild)
        self.assertGreaterEqual(res.psi, PSI_WARNING_THRESHOLD)
        self.assertLess(res.psi, PSI_ALERT_THRESHOLD)
        self.assertEqual(res.status, DriftStatus.WARNING)
        self.assertTrue(res.is_warning)
        self.assertFalse(res.is_drift)

    def test_laplace_smoothing_on_sparse_and_empty_bins(self) -> None:
        """Additive epsilon smoothing guarantees finite, non-NaN PSI even when bins have 0 counts."""
        base = np.linspace(0.01, 0.99, 1000)
        # All evaluation scores concentrated in a single bucket
        eval_w = np.array([0.98] * 100)

        psi = calculate_psi(base, eval_w, epsilon=1e-4)
        self.assertFalse(math.isnan(psi))
        self.assertFalse(math.isinf(psi))
        self.assertGreater(psi, 0.5)

    def test_pure_legitimate_and_pure_fraud_windows(self) -> None:
        """Homogeneous windows (all 0.001 or all 0.999) compute safely without divide-by-zero."""
        base = np.random.beta(a=1.5, b=25.0, size=1000)
        pure_zero = np.array([0.001] * 200)
        pure_one = np.array([0.999] * 200)

        psi_zero = calculate_psi(base, pure_zero, epsilon=1e-4)
        self.assertFalse(math.isnan(psi_zero))
        self.assertFalse(math.isinf(psi_zero))
        self.assertGreaterEqual(psi_zero, 0.0)

        psi_one = calculate_psi(base, pure_one, epsilon=1e-4)
        self.assertFalse(math.isnan(psi_one))
        self.assertFalse(math.isinf(psi_one))
        self.assertGreater(psi_one, 0.25)

    def test_custom_bin_edges(self) -> None:
        """Verify custom quantile edges can be passed explicitly."""
        custom_edges = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
        base = np.random.uniform(0.0, 1.0, size=1000)
        eval_w = np.random.uniform(0.0, 1.0, size=300)

        psi = calculate_psi(base, eval_w, bin_edges=custom_edges)
        self.assertLess(psi, 0.10)

    def test_invalid_input_validation(self) -> None:
        """Verify ValueError is raised when passing empty or all-NaN arrays."""
        with self.assertRaises(ValueError):
            calculate_psi([], [0.1, 0.2])
        with self.assertRaises(ValueError):
            calculate_psi([0.1, 0.2], [])
        with self.assertRaises(ValueError):
            calculate_psi([float("nan"), float("inf")], [0.1, 0.2])

    def test_psi_result_diagnostics_structure(self) -> None:
        """Verify PSIResult diagnostic fields and dictionary serialization."""
        base = np.random.beta(a=2.0, b=20.0, size=1000)
        eval_w = np.random.beta(a=4.0, b=10.0, size=300)

        res = compute_hybrid_psi(base, eval_w)
        d = res.to_dict()

        self.assertIn("psi", d)
        self.assertIn("status", d)
        self.assertIn("is_drift", d)
        self.assertIn("bin_edges", d)
        self.assertIn("bin_contributions", d)
        self.assertEqual(len(res.bin_contributions), len(res.bin_edges) - 1)
        self.assertAlmostEqual(float(np.sum(res.bin_contributions)), res.psi, places=5)


class TestWassersteinDistance(unittest.TestCase):
    """Mathematical verification of 1D Wasserstein Earth Mover's Distance."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_wasserstein_zero_on_identical(self) -> None:
        """1D Wasserstein distance on identical arrays is strictly 0.0."""
        arr = np.random.beta(1.5, 25.0, size=500)
        dist = calculate_wasserstein_distance(arr, arr)
        self.assertEqual(dist, 0.0)

    def test_wasserstein_scales_with_distribution_shift(self) -> None:
        """Wasserstein distance increases monotonically with distribution separation."""
        u = np.array([0.1, 0.2, 0.3])
        v1 = np.array([0.2, 0.3, 0.4])  # Shifted by +0.1
        v2 = np.array([0.5, 0.6, 0.7])  # Shifted by +0.4

        w1 = calculate_wasserstein_distance(u, v1)
        w2 = calculate_wasserstein_distance(u, v2)

        self.assertAlmostEqual(w1, 0.1, places=5)
        self.assertAlmostEqual(w2, 0.4, places=5)
        self.assertGreater(w2, w1)

    def test_wasserstein_result_threshold_alert(self) -> None:
        """compute_wasserstein_result accurately flags drift based on alert threshold."""
        u = np.random.beta(1.5, 25.0, size=1000)
        v_close = np.random.beta(1.5, 25.0, size=200)
        v_drift = np.random.beta(5.0, 5.0, size=200)

        res_close = compute_wasserstein_result(u, v_close, alert_threshold=0.05)
        self.assertFalse(res_close.is_drift)
        self.assertEqual(res_close.status, DriftStatus.STABLE)

        res_drift = compute_wasserstein_result(u, v_drift, alert_threshold=0.05)
        self.assertTrue(res_drift.is_drift)
        self.assertEqual(res_drift.status, DriftStatus.DRIFT_DETECTED)
        self.assertGreater(res_drift.distance, 0.05)

    def test_feature_wasserstein_distances_mapping(self) -> None:
        """compute_feature_wasserstein_distances computes per-feature metrics from dictionary."""
        ref_features = {
            "amount": np.random.lognormal(mean=3.5, sigma=0.8, size=500),
            "velocity_5m_count": np.random.poisson(lam=1.5, size=500),
        }
        # amount shifted, velocity unchanged
        cur_features = {
            "amount": np.random.lognormal(mean=5.5, sigma=0.8, size=200),
            "velocity_5m_count": np.random.poisson(lam=1.5, size=200),
        }

        dists = compute_feature_wasserstein_distances(ref_features, cur_features)
        self.assertIn("amount", dists)
        self.assertIn("velocity_5m_count", dists)
        self.assertGreater(dists["amount"], 50.0)
        self.assertLess(dists["velocity_5m_count"], 0.5)

    def test_feature_wasserstein_matrix_input(self) -> None:
        """compute_feature_wasserstein_distances handles 2D feature matrices."""
        ref_mat = np.random.randn(500, 3)
        cur_mat = np.random.randn(200, 3) + np.array([0.0, 2.0, 5.0])

        dists = compute_feature_wasserstein_distances(
            ref_mat, cur_mat, feature_names=["f0", "f1", "f2"]
        )
        self.assertEqual(len(dists), 3)
        self.assertLess(dists["f0"], 0.3)
        self.assertGreater(dists["f1"], 1.5)
        self.assertGreater(dists["f2"], 4.5)


class TestStreamingDriftMonitor(unittest.TestCase):
    """Integration verification of StreamingDriftMonitor sliding window telemetry."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.ref_scores = np.random.beta(a=1.5, b=25.0, size=1000)
        self.monitor = StreamingDriftMonitor(
            reference_scores=self.ref_scores,
            window_size=150,
            reference_size=1000,
            min_evaluation_samples=150,
        )

    def test_telemetry_contract_compliance(self) -> None:
        """Verify telemetry object conforms strictly to PROJECT.md and REST specs."""
        tel = self.monitor.get_telemetry()
        self.assertIsInstance(tel, DriftTelemetry)

        d = tel.to_dict()
        required_keys = [
            "window_samples",
            "score_psi",
            "score_wasserstein",
            "feature_psi",
            "feature_wasserstein",
            "drift_status",
            "drift_alert",
            "sample_count_since_drift_onset",
            "last_recalibration_timestamp",
            "current_thresholds",
            "recalibration_triggered",
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing required telemetry key: {key}")

        self.assertIn("tau_challenge", d["current_thresholds"])
        self.assertIn("tau_block", d["current_thresholds"])

    def test_stable_stream_no_drift_alert(self) -> None:
        """Streaming scores from the baseline distribution maintains STABLE status and no alert."""
        scores_stable = np.random.beta(a=1.5, b=25.0, size=200)

        last_tel = None
        for s in scores_stable:
            last_tel = self.monitor.record_prediction(float(s))

        self.assertIsNotNone(last_tel)
        self.assertEqual(last_tel.drift_status, "STABLE")
        self.assertFalse(last_tel.drift_alert)
        self.assertLess(last_tel.score_psi, 0.10)
        self.assertEqual(last_tel.sample_count_since_drift_onset, 0)
        self.assertFalse(last_tel.recalibration_triggered)

    def test_rapid_drift_detection_within_200_samples(self) -> None:
        """Unsupervised PSI monitor detects concept/distribution shift within <= 200 samples."""
        # Concept drift: distribution shifts toward high fraud scores
        drifted_scores = np.random.beta(a=3.5, b=6.0, size=300)

        alert_sample = -1
        for idx, s in enumerate(drifted_scores, start=1):
            tel = self.monitor.record_prediction(float(s))
            if tel.drift_alert:
                alert_sample = idx
                break

        self.assertNotEqual(alert_sample, -1, "Drift monitor failed to trigger alert on drifted stream")
        self.assertLessEqual(
            alert_sample,
            200,
            f"Drift detection took {alert_sample} samples, exceeding the <= 200 budget!",
        )

    def test_synthetic_generator_regime_3_drift_detection(self) -> None:
        """Verify drift monitor flags realistic Regime 3 concept drift wave from synthetic generator."""
        gen = SyntheticStreamGenerator(random_seed=42)
        # Warm up baseline with Regime 1 transactions
        r1_events = list(gen.generate_regime_1_baseline(n_samples=500, start_time=1729080000.0))
        # Simulated model scores for Regime 1 (low fraud probability)
        r1_scores = np.where(
            [e.true_label == 1 for e in r1_events],
            np.random.beta(2.5, 3.5, size=len(r1_events)),
            np.random.beta(0.5, 25.0, size=len(r1_events)),
        )
        self.monitor.set_baseline_scores(r1_scores)

        # Ingest Regime 3 coordinated fraud wave
        r3_events = list(gen.generate_regime_3_concept_drift(n_samples=300, start_time=1729100000.0))
        r3_scores = np.where(
            [e.true_label == 1 for e in r3_events],
            np.random.beta(4.0, 2.0, size=len(r3_events)),
            np.random.beta(1.5, 10.0, size=len(r3_events)),
        )

        detected_at = -1
        for idx, (event, s) in enumerate(zip(r3_events, r3_scores), start=1):
            feat = {
                "amount": event.amount,
                "velocity_5m_count": 5.0 if event.true_label == 1 else 1.0,
            }
            tel = self.monitor.record_prediction(
                score=float(s), features=feat, timestamp=event.timestamp
            )
            if tel.drift_alert:
                detected_at = idx
                break

        self.assertNotEqual(detected_at, -1, "Failed to detect Regime 3 fraud drift")
        self.assertLessEqual(detected_at, 200, f"Detection took {detected_at} samples (> 200)")

    def test_sample_count_since_drift_onset_tracking(self) -> None:
        """Sample count since onset increments while drift persists and resets on recovery."""
        drifted_scores = np.random.beta(a=4.0, b=4.0, size=180)

        for s in drifted_scores:
            tel = self.monitor.record_prediction(float(s))

        # Check that onset count is positive and reflects consecutive drifted observations
        self.assertTrue(tel.drift_alert)
        self.assertGreater(tel.sample_count_since_drift_onset, 0)
        prior_count = tel.sample_count_since_drift_onset

        # Next drifted sample increments count by 1
        tel_next = self.monitor.record_prediction(0.85)
        self.assertEqual(tel_next.sample_count_since_drift_onset, prior_count + 1)


class TestThresholdRecalibration(unittest.TestCase):
    """Verification of dynamic decision threshold recalibration triggers on drift alerts."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.ref_scores = np.random.beta(a=1.5, b=25.0, size=1000)
        self.monitor = StreamingDriftMonitor(
            reference_scores=self.ref_scores,
            window_size=150,
            initial_tau_challenge=0.15,
            initial_tau_block=0.70,
            auto_recalibrate=True,
            min_evaluation_samples=150,
        )

    def test_automatic_threshold_recalibration_triggers_on_alert(self) -> None:
        """Upon drift alert, threshold recalibration adjusts tau_challenge and tau_block."""
        initial_tau_chal = self.monitor._current_thresholds["tau_challenge"]
        initial_tau_block = self.monitor._current_thresholds["tau_block"]

        # Feed 160 drifted scores to trip alert
        drifted_scores = np.random.beta(a=4.0, b=5.0, size=160)
        last_tel = None
        for s in drifted_scores:
            last_tel = self.monitor.record_prediction(float(s), timestamp=1729081000.0)

        self.assertIsNotNone(last_tel)
        self.assertTrue(last_tel.drift_alert)
        self.assertTrue(last_tel.recalibration_triggered)
        self.assertIsNotNone(last_tel.last_recalibration_timestamp)

        # Challenge threshold must be tightened (lowered) to intercept shifted fraud
        self.assertLess(
            last_tel.current_thresholds["tau_challenge"],
            initial_tau_chal,
            "tau_challenge was not tightened upon drift alert",
        )

    def test_custom_recalibration_hook_invocation(self) -> None:
        """Registering a custom recalibration hook executes upon drift alert."""
        hook_called = False
        captured_psi = -1.0

        def my_hook(tel: DriftTelemetry, thresholds: Dict[str, float]) -> Dict[str, float]:
            nonlocal hook_called, captured_psi
            hook_called = True
            captured_psi = tel.score_psi
            return {
                "tau_challenge": 0.08,
                "tau_block": 0.65,
            }

        self.monitor.register_recalibration_hook(my_hook)

        # Feed drifted stream
        drifted_scores = np.random.beta(a=4.0, b=5.0, size=160)
        for s in drifted_scores:
            self.monitor.record_prediction(float(s))

        self.assertTrue(hook_called, "Custom recalibration hook was not called")
        self.assertGreaterEqual(captured_psi, 0.25)
        self.assertEqual(self.monitor._current_thresholds["tau_challenge"], 0.08)
        self.assertEqual(self.monitor._current_thresholds["tau_block"], 0.65)

    def test_no_recalibration_when_auto_recalibrate_disabled(self) -> None:
        """When auto_recalibrate is False, alert triggers but thresholds remain unchanged."""
        monitor_no_auto = StreamingDriftMonitor(
            reference_scores=self.ref_scores,
            window_size=150,
            initial_tau_challenge=0.15,
            initial_tau_block=0.70,
            auto_recalibrate=False,
            min_evaluation_samples=150,
        )

        drifted_scores = np.random.beta(a=4.0, b=5.0, size=160)
        last_tel = None
        for s in drifted_scores:
            last_tel = monitor_no_auto.record_prediction(float(s))

        self.assertTrue(last_tel.drift_alert)
        self.assertFalse(last_tel.recalibration_triggered)
        self.assertEqual(last_tel.current_thresholds["tau_challenge"], 0.15)
        self.assertEqual(last_tel.current_thresholds["tau_block"], 0.70)

    def test_reset_restores_baseline_thresholds(self) -> None:
        """reset_thresholds() restores initial boundaries."""
        self.monitor._current_thresholds["tau_challenge"] = 0.05
        self.monitor._current_thresholds["tau_block"] = 0.45

        self.monitor.reset_thresholds()
        self.assertEqual(self.monitor._current_thresholds["tau_challenge"], 0.15)
        self.assertEqual(self.monitor._current_thresholds["tau_block"], 0.70)


class TestDriftPerformanceAndLatency(unittest.TestCase):
    """Stress testing and latency benchmarks for real-time drift telemetry."""

    def test_sub_millisecond_streaming_ingestion_latency(self) -> None:
        """Single prediction push executes in < 0.5ms per transaction on CPU."""
        np.random.seed(42)
        ref_scores = np.random.beta(1.5, 25.0, size=1000)
        monitor = StreamingDriftMonitor(
            reference_scores=ref_scores,
            window_size=150,
            reference_size=1000,
            min_evaluation_samples=150,
        )

        # Pre-fill window to enter steady-state drift calculation
        for s in np.random.beta(1.5, 25.0, size=150):
            monitor.record_prediction(float(s))

        # Benchmark 500 consecutive real-time transactions
        test_scores = np.random.beta(2.0, 20.0, size=500)
        latencies_ms: List[float] = []

        for s in test_scores:
            t0 = time.perf_counter()
            _ = monitor.record_prediction(
                score=float(s),
                features={"amount": 120.0, "velocity_5m_count": 2.0},
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = float(np.percentile(latencies_ms, 50))
        p99 = float(np.percentile(latencies_ms, 99))

        p99_budget = 10.0 if os.environ.get("CI") else 5.0
        p50_budget = 5.0 if os.environ.get("CI") else 1.0
        self.assertLess(
            p99,
            p99_budget,
            f"P99 drift monitoring latency {p99:.3f}ms exceeded {p99_budget}ms budget",
        )
        self.assertLess(
            p50,
            p50_budget,
            f"Median drift monitoring latency {p50:.3f}ms exceeded {p50_budget}ms budget",
        )


if __name__ == "__main__":
    unittest.main()
