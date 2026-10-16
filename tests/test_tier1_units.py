"""Tier 1: Functional Unit Tests for StreamGuard Fraud Engine.

Verifies:
  1. Sliding-window multi-scale deques, velocity ratios, and entity isolation.
  2. Haversine distance, speed-over-ground, date-line wrap, and impossible travel flags.
  3. Dual-memory asynchronous delayed feedback queues and temporal isolation.
  4. Population Stability Index (PSI) mathematical invariants and Dirichlet smoothing.
  5. 3-Tier cost-matrix decision routing logic and expected loss boundaries.
  6. Cost-sensitive tree classification probability outputs on contiguous arrays.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from app.synthetic.stream_generator import (
    DelayedFeedbackQueue,
    InvestigatorFeedback,
    ChargebackFeedback,
    StreamRegime,
    SyntheticStreamGenerator,
    TransactionEntityEvent,
)
from benchmarks.benchmark_engine import (
    SlidingEntityWindow,
    calculate_psi,
    evaluate_cost_matrix_policy,
    haversine_distance_km,
)


class TestSlidingWindowFeatureStore(unittest.TestCase):
    """Unit tests for multi-scale sliding windows and velocity aggregations."""

    def setUp(self) -> None:
        self.window = SlidingEntityWindow()
        self.t0 = 1729080000.0  # reference epoch

    def test_single_event_initialization(self) -> None:
        """First event initializes with 0 prior historical counts and default baseline ratio."""
        res = self.window.prune_and_aggregate(
            current_time=self.t0,
            new_amount=100.0,
            new_lat=40.7128,
            new_lon=-74.0060,
        )
        cnt_5m, cnt_1h, cnt_24h, amt_ratio, speed_kmh, imp_flag = res
        self.assertEqual(cnt_5m, 0, "Prior 5m count must be 0 before committing first event")
        self.assertEqual(cnt_1h, 0, "Prior 1h count must be 0 before committing first event")
        self.assertEqual(cnt_24h, 0, "Prior 24h count must be 0 before committing first event")
        self.assertAlmostEqual(amt_ratio, 1.0, delta=0.05)
        self.assertEqual(speed_kmh, 0.0)
        self.assertEqual(imp_flag, 0.0)

        # Second event immediately following must see prior event
        res2 = self.window.prune_and_aggregate(
            current_time=self.t0 + 10.0,
            new_amount=150.0,
            new_lat=40.7128,
            new_lon=-74.0060,
        )
        cnt_5m_2, cnt_1h_2, cnt_24h_2, amt_ratio_2, _, _ = res2
        self.assertEqual(cnt_5m_2, 1)
        self.assertEqual(cnt_1h_2, 1)
        self.assertEqual(cnt_24h_2, 1)
        self.assertAlmostEqual(amt_ratio_2, 1.5, delta=0.05)

    def test_multi_scale_temporal_pruning(self) -> None:
        """Events older than 5m, 1h, and 24h are correctly pruned."""
        # Event 1 at t0
        self.window.prune_and_aggregate(self.t0, 50.0, 40.7128, -74.0060)
        # Event 2 at t0 + 400s (>5m = 300s, <=1h = 3600s from t0)
        res2 = self.window.prune_and_aggregate(self.t0 + 400.0, 75.0, 40.7128, -74.0060)
        cnt_5m, cnt_1h, cnt_24h, _, _, _ = res2
        self.assertEqual(cnt_5m, 0, "Event 1 (400s old) is outside 5m (300s) window")
        self.assertEqual(cnt_1h, 1, "Event 1 is inside 1h window")
        self.assertEqual(cnt_24h, 1, "Event 1 is inside 24h window")

        # Event 3 at t0 + 4100s (>1h = 3600s from event 2 at t0+400s)
        res3 = self.window.prune_and_aggregate(self.t0 + 4100.0, 120.0, 40.7128, -74.0060)
        cnt_5m, cnt_1h, cnt_24h, _, _, _ = res3
        self.assertEqual(cnt_5m, 0)
        self.assertEqual(cnt_1h, 0, "Events 1 and 2 are older than 1h from t0+4100s")
        self.assertEqual(cnt_24h, 2, "Events 1 and 2 are still in 24h window")

        # Event 4 at t0 + 100000s (>24h = 86400s from event 3 at t0+4100s)
        res4 = self.window.prune_and_aggregate(self.t0 + 100000.0, 200.0, 40.7128, -74.0060)
        cnt_5m, cnt_1h, cnt_24h, _, _, _ = res4
        self.assertEqual(cnt_5m, 0)
        self.assertEqual(cnt_1h, 0)
        self.assertEqual(cnt_24h, 0, "All prior events older than 24h are evicted")

    def test_entity_isolation(self) -> None:
        """Transactions for customer A must not contaminate customer B."""
        win_a = SlidingEntityWindow()
        win_b = SlidingEntityWindow()

        win_a.prune_and_aggregate(self.t0, 100.0, 40.7128, -74.0060)
        win_a.prune_and_aggregate(self.t0 + 10.0, 200.0, 40.7128, -74.0060)

        res_b = win_b.prune_and_aggregate(self.t0 + 20.0, 30.0, 34.0522, -118.2437)
        cnt_5m_b, _, cnt_24h_b, _, _, _ = res_b
        self.assertEqual(cnt_5m_b, 0, "Customer B has 0 prior transactions")
        self.assertEqual(cnt_24h_b, 0, "Customer B has 0 prior transactions")
        self.assertEqual(len(win_a.events), 2)
        self.assertEqual(len(win_b.events), 1)


class TestHaversineGeoVelocity(unittest.TestCase):
    """Unit tests for spatial distance, velocity, and jump detection."""

    def test_zero_distance_identical_coordinates(self) -> None:
        """Identical coordinates produce zero km distance."""
        dist = haversine_distance_km(40.7128, -74.0060, 40.7128, -74.0060)
        self.assertEqual(dist, 0.0)

    def test_known_intercity_distance(self) -> None:
        """New York (40.7128, -74.0060) to London (51.5074, -0.1278) is approx 5570 km."""
        dist = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278)
        self.assertGreater(dist, 5500.0)
        self.assertLess(dist, 5650.0)

    def test_date_line_wrap_around(self) -> None:
        """Coordinates crossing the 180th meridian must not produce 350-degree error."""
        # Point A at Lon 179.5, Point B at Lon -179.5 near equator
        dist = haversine_distance_km(0.0, 179.5, 0.0, -179.5)
        # 1 degree of longitude at equator is approx 111.32 km
        self.assertLess(dist, 150.0, "Date-line wrap must compute shortest Great Circle distance")

    def test_impossible_travel_jump_flag(self) -> None:
        """NYC to London in 15 minutes implies speed > 20,000 km/h, triggering flag."""
        win = SlidingEntityWindow()
        t0 = 1729080000.0
        # First event in NYC
        win.prune_and_aggregate(t0, 25.0, 40.7128, -74.0060)
        # Second event 900s (15 min) later in London
        res = win.prune_and_aggregate(t0 + 900.0, 1200.0, 51.5074, -0.1278)
        _, _, _, _, speed_kmh, imp_flag = res
        self.assertGreater(speed_kmh, 15000.0)
        self.assertEqual(imp_flag, 1.0)


class TestDualMemoryQueues(unittest.TestCase):
    """Unit tests for asynchronous delayed feedback and temporal isolation."""

    def setUp(self) -> None:
        self.queue = DelayedFeedbackQueue()
        self.t0 = 1729080000.0

    def test_investigator_feedback_temporal_gating(self) -> None:
        """Fast feedback scheduled at t+2h is invisible at t+1h and emitted at t+2.5h."""
        self.queue.schedule_investigator_feedback(
            transaction_id="tx_test_01",
            tx_timestamp=self.t0,
            is_fraud=True,
            delay_seconds=7200.0,  # 2 hours
            investigator_id="inv_42",
        )
        self.assertEqual(self.queue.pending_count(), 1)

        # Poll at t0 + 3600s (1 hour): event must NOT mature yet
        inv_early, cb_early = self.queue.poll(self.t0 + 3600.0)
        self.assertEqual(len(inv_early), 0, "Temporal leakage detected in investigator queue!")
        self.assertEqual(len(cb_early), 0)
        self.assertEqual(self.queue.pending_count(), 1)

        # Poll at t0 + 7500s (> 2 hours): event must be delivered
        inv_ready, cb_ready = self.queue.poll(self.t0 + 7500.0)
        self.assertEqual(len(inv_ready), 1)
        self.assertEqual(inv_ready[0].transaction_id, "tx_test_01")
        self.assertEqual(inv_ready[0].verdict, "FRAUD")
        self.assertEqual(self.queue.pending_count(), 0)

    def test_chargeback_feedback_delayed_maturation(self) -> None:
        """Chargeback scheduled at t+20d is invisible at t+10d and emitted at t+21d."""
        delay_20d = 20.0 * 86400.0
        self.queue.schedule_chargeback_feedback(
            transaction_id="tx_test_02",
            tx_timestamp=self.t0,
            amount=450.0,
            is_fraud=True,
            delay_seconds=delay_20d,
        )

        # Poll at t0 + 10 days
        _, cb_10d = self.queue.poll(self.t0 + 10.0 * 86400.0)
        self.assertEqual(len(cb_10d), 0, "Temporal leakage in chargeback queue!")

        # Poll at t0 + 21 days
        _, cb_21d = self.queue.poll(self.t0 + 21.0 * 86400.0)
        self.assertEqual(len(cb_21d), 1)
        self.assertEqual(cb_21d[0].chargeback_amount, 450.0)
        self.assertTrue(cb_21d[0].is_fraud)


class TestPSIDriftMath(unittest.TestCase):
    """Unit tests for Population Stability Index calculation and numerical stability."""

    def test_identical_distributions_yield_near_zero(self) -> None:
        """Identical score distributions produce PSI approx 0.0."""
        np.random.seed(42)
        base = np.random.beta(a=2.0, b=20.0, size=2000)
        eval_w = np.random.beta(a=2.0, b=20.0, size=500)
        psi = calculate_psi(base, eval_w, num_bins=10)
        self.assertLess(psi, 0.05, f"PSI on identical distribution was {psi}")

    def test_shifted_distribution_yields_significant_drift(self) -> None:
        """Scores shifted from low to high produce PSI > 0.25."""
        np.random.seed(42)
        base = np.random.beta(a=1.5, b=25.0, size=2000)
        drifted = np.random.beta(a=5.0, b=5.0, size=500)
        psi = calculate_psi(base, drifted, num_bins=10)
        self.assertGreater(psi, 0.25, f"PSI on drifted distribution was {psi}")

    def test_laplace_smoothing_on_zero_count_bins(self) -> None:
        """Empty quantile bins do not cause division by zero or ln(0) errors."""
        base = np.linspace(0.01, 0.99, 1000)
        # All evaluation scores in upper bucket
        eval_w = np.array([0.98] * 100)
        psi = calculate_psi(base, eval_w, num_bins=10, epsilon=1e-4)
        self.assertFalse(math.isnan(psi))
        self.assertFalse(math.isinf(psi))
        self.assertGreater(psi, 0.5)


class TestCostMatrixRouting(unittest.TestCase):
    """Unit tests for cost-sensitive 3-tier routing boundaries."""

    def test_cost_reduction_over_static_tau(self) -> None:
        """Dynamic cost matrix routing achieves >=25% savings over static tau=0.5."""
        np.random.seed(42)
        n = 3000
        true_labels = (np.random.rand(n) < 0.02).astype(np.int32)
        probs = np.where(
            true_labels == 1,
            np.random.beta(a=2.5, b=3.5, size=n),
            np.random.beta(a=0.5, b=25.0, size=n),
        )
        amounts = np.exp(np.random.normal(loc=4.6, scale=0.9, size=n))

        loss_static, loss_3tier, reduction = evaluate_cost_matrix_policy(probs, amounts, true_labels)
        self.assertGreater(loss_static, loss_3tier)
        self.assertGreaterEqual(reduction, 25.0)


class TestFocalLossTreeWrapper(unittest.TestCase):
    """Unit tests for LightGBM CPU inference with contiguous float32 input arrays."""

    def test_contiguous_float32_compatibility(self) -> None:
        """LightGBM requires contiguous float32 to avoid Windows access violations."""
        import lightgbm as lgb
        rng = np.random.RandomState(42)
        X = np.ascontiguousarray(rng.randn(200, 16), dtype=np.float32)
        y = np.ascontiguousarray((rng.rand(200) > 0.98), dtype=np.int32)
        y[0] = 1
        y[1] = 0

        clf = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            num_leaves=7,
            n_jobs=1,
            verbose=-1,
            random_state=42,
        )
        clf.fit(X, y)

        sample = np.ascontiguousarray(rng.randn(1, 16), dtype=np.float32)
        probs = clf.predict_proba(sample)
        self.assertEqual(probs.shape, (1, 2))
        self.assertGreaterEqual(probs[0, 1], 0.0)
        self.assertLessEqual(probs[0, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
