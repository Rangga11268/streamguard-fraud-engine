"""Tier 2: Boundary & Edge Case Tests for StreamGuard Fraud Engine.

Verifies:
  1. Cold-start brand new entity with zero prior transaction history.
  2. Zero-variance customer spending history (no division by zero in variance or ratios).
  3. Out-of-order arriving timestamps (monotonic and network delay safeguards).
  4. Extreme transaction amounts (micro-transactions $0.00, $0.01 and whale transactions $10,000,000.00).
  5. Coordinate edge cases (North/South poles +-90 deg, Prime Meridian, Anti-meridian +-180 deg).
  6. Missing or malformed string fields (empty MCC, missing device ID).
  7. Single-class evaluation windows in drift monitors (0% fraud or 100% fraud) with Dirichlet smoothing.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from app.synthetic.stream_generator import (
    DelayedFeedbackQueue,
    SyntheticStreamGenerator,
    TransactionEntityEvent,
)
from benchmarks.benchmark_engine import (
    SlidingEntityWindow,
    calculate_psi,
    evaluate_cost_matrix_policy,
    haversine_distance_km,
)


class TestBrandNewEntity(unittest.TestCase):
    """Test cold-start behavior when entity has zero historical transactions."""

    def test_cold_start_window_aggregation(self) -> None:
        """First transaction on clean window produces 0 prior counts and finite ratio."""
        win = SlidingEntityWindow()
        cnt_5m, cnt_1h, cnt_24h, amt_ratio, speed_kmh, imp_flag = win.prune_and_aggregate(
            current_time=1729080000.0,
            new_amount=45.0,
            new_lat=40.7128,
            new_lon=-74.0060,
        )
        self.assertEqual(cnt_5m, 0, "Cold start entity has 0 prior 5m transactions")
        self.assertEqual(cnt_1h, 0, "Cold start entity has 0 prior 1h transactions")
        self.assertEqual(cnt_24h, 0, "Cold start entity has 0 prior 24h transactions")
        self.assertFalse(math.isnan(amt_ratio))
        self.assertFalse(math.isinf(amt_ratio))
        self.assertEqual(speed_kmh, 0.0)
        self.assertEqual(imp_flag, 0.0)


class TestZeroVarianceCustomer(unittest.TestCase):
    """Test customers with completely static spending amounts."""

    def test_identical_amounts_ratio(self) -> None:
        """Customer spending exactly $50.00 every time has amount_to_mean24h_ratio ~ 1.0."""
        win = SlidingEntityWindow()
        now = 1729080000.0
        for i in range(10):
            res = win.prune_and_aggregate(now + i * 60.0, 50.0, 40.7128, -74.0060)
        
        _, _, cnt_24h, ratio, _, _ = res
        self.assertEqual(cnt_24h, 9, "On 10th event, exactly 9 prior events exist in window")
        self.assertEqual(len(win.events), 10)
        self.assertAlmostEqual(ratio, 1.0, places=3)
        self.assertFalse(math.isnan(ratio))


class TestOutOfOrderTimestamps(unittest.TestCase):
    """Test resilience against clock skew or network delays."""

    def test_out_of_order_event_handling(self) -> None:
        """An event arriving with a timestamp older than the latest seen event."""
        win = SlidingEntityWindow()
        now = 1729080000.0
        # Event at t=100
        win.prune_and_aggregate(now + 100.0, 50.0, 40.7128, -74.0060)
        # Delayed event arriving with timestamp t=80
        res = win.prune_and_aggregate(now + 80.0, 30.0, 40.7128, -74.0060)
        _, _, cnt_24h, ratio, speed_kmh, _ = res
        self.assertGreaterEqual(cnt_24h, 1)
        self.assertFalse(math.isnan(ratio))
        # Zero or non-negative speed
        self.assertGreaterEqual(speed_kmh, 0.0)


class TestExtremeAmounts(unittest.TestCase):
    """Test numerical stability under extreme ticket amounts."""

    def test_zero_amount_authorization_ping(self) -> None:
        """Card authorization ping of $0.00 does not trigger division by zero."""
        win = SlidingEntityWindow()
        now = 1729080000.0
        res = win.prune_and_aggregate(now, 0.0, 40.7128, -74.0060)
        _, _, _, ratio, _, _ = res
        self.assertFalse(math.isnan(ratio))
        self.assertAlmostEqual(ratio, 0.0, places=4)

    def test_penny_micro_transaction(self) -> None:
        """$0.01 micro transaction is represented accurately."""
        win = SlidingEntityWindow()
        now = 1729080000.0
        res = win.prune_and_aggregate(now, 0.01, 40.7128, -74.0060)
        _, _, _, ratio, _, _ = res
        self.assertFalse(math.isnan(ratio))
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_whale_ten_million_dollar_transaction(self) -> None:
        """Whale transaction of $10,000,000.00 does not cause float overflow."""
        win = SlidingEntityWindow()
        now = 1729080000.0
        # Baseline modest history
        win.prune_and_aggregate(now, 50.0, 40.7128, -74.0060)
        res = win.prune_and_aggregate(now + 100.0, 10_000_000.0, 40.7128, -74.0060)
        _, _, _, ratio, _, _ = res
        self.assertFalse(math.isinf(ratio))
        self.assertFalse(math.isnan(ratio))
        self.assertGreater(ratio, 1000.0)


class TestCoordinatePolesAndMeridian(unittest.TestCase):
    """Test boundary conditions for spatial coordinates."""

    def test_north_pole_to_south_pole(self) -> None:
        """Distance from North Pole (+90) to South Pole (-90) is approx half Earth circumference (20,015 km)."""
        dist = haversine_distance_km(90.0, 0.0, -90.0, 0.0)
        self.assertAlmostEqual(dist, 20015.08, delta=100.0)

    def test_prime_meridian_crossing(self) -> None:
        """Crossing Lon 0.0 (Greenwich) behaves continuously."""
        dist = haversine_distance_km(51.4769, -0.0005, 51.4769, 0.0005)
        self.assertLess(dist, 0.1)

    def test_international_date_line_extremes(self) -> None:
        """Crossing Lon +180.0 to -180.0 is zero distance."""
        dist = haversine_distance_km(0.0, 180.0, 0.0, -180.0)
        self.assertAlmostEqual(dist, 0.0, places=4)


class TestMissingAndMalformedFields(unittest.TestCase):
    """Test handling of optional and missing schema fields."""

    def test_event_dict_default_fill(self) -> None:
        """TransactionEntityEvent handles missing optional fields with defaults."""
        event = TransactionEntityEvent(
            transaction_id="tx_default_01",
            customer_id="cust_0001",
            card_id="card_0001",
            amount=50.0,
            timestamp=1729080000.0,
            merchant_category_code="",  # empty MCC
            latitude=0.0,
            longitude=0.0,
        )
        self.assertEqual(event.merchant_id, "")
        self.assertEqual(event.currency, "USD")
        self.assertEqual(event.channel, "e-commerce")
        self.assertEqual(event.device_id, "")
        self.assertEqual(event.true_label, 0)
        d = event.to_dict()
        self.assertIn("transaction_id", d)
        self.assertEqual(d["amount"], 50.0)


class TestSingleClassWindow(unittest.TestCase):
    """Test PSI drift monitor with single-class / homogenous windows."""

    def test_pure_legitimate_window_psi(self) -> None:
        """A window with 100% zero/near-zero scores does not crash PSI calculation."""
        base = np.random.beta(a=1.5, b=25.0, size=1000)
        # All evaluation scores are exactly 0.001
        pure_zero = np.array([0.001] * 200)
        psi = calculate_psi(base, pure_zero, num_bins=10, epsilon=1e-4)
        self.assertFalse(math.isnan(psi))
        self.assertFalse(math.isinf(psi))
        self.assertGreaterEqual(psi, 0.0)

    def test_pure_fraud_window_psi(self) -> None:
        """A window with 100% maximum risk scores does not crash PSI calculation."""
        base = np.random.beta(a=1.5, b=25.0, size=1000)
        pure_one = np.array([0.999] * 200)
        psi = calculate_psi(base, pure_one, num_bins=10, epsilon=1e-4)
        self.assertFalse(math.isnan(psi))
        self.assertFalse(math.isinf(psi))
        self.assertGreater(psi, 0.25)


if __name__ == "__main__":
    unittest.main()
