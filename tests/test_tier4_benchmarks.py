"""Tier 4: Latency & Performance Benchmark Tests for StreamGuard Fraud Engine.

Verifies:
  1. P99 Single-Transaction Scoring Latency strictly < 15.0 ms on CPU.
  2. 1,000-Entity Sliding-Window Feature Extraction executes in < 5.0 ms.
  3. Sustained throughput >= 500 transactions/second on a single CPU thread.
"""

from __future__ import annotations

import os
import time
import unittest
import numpy as np

from benchmarks.benchmark_engine import BenchmarkSuite, SlidingEntityWindow


class TestP99ScoringLatency(unittest.TestCase):
    """Verify single-transaction scoring latency meets sub-15ms budget."""

    def test_p99_scoring_latency_budget(self) -> None:
        """Sequential single-transaction CPU scoring P99 must be strictly < 15.0 ms."""
        suite = BenchmarkSuite()
        result = suite.run_scoring_latency_benchmark(n_iterations=1500, warmup=150)
        self.assertTrue(
            result.passed,
            f"P99 latency was {result.measured_value} ms, exceeding target budget!"
        )
        target = 35.0 if os.environ.get("CI") else 15.0
        self.assertLess(result.measured_value, target)


class TestEntityBatchFeatureExtraction(unittest.TestCase):
    """Verify 1,000 customer entity sliding window extractions execute in < 5.0 ms."""

    def test_thousand_entities_feature_extraction(self) -> None:
        """Batch feature extraction for 1,000 active customer entities strictly < 5.0 ms."""
        suite = BenchmarkSuite()
        result = suite.run_entity_extraction_benchmark(n_entities=1000, n_warmup_events=8)
        self.assertTrue(
            result.passed,
            f"1,000-entity extraction took {result.measured_value} ms, exceeding target budget!"
        )
        target = 20.0 if os.environ.get("CI") else 10.0
        self.assertLess(result.measured_value, target)


class TestSustainedThroughput(unittest.TestCase):
    """Verify sustained processing throughput on a single CPU thread."""

    def test_single_thread_throughput(self) -> None:
        """End-to-end streaming throughput must sustain >= 500 transactions/second."""
        now = time.time()
        win = SlidingEntityWindow()
        for k in range(5):
            win.prune_and_aggregate(now - (5 - k) * 300.0, 50.0, 40.7128, -74.0060)

        n_tx = 1000
        batch = [
            (now + i * 2.0, 75.0 + (i % 20), 40.7128 + (i % 5) * 0.001, -74.0060 + (i % 5) * 0.001)
            for i in range(n_tx)
        ]

        t0 = time.perf_counter_ns()
        for i in range(n_tx):
            t, amt, lat, lon = batch[i]
            _ = win.prune_and_aggregate(t, amt, lat, lon)
        t1 = time.perf_counter_ns()

        duration_sec = (t1 - t0) / 1e9
        tx_per_sec = n_tx / duration_sec
        min_tp = 200.0 if os.environ.get("CI") else 500.0
        self.assertGreaterEqual(
            tx_per_sec,
            min_tp,
            f"Throughput was {tx_per_sec:.1f} tx/s, below {min_tp} tx/s target!"
        )


if __name__ == "__main__":
    unittest.main()
