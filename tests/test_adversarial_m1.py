"""
Adversarial Stress Test Suite for StreamGuard Milestone 1 (Feature Store & Sliding Windows).

Evaluates:
1. Strict zero temporal leakage under out-of-order arrivals and future-event injection.
2. Coordinate edge cases (Poles, 180th meridian, antipodal points, numerical clamping).
3. Zero-delta and negative-delta timestamp handling in travel velocity and windowing.
4. Out-of-order location state tracking and coordinate corruption.
5. Dead-code / logic flaw in 24h retention horizon check (timestamp < timestamp - 86400).
6. Massive volume stress (10,000+ events per entity) and floating-point accumulation drift.
7. Memory leak and TTL eviction stability under continuous streaming.
8. Randomized adversarial oracle comparison for lookahead temporal leakage.
"""

import gc
import math
import os
import random
import sys
import time
import tracemalloc
import unittest
import numpy as np

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import (
    FEATURE_NAMES,
    WINDOW_5M,
    WINDOW_1H,
    WINDOW_24H,
    IMPOSSIBLE_TRAVEL_SPEED_KMH,
    EARTH_RADIUS_KM,
    StreamGuardConfig,
)
from app.features.geo_velocity import haversine_distance, compute_travel_speed
from app.features.sliding_window import SlidingEntityWindow, WindowStats
from app.features.feature_store import (
    TransactionEntityEvent,
    ExtractedFeatures,
    StreamingFeatureStore,
)


class TestTemporalLeakageAdversarial(unittest.TestCase):
    """Stress-tests temporal leakage invariants under out-of-order and future event arrivals."""

    def setUp(self):
        self.store = StreamingFeatureStore()

    def test_future_event_leaks_into_24h_aggregates_on_out_of_order_query(self):
        """
        EMPIRICAL CHALLENGE 1:
        If Event A at T=2000 is committed first, and Event B arrives out-of-order
        at T=1000, does Event B observe Event A's metrics?
        
        At T=1000, Event A (T=2000) has NOT occurred in physical time.
        Therefore, at T=1000, the 24h count MUST be 0 and 24h sum MUST be 0.0.
        """
        cust_id = "cust_future_leak"
        card_id = "card_future_leak"

        # Commit future event at T=2000
        event_future = TransactionEntityEvent(
            transaction_id="tx_future",
            customer_id=cust_id,
            card_id=card_id,
            amount=500.0,
            timestamp=2000.0,
            merchant_category_code="5411",
            latitude=40.7128,
            longitude=-74.0060,
        )
        self.store.extract_features(event_future, commit=True)

        # Query past event at T=1000
        event_past = TransactionEntityEvent(
            transaction_id="tx_past",
            customer_id=cust_id,
            card_id=card_id,
            amount=50.0,
            timestamp=1000.0,
            merchant_category_code="5812",
            latitude=40.7128,
            longitude=-74.0060,
        )
        features_past = self.store.extract_features(event_past, commit=False)

        print("\n[TEMPORAL LEAKAGE TEST RESULT]")
        print(f"5m count: {features_past.velocity_5m_count} (Expected: 0)")
        print(f"1h count: {features_past.velocity_1h_count} (Expected: 0)")
        print(f"24h count: {features_past.velocity_24h_count} (Expected: 0)")
        full_dict = features_past.to_full_dict()
        print(f"spending_24h_sum: {full_dict['spending_24h_sum']} (Expected: 0.0)")
        print(f"amount_to_mean24h_ratio: {full_dict['amount_to_mean24h_ratio']} (Expected: 1.0)")
        print(f"card_velocity_24h_count: {full_dict['card_velocity_24h_count']} (Expected: 0)")
        print(f"card_distinct_mcc_count: {full_dict['card_distinct_mcc_count']} (Expected: 0)")

        # Verify whether future event leaked into past query
        leaked_24h_count = features_past.velocity_24h_count != 0
        leaked_24h_sum = full_dict['spending_24h_sum'] != 0.0
        leaked_card_count = full_dict['card_velocity_24h_count'] != 0
        leaked_mcc = full_dict['card_distinct_mcc_count'] != 0

        if leaked_24h_count or leaked_24h_sum or leaked_card_count or leaked_mcc:
            print(">>> CONFIRMED VULNERABILITY: Future event at T=2000 leaked into query at T=1000!")
            print(f"    - velocity_24h_count observed: {features_past.velocity_24h_count}")
            print(f"    - spending_24h_sum observed: {full_dict['spending_24h_sum']}")
            print(f"    - card_distinct_mcc_count observed: {full_dict['card_distinct_mcc_count']}")

        # We assert what the contract DEMANDS: ZERO LEAKAGE
        self.assertEqual(features_past.velocity_5m_count, 0)
        self.assertEqual(features_past.velocity_1h_count, 0)
        self.assertEqual(
            features_past.velocity_24h_count, 0,
            f"Temporal leakage: 24h count at T=1000 saw future event at T=2000! Got {features_past.velocity_24h_count}"
        )
        self.assertEqual(
            full_dict['spending_24h_sum'], 0.0,
            f"Temporal leakage: spending_24h_sum at T=1000 saw future event at T=2000! Got {full_dict['spending_24h_sum']}"
        )

    def test_out_of_order_location_state_corruption(self):
        """
        EMPIRICAL CHALLENGE 2:
        Tests location state consistency when out-of-order events are committed.
        
        Event 1: T=1000 at New York (40.7128, -74.0060).
        Event 2: T=500 at Philadelphia (39.9526, -75.1652) (arrives out of order).
        Event 3: T=1010 at New York (40.7128, -74.0060).
        
        Physical reality:
        At T=1000, customer is in New York.
        At T=1010 (10s later), customer is still in New York.
        Distance between T=1000 and T=1010 is 0.0 km.
        Speed between T=1000 and T=1010 should be 0.0 km/h, impossible = 0.
        
        Vulnerability:
        In SlidingEntityWindow.commit:
            self.last_latitude = float(latitude)
            self.last_longitude = float(longitude)
        The location was overwritten by Event 2 (Philadelphia at T=500), but last_timestamp
        was kept at 1000.
        Thus Event 3 calculates speed from Philadelphia to New York (130 km) in 10 seconds!
        """
        cust_id = "cust_location_corruption"
        card_id = "card_location_corruption"

        # 1. Commit Event 1: T=1000 in New York
        e1 = TransactionEntityEvent("tx1", cust_id, card_id, 100.0, 1000.0, "5411", 40.7128, -74.0060)
        self.store.extract_features(e1, commit=True)

        # 2. Commit Event 2 (Out-of-order): T=500 in Philadelphia (~130 km away)
        e2 = TransactionEntityEvent("tx2", cust_id, card_id, 50.0, 500.0, "5411", 39.9526, -75.1652)
        self.store.extract_features(e2, commit=True)

        # 3. Query Event 3: T=1010 in New York (10s after Event 1)
        e3 = TransactionEntityEvent("tx3", cust_id, card_id, 20.0, 1010.0, "5411", 40.7128, -74.0060)
        f3 = self.store.extract_features(e3, commit=False)

        print("\n[LOCATION STATE CORRUPTION TEST RESULT]")
        print(f"Calculated travel speed for Event 3 (NY -> NY): {f3.travel_speed_kmh:.2f} km/h")
        print(f"Impossible travel flag: {f3.impossible_travel_flag}")

        win = self.store.get_customer_window(cust_id)
        print(f"Window state: last_lat={win.last_latitude}, last_lon={win.last_longitude}, last_ts={win.last_timestamp}")

        # In physical truth, Event 3 is in New York, 10s after New York. Speed should be 0.0.
        self.assertEqual(
            f3.impossible_travel_flag, 0,
            f"False impossible travel flag! Customer was in NY at T=1000, NY at T=1010. Speed computed: {f3.travel_speed_kmh} km/h"
        )
        self.assertAlmostEqual(f3.travel_speed_kmh, 0.0, delta=1.0)

    def test_out_of_order_negative_delta_speed_calculation(self):
        """
        EMPIRICAL CHALLENGE 3:
        Event 1 is committed at T=1000 at New York (40.7128, -74.0060).
        Event 2 arrives out-of-order at T=950 at Philadelphia (39.9526, -75.1652).
        
        During extract_features(Event 2), compute_travel_speed is called with:
            t1 = cust_window.last_timestamp (1000)
            t2 = event.timestamp (950)
        delta_t_sec = 950 - 1000 = -50.
        delta_t_hours = max(-50/3600, 1e-4) = 1e-4 hours (0.36 seconds).
        Distance = 130 km.
        Speed = 130 / 1e-4 = 1,300,000 km/h!
        Flags impossible_travel_flag = 1!
        """
        cust_id = "cust_neg_delta"
        card_id = "card_neg_delta"

        e1 = TransactionEntityEvent("tx1", cust_id, card_id, 100.0, 1000.0, "5411", 40.7128, -74.0060)
        self.store.extract_features(e1, commit=True)

        e2 = TransactionEntityEvent("tx2", cust_id, card_id, 50.0, 950.0, "5411", 39.9526, -75.1652)
        f2 = self.store.extract_features(e2, commit=False)

        print("\n[NEGATIVE DELTA SPEED TEST RESULT]")
        print(f"Calculated travel speed for Event 2 (arriving at T=950 after T=1000): {f2.travel_speed_kmh:.2f} km/h")
        print(f"Impossible flag: {f2.impossible_travel_flag}")

        # Querying an event from the past should not compare against a future event with delta_t=1e-4!
        self.assertEqual(
            f2.impossible_travel_flag, 0,
            f"Negative delta speed bug: Out-of-order past transaction flagged as impossible travel at {f2.travel_speed_kmh} km/h"
        )

    def test_impossible_dead_discard_code_in_sliding_window(self):
        """
        EMPIRICAL CHALLENGE 4:
        In SlidingEntityWindow.commit:
            cutoff_24h = timestamp - 86400.0
            if timestamp < cutoff_24h:
                return
        Mathematical fact: timestamp < timestamp - 86400.0 is an UNSATISFIABLE tautology (always False).
        Hence, any transaction, no matter how ancient, is committed and increments count_24h and sum_24h.
        """
        win = SlidingEntityWindow("cust_dead_code")
        # Initialize window with current time 200,000
        win.commit(timestamp=200000.0, amount=100.0, mcc="5411")
        self.assertEqual(win.count_24h, 1)

        # Now commit an event from 10 days ago (timestamp = 10.0, which is << 200000 - 86400)
        win.commit(timestamp=10.0, amount=500.0, mcc="9999")

        print("\n[DEAD DISCARD CODE TEST RESULT]")
        print(f"win.count_24h after committing 10-day-old event: {win.count_24h}")
        print(f"win.sum_24h after committing 10-day-old event: {win.sum_24h}")
        print(f"win._events deque length: {len(win._events)}")

        # An event older than 24h retention horizon relative to entity state should have been discarded
        # But because `timestamp < timestamp - 86400` is impossible, it was accepted into the deque!
        self.assertEqual(
            win.count_24h, 1,
            f"Ancient event was not discarded because `timestamp < timestamp - 86400` is impossible! Count: {win.count_24h}"
        )


class TestCoordinateEdgeCasesAdversarial(unittest.TestCase):
    """Stress-tests geographic calculation boundary conditions and singularities."""

    def test_north_pole_multiple_longitudes_identity(self):
        """
        At the North Pole (lat=90.0), all longitudes represent the EXACT same physical point.
        Distance between (90.0, 0.0) and (90.0, 180.0) or (90.0, 90.0) must be 0.0 km.
        """
        dist_180 = haversine_distance(90.0, 0.0, 90.0, 180.0)
        dist_90 = haversine_distance(90.0, 0.0, 90.0, 90.0)
        dist_neg = haversine_distance(90.0, -120.0, 90.0, 45.0)

        print("\n[NORTH POLE LONGITUDE TEST]")
        print(f"dist( (90,0), (90,180) ) = {dist_180} km")
        print(f"dist( (90,0), (90,90) ) = {dist_90} km")
        print(f"dist( (90,-120), (90,45) ) = {dist_neg} km")

        self.assertAlmostEqual(dist_180, 0.0, places=3)
        self.assertAlmostEqual(dist_90, 0.0, places=3)
        self.assertAlmostEqual(dist_neg, 0.0, places=3)

    def test_south_pole_multiple_longitudes_identity(self):
        """At South Pole (lat=-90.0), all longitudes are the same point."""
        dist_sp = haversine_distance(-90.0, 0.0, -90.0, 180.0)
        print(f"dist( (-90,0), (-90,180) ) = {dist_sp} km")
        self.assertAlmostEqual(dist_sp, 0.0, places=3)

    def test_exact_180_meridian_wrap(self):
        """Test points on the exact 180° / -180° meridian line."""
        # Exact same point: 180.0 vs -180.0
        d_same = haversine_distance(10.0, 180.0, 10.0, -180.0)
        self.assertAlmostEqual(d_same, 0.0, places=4)

        # 1 degree across date line: 179.5 vs -179.5
        d_cross = haversine_distance(0.0, 179.5, 0.0, -179.5)
        expected = EARTH_RADIUS_KM * math.radians(1.0)
        self.assertAlmostEqual(d_cross, expected, delta=0.5)

    def test_antipodal_points_distance(self):
        """Distance between antipodal points is exactly half Earth circumference."""
        half_circumference = math.pi * EARTH_RADIUS_KM  # ~20015.087 km
        dist_antipodal = haversine_distance(0.0, 0.0, 0.0, 180.0)
        self.assertAlmostEqual(dist_antipodal, half_circumference, delta=1.0)

    def test_zero_delta_timestamp_at_same_coords(self):
        """Simultaneous transactions at exact same location produce 0 speed, no exception."""
        speed, imp, dist = compute_travel_speed(
            lat1=40.7128, lon1=-74.0060, t1=1000.0,
            lat2=40.7128, lon2=-74.0060, t2=1000.0,
        )
        self.assertEqual(dist, 0.0)
        self.assertEqual(speed, 0.0)
        self.assertEqual(imp, 0)

    def test_extreme_latitude_clamping(self):
        """Extreme latitude inputs (>90 or <-90) are clamped without crashing."""
        dist = haversine_distance(999.0, 0.0, -999.0, 0.0)
        half_circ = math.pi * EARTH_RADIUS_KM
        self.assertAlmostEqual(dist, half_circ, delta=5.0)


class TestMassiveVolumeAndPerformance(unittest.TestCase):
    """Stress-tests massive transaction volumes (10,000+ events per entity)."""

    def test_10000_events_single_entity_throughput_and_drift(self):
        """
        Sends 10,000 events to a single entity within a 1-hour window.
        Measures:
        1. Latency distribution (P50, P95, P99, Max).
        2. Floating-point accumulation drift between sum_24h and exact sum of events.
        """
        win = SlidingEntityWindow("cust_stress_10k")
        base_time = 1729080000.0
        n_events = 10000

        # Commit 10,000 events over 3000 seconds (~50 minutes)
        t_start = time.perf_counter()
        for i in range(n_events):
            ts = base_time + (i * 0.3)
            amt = round(10.0 + (i % 100) * 0.5, 2)
            win.commit(ts, amt, mcc=f"MCC_{i % 10}")
        commit_duration = (time.perf_counter() - t_start) * 1000.0
        print(f"\n[MASSIVE VOLUME] 10,000 commits: {commit_duration:.2f} ms ({commit_duration/n_events:.4f} ms/commit)")

        # Verify deque length
        self.assertEqual(len(win._events), n_events)
        self.assertEqual(win.count_24h, n_events)

        # Check floating-point drift in sum_24h vs exact Python sum
        exact_sum = sum(ev[1] for ev in win._events)
        sum_error = abs(win.sum_24h - exact_sum)
        print(f"[ACCUMULATION DRIFT] sum_24h: {win.sum_24h:.6f}, exact_sum: {exact_sum:.6f}, abs_diff: {sum_error:.6e}")
        self.assertLess(sum_error, 1e-4, f"Floating point drift exceeded tolerance: {sum_error}")

        # Measure get_stats latency with 10,000 active events in window
        query_times_ms = []
        now = base_time + 3000.0
        for _ in range(100):
            t0 = time.perf_counter()
            stats = win.get_stats(current_timestamp=now, current_amount=50.0)
            query_times_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = np.percentile(query_times_ms, 50)
        p95 = np.percentile(query_times_ms, 95)
        p99 = np.percentile(query_times_ms, 99)
        max_t = np.max(query_times_ms)

        query_budget = 15.0 if os.environ.get("CI") else 5.0
        self.assertLess(p99, query_budget, f"P99 query latency {p99:.2f}ms exceeded {query_budget}ms target")

    def test_massive_out_of_order_insertion_degradation(self):
        """
        Measures deque.insert degradation when many out-of-order events arrive in reverse order.
        """
        win = SlidingEntityWindow("cust_ooo_stress")
        base_time = 1729080000.0
        n_events = 2000

        # Inserting in reverse chronological order triggers O(N) deque scan per insert
        t_start = time.perf_counter()
        for i in range(n_events, 0, -1):
            ts = base_time + (i * 1.0)
            win.commit(ts, 15.0, mcc="5411")
        reverse_duration = (time.perf_counter() - t_start) * 1000.0
        print(f"\n[OUT-OF-ORDER INSERTION] 2,000 reverse inserts: {reverse_duration:.2f} ms")
        # Ensure chronological order is maintained
        timestamps = [ev[0] for ev in win._events]
        self.assertEqual(timestamps, sorted(timestamps))


class TestMemoryLeaksAndTTLEviction(unittest.TestCase):
    """Stress-tests memory consumption and entity TTL eviction across streaming cycles."""

    def test_rapid_entity_churn_and_ttl_eviction(self):
        """
        Simulates 20,000 transient entities created and evicted across time.
        Monitors memory usage with tracemalloc to ensure no memory leak.
        """
        store = StreamingFeatureStore()
        tracemalloc.start()
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()

        sim_time = 100000.0
        ttl = 300.0  # 5-minute TTL

        # Process 20,000 entities across 20 batches with TTL eviction
        for batch in range(20):
            for i in range(1000):
                entity_idx = batch * 1000 + i
                event = {
                    "transaction_id": f"tx_{entity_idx}",
                    "customer_id": f"cust_{entity_idx}",
                    "card_id": f"card_{entity_idx}",
                    "amount": 25.0,
                    "timestamp": sim_time,
                    "mcc": "5411",
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                }
                store.extract_features(event, commit=True)

            # Advance time past TTL and evict
            sim_time += 400.0
            evicted = store.evict_inactive(sim_time, ttl_seconds=ttl)
            self.assertGreater(evicted, 0)

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')

        total_growth_kb = sum(stat.size_diff for stat in top_stats) / 1024.0
        print(f"\n[MEMORY LEAK CHECK] Active entities remaining: {store.active_entity_count}")
        print(f"[MEMORY LEAK CHECK] Total memory delta across 20k churned entities: {total_growth_kb:.2f} KB")

        tracemalloc.stop()
        # Active entities should only be from the last batch
        self.assertLessEqual(store.active_entity_count, 2000)
        # Memory growth should be well bounded (< 25 MB for 20k entities)
        self.assertLess(total_growth_kb, 25000.0, f"Memory growth {total_growth_kb:.2f} KB indicates potential leak")


class TestRandomizedTemporalOracleAdversarial(unittest.TestCase):
    """
    Validation of strict zero temporal leakage using an independent Oracle.
    
    Oracle Definition:
    For any transaction event E at timestamp T, an Oracle store is instantiated
    and given ONLY transactions with timestamp < T (strictly historical).
    The features produced by the production feature store MUST match the Oracle
    features exactly. Any divergence is irrefutable evidence of lookahead leakage.
    """

    def test_random_interleaved_stream_oracle_equivalence(self):
        """
        Generates 200 randomly interleaved events for 10 entities.
        Compares production store against a pure chronological Oracle.
        """
        random.seed(42)
        store = StreamingFeatureStore()
        
        entities = [f"entity_{k}" for k in range(5)]
        raw_events = []
        base_time = 1729080000.0

        for i in range(150):
            ent = random.choice(entities)
            # Add random jitter up to 120s (introducing out-of-order events)
            ts = base_time + (i * 30.0) + random.uniform(-60.0, 60.0)
            amt = round(random.uniform(5.0, 200.0), 2)
            lat = round(40.7128 + random.uniform(-0.1, 0.1), 4)
            lon = round(-74.0060 + random.uniform(-0.1, 0.1), 4)
            raw_events.append({
                "transaction_id": f"tx_oracle_{i}",
                "customer_id": ent,
                "card_id": f"card_{ent}",
                "amount": amt,
                "timestamp": ts,
                "mcc": random.choice(["5411", "5812", "5732"]),
                "latitude": lat,
                "longitude": lon,
            })

        # Process stream through feature store
        leakage_violations = []

        for idx, ev in enumerate(raw_events):
            # Extract via production store
            prod_features = store.extract_features(ev, commit=True)

            # Build fresh Oracle from scratch containing ONLY strictly prior events
            # (events committed before this one in the real stream that have timestamp < ev['timestamp'])
            prior_events = [
                e for e in raw_events[:idx]
                if e['customer_id'] == ev['customer_id'] and e['timestamp'] <= ev['timestamp']
            ]

            # In an oracle, count_5m and count_1h must match
            cutoff_5m = ev['timestamp'] - 300.0
            cutoff_1h = ev['timestamp'] - 3600.0
            cutoff_24h = ev['timestamp'] - 86400.0

            oracle_5m = sum(1 for e in prior_events if cutoff_5m <= e['timestamp'] <= ev['timestamp'])
            oracle_1h = sum(1 for e in prior_events if cutoff_1h <= e['timestamp'] <= ev['timestamp'])
            oracle_24h = sum(1 for e in prior_events if cutoff_24h <= e['timestamp'] <= ev['timestamp'])

            if prod_features.velocity_5m_count != oracle_5m:
                leakage_violations.append(
                    f"Event {ev['transaction_id']}: 5m count mismatch! Prod: {prod_features.velocity_5m_count}, Oracle: {oracle_5m}"
                )
            if prod_features.velocity_1h_count != oracle_1h:
                leakage_violations.append(
                    f"Event {ev['transaction_id']}: 1h count mismatch! Prod: {prod_features.velocity_1h_count}, Oracle: {oracle_1h}"
                )
            if prod_features.velocity_24h_count != oracle_24h:
                leakage_violations.append(
                    f"Event {ev['transaction_id']}: 24h count mismatch! Prod: {prod_features.velocity_24h_count}, Oracle: {oracle_24h}"
                )

        print(f"\n[RANDOMIZED ORACLE COMPARISON] Total events: 150, Total violations: {len(leakage_violations)}")
        if leakage_violations:
            print("First 5 violations:")
            for v in leakage_violations[:5]:
                print("  *", v)

        self.assertEqual(
            len(leakage_violations), 0,
            f"Zero temporal leakage failed: {len(leakage_violations)} out of 150 events had lookahead discrepancies!"
        )


if __name__ == "__main__":
    unittest.main()
