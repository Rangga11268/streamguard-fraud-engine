"""Adversarial Challenge Harness for StreamGuard E2E Test Suite and Synthetic Stream Generator.

Authored by teamwork_preview_challenger_m1_2.
Runs empirical probes covering:
  Probe 1: Regime 1 baseline fraud rate strict preservation (<0.1%) across 100 seeds and large N.
  Probe 2: Regime 2 Black Friday covariate shift realism and absence of label leakage.
  Probe 3: Regime 3 concept drift attack realism and timestamp monotonicity (time-travel detection).
  Probe 4: DelayedFeedbackQueue temporal isolation, off-by-one second leakage, and concurrency/race conditions.
  Probe 5: Multi-tier test assertion rigidity and mock resistance analysis.
"""

from __future__ import annotations

import collections
import concurrent.futures
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import numpy as np

from app.synthetic.stream_generator import (
    ChargebackFeedback,
    CustomerProfile,
    DelayedFeedbackQueue,
    InvestigatorFeedback,
    StreamRegime,
    SyntheticStreamGenerator,
    TransactionEntityEvent,
)
from benchmarks.benchmark_engine import (
    BenchmarkSuite,
    SlidingEntityWindow,
    calculate_psi,
    evaluate_cost_matrix_policy,
    haversine_distance_km,
)


def probe_regime_1_baseline_fraud_rate() -> Dict[str, Any]:
    """Probe 1: Verify whether Regime 1 strictly preserves <0.1% fraud rate."""
    print("\n--- PROBE 1: Regime 1 Baseline Fraud Rate (<0.1%) ---")
    
    # 1.1 Large N test (N = 100,000)
    gen = SyntheticStreamGenerator(num_customers=500, random_seed=42)
    n_large = 100000
    events = list(gen.generate_regime_1_baseline(n_samples=n_large))
    frauds = sum(1 for e in events if e.true_label == 1)
    rate_large = frauds / n_large
    
    # 1.2 Multi-seed test (50 seeds of N=4,000 each, matching test_tier3)
    seed_rates = []
    exceeded_count_4k = 0
    for s in range(50):
        g = SyntheticStreamGenerator(num_customers=200, random_seed=1000 + s)
        evs = list(g.generate_regime_1_baseline(n_samples=4000))
        f_cnt = sum(1 for e in evs if e.true_label == 1)
        r = f_cnt / 4000
        seed_rates.append(r)
        if r >= 0.001:  # 0.1%
            exceeded_count_4k += 1

    # 1.3 Test tier 3 assertion check: test_tier3 line 48 asserts < 0.002 (0.2%)
    # Is the generator parameter 0.0008?
    # What fraction of runs exceed 0.1% when N=4000?
    results = {
        "large_n_samples": n_large,
        "large_n_frauds": frauds,
        "large_n_rate_pct": round(rate_large * 100, 4),
        "large_n_strictly_under_0_1_pct": rate_large < 0.0010,
        "multi_seed_runs": 50,
        "seed_rates_min_pct": round(min(seed_rates) * 100, 4),
        "seed_rates_max_pct": round(max(seed_rates) * 100, 4),
        "seed_rates_mean_pct": round(float(np.mean(seed_rates)) * 100, 4),
        "runs_exceeding_0_1_pct": exceeded_count_4k,
        "pct_runs_exceeding_0_1_pct": round((exceeded_count_4k / 50) * 100, 2),
    }
    print(f"Large N (100k) fraud rate: {results['large_n_rate_pct']}% (Strictly < 0.1%: {results['large_n_strictly_under_0_1_pct']})")
    print(f"N=4,000 across 50 seeds: Mean={results['seed_rates_mean_pct']}%, Min={results['seed_rates_min_pct']}%, Max={results['seed_rates_max_pct']}%")
    print(f"Runs with N=4,000 exceeding 0.1% ceiling: {exceeded_count_4k}/50 ({results['pct_runs_exceeding_0_1_pct']}%)")
    return results


def probe_regime_2_covariate_shift_and_label_leakage() -> Dict[str, Any]:
    """Probe 2: Verify Regime 2 4x holiday volume surge and ensure no fraud label leakage."""
    print("\n--- PROBE 2: Regime 2 Covariate Shift & Label Leakage ---")
    gen = SyntheticStreamGenerator(num_customers=500, random_seed=42)
    n_samples = 10000
    events = list(gen.generate_regime_2_covariate_surge(n_samples=n_samples, start_time=1729080000.0))
    
    frauds = [e for e in events if e.true_label == 1]
    legits = [e for e in events if e.true_label == 0]
    
    fraud_rate = len(frauds) / n_samples
    
    # Inter-arrival time (rate)
    dts = [events[i].timestamp - events[i - 1].timestamp for i in range(1, len(events))]
    mean_dt = float(np.mean(dts))
    rate_tx_per_sec = 1.0 / mean_dt
    
    # Check ticket distribution
    legit_amounts = [e.amount for e in legits]
    fraud_amounts = [e.amount for e in frauds]
    
    # Check label leakage: Are any features 1-to-1 deterministic indicators of fraud?
    # Check MCCs
    fraud_mccs = collections.Counter(e.merchant_category_code for e in frauds)
    legit_mccs = collections.Counter(e.merchant_category_code for e in legits)
    
    # Check if fraud MCCs exist in legit MCCs
    mcc_overlap = set(fraud_mccs.keys()).issubset(set(legit_mccs.keys()))
    
    # Check device_id, channel, ip_address format
    channel_overlap = set(e.channel for e in frauds).issubset(set(e.channel for e in legits))
    
    # Test if simple threshold on amount can perfectly separate fraud from legit
    # (Checking for artificial leakage)
    min_fraud_amt = min(fraud_amounts) if fraud_amounts else 0.0
    max_legit_amt = max(legit_amounts) if legit_amounts else 0.0
    amount_separable = min_fraud_amt > max_legit_amt

    results = {
        "n_samples": n_samples,
        "fraud_rate_pct": round(fraud_rate * 100, 4),
        "mean_dt_sec": round(mean_dt, 4),
        "volume_tx_per_sec": round(rate_tx_per_sec, 2),
        "is_4x_volume_surge": 0.65 <= rate_tx_per_sec <= 0.95,
        "legit_mean_amount": round(float(np.mean(legit_amounts)), 2),
        "fraud_mean_amount": round(float(np.mean(fraud_amounts)), 2) if fraud_amounts else 0.0,
        "mcc_overlap_no_exclusive_fraud_mcc": mcc_overlap,
        "channel_overlap": channel_overlap,
        "amount_trivially_separable": amount_separable,
    }
    print(f"Volume rate: {results['volume_tx_per_sec']} tx/s (Target ~0.8 tx/s, 4x of 0.2): {results['is_4x_volume_surge']}")
    print(f"Legit mean amt: ${results['legit_mean_amount']}, Fraud mean amt: ${results['fraud_mean_amount']}")
    print(f"MCC overlap (no exclusive leakage MCC): {mcc_overlap}")
    print(f"Trivially separable amounts: {amount_separable} (False is good = authentic overlap)")
    return results


def probe_regime_3_concept_drift_and_timestamp_ordering() -> Dict[str, Any]:
    """Probe 3: Verify concept drift attack patterns and test timestamp monotonicity."""
    print("\n--- PROBE 3: Regime 3 Concept Drift & Timestamp Monotonicity ---")
    gen = SyntheticStreamGenerator(num_customers=500, random_seed=42)
    n_samples = 3000
    events = list(gen.generate_regime_3_concept_drift(n_samples=n_samples, start_time=1729080000.0))
    
    frauds = [e for e in events if e.true_label == 1]
    attack_rate = len(frauds) / len(events)
    
    # Check attack components
    micro_charges = [f for f in frauds if f.amount < 2.00]
    drain_charges = [f for f in frauds if f.amount > 800.0]
    
    # Check geo jumps
    # Scan customer events for consecutive speed > 1500 km/h
    cust_events: Dict[str, List[TransactionEntityEvent]] = collections.defaultdict(list)
    for e in events:
        cust_events[e.customer_id].append(e)
    
    geo_jumps = 0
    for cid, ev_list in cust_events.items():
        for i in range(1, len(ev_list)):
            e1, e2 = ev_list[i - 1], ev_list[i]
            dt = e2.timestamp - e1.timestamp
            if dt > 0:
                dist = haversine_distance_km(e1.latitude, e1.longitude, e2.latitude, e2.longitude)
                speed = dist / (dt / 3600.0)
                if speed > 1500.0:
                    geo_jumps += 1

    # CRITICAL AUDIT: Timestamp Monotonicity (Time-Travel Bug Detection)
    backwards_jumps = []
    for i in range(1, len(events)):
        t_prev = events[i - 1].timestamp
        t_curr = events[i].timestamp
        if t_curr < t_prev:
            backwards_jumps.append({
                "index": i,
                "prev_tx": events[i - 1].transaction_id,
                "prev_time": t_prev,
                "curr_tx": events[i].transaction_id,
                "curr_time": t_curr,
                "delta_seconds": round(t_curr - t_prev, 4),
            })

    results = {
        "n_samples": len(events),
        "attack_rate_pct": round(attack_rate * 100, 2),
        "micro_charges_count": len(micro_charges),
        "drain_charges_count": len(drain_charges),
        "impossible_geo_jumps_detected": geo_jumps,
        "backwards_timestamp_jumps_count": len(backwards_jumps),
        "worst_backwards_jump_sec": min((j["delta_seconds"] for j in backwards_jumps), default=0.0),
        "sample_backwards_jumps": backwards_jumps[:3],
    }
    print(f"Attack rate: {results['attack_rate_pct']}% (Target 3.5%-5.0%)")
    print(f"Micro-charges: {len(micro_charges)}, Drain charges: {len(drain_charges)}, Geo jumps: {geo_jumps}")
    print(f"BACKWARDS TIMESTAMP JUMPS: {len(backwards_jumps)} detected!")
    if backwards_jumps:
        print(f"Worst backwards jump: {results['worst_backwards_jump_sec']} seconds")
        for j in backwards_jumps[:3]:
            print(f"  Jump at idx {j['index']}: prev t={j['prev_time']} ({j['prev_tx']}) -> curr t={j['curr_time']} ({j['curr_tx']}), delta={j['delta_seconds']}s")
    return results


def probe_delayed_feedback_queue() -> Dict[str, Any]:
    """Probe 4: Test DelayedFeedbackQueue temporal integrity, off-by-one, and concurrency."""
    print("\n--- PROBE 4: DelayedFeedbackQueue Temporal Isolation & Concurrency ---")
    queue = DelayedFeedbackQueue()
    t0 = 1729080000.0
    
    # 4.1 Sub-second and boundary off-by-one test
    delivery_time_1 = queue.schedule_investigator_feedback("tx_exact_1", t0, True, delay_seconds=3600.0)
    delivery_time_2 = queue.schedule_chargeback_feedback("tx_exact_2", t0, 500.0, True, delay_seconds=14.0 * 86400.0)
    
    # Poll at delivery_time - 1e-6 (1 microsecond before maturity)
    inv_before, cb_before = queue.poll(delivery_time_1 - 1e-6)
    leak_1us = len(inv_before) > 0 or len(cb_before) > 0
    
    # Poll at delivery_time - 1.0 (1 second before maturity)
    inv_1s, cb_1s = queue.poll(delivery_time_2 - 1.0)
    leak_1s = len(inv_1s) > 0 or len(cb_1s) > 0
    
    # Poll at exact delivery_time_1
    inv_exact, cb_exact = queue.poll(delivery_time_1)
    delivered_on_time = len(inv_exact) == 1 and inv_exact[0].transaction_id == "tx_exact_1"
    
    # Chargeback must still be pending
    self_pending_cb = queue.pending_count() == 1
    
    # Poll chargeback at exact delivery_time_2
    inv_cb_time, cb_cb_time = queue.poll(delivery_time_2)
    delivered_cb_on_time = len(cb_cb_time) == 1 and cb_cb_time[0].transaction_id == "tx_exact_2"

    # 4.2 Non-monotonic poll test: what happens if poll is called with decreasing time?
    # Queue is empty now, schedule 3 events at t0+10, t0+20, t0+30
    queue.clear()
    queue.schedule_investigator_feedback("tx_10", t0, True, delay_seconds=10.0)
    queue.schedule_investigator_feedback("tx_20", t0, True, delay_seconds=20.0)
    queue.schedule_investigator_feedback("tx_30", t0, True, delay_seconds=30.0)
    
    # Caller poll at t0+25 (delivers tx_10 and tx_20)
    i_25, _ = queue.poll(t0 + 25.0)
    # Caller inadvertently calls poll at t0+15 (due to backwards timestamp stream)
    i_15, _ = queue.poll(t0 + 15.0)
    
    # 4.3 Concurrent thread-safety stress test
    # 10 threads enqueuing 100 events each while 2 threads poll continuously
    queue.clear()
    concurrent_errors = []
    enqueued_total = 1000
    
    def worker_enqueue(worker_id: int):
        try:
            for k in range(100):
                queue.schedule_investigator_feedback(
                    f"tx_thread_{worker_id}_{k}",
                    t0 + k * 10,
                    is_fraud=True,
                    delay_seconds=random.uniform(50.0, 500.0),
                )
        except Exception as e:
            concurrent_errors.append(f"Enqueue error: {e}")

    def worker_poll():
        try:
            for _ in range(50):
                time.sleep(0.001)
                _ = queue.poll(t0 + 300.0)
        except Exception as e:
            concurrent_errors.append(f"Poll error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        f_polls = [executor.submit(worker_poll) for _ in range(2)]
        f_enqs = [executor.submit(worker_enqueue, i) for i in range(10)]
        concurrent.futures.wait(f_enqs + f_polls)

    # Check queue integrity after concurrent storm
    stats = queue.stats()
    queue_corrupted = len(concurrent_errors) > 0 or (stats["enqueued"] != enqueued_total)

    results = {
        "leak_1us_before_maturity": leak_1us,
        "leak_1s_before_maturity": leak_1s,
        "delivered_on_exact_delivery_time": delivered_on_time,
        "cb_pending_until_full_horizon": self_pending_cb,
        "cb_delivered_on_exact_time": delivered_cb_on_time,
        "poll_decreasing_time_behavior": f"Delivered 25s: {len(i_25)}, Delivered 15s after: {len(i_15)}",
        "concurrency_errors": concurrent_errors,
        "concurrency_enqueued_expected": enqueued_total,
        "concurrency_enqueued_actual": stats["enqueued"],
        "is_thread_safe": not queue_corrupted,
    }
    print(f"Leakage 1 microsecond before maturity: {leak_1us} (False is correct)")
    print(f"Leakage 1 second before maturity: {leak_1s} (False is correct)")
    print(f"Delivered exactly on maturation timestamp: {delivered_on_time}")
    print(f"Chargeback delivered at exact 14-30d maturity: {delivered_cb_on_time}")
    print(f"Thread safety without external lock: {results['is_thread_safe']} (Errors: {len(concurrent_errors)})")
    return results


def probe_multi_tier_test_assertions() -> Dict[str, Any]:
    """Probe 5: Audit test assertions in test_tier1 through test_tier4 for trivial bypasses."""
    print("\n--- PROBE 5: Test Tier Assertions Rigidity & Mock Bypass Audit ---")
    
    # Audit 5.1: PR-AUC uplift test in test_tier3_statistical.py
    # In test_pr_auc_uplift_under_drift:
    # Does it test any model code from app.models or synthetic stream?
    # We examine if the test can be satisfied without any real model:
    # Notice: static_scores is beta(1.2, 15.0), adaptive_scores is beta(5.5, 2.5) if true_label==1 else beta(0.8, 22.0)
    is_pr_auc_a_pure_oracle_mock = True
    
    # Audit 5.2: Cost matrix test in test_tier3 line 153-173
    # true_labels, p_legit, p_fraud, probs = np.where(true_labels == 1, p_fraud, p_legit)
    # Again pure synthetic beta distribution, no model or feature store involved.
    is_cost_matrix_synthetic_simulation = True
    
    # Audit 5.3: Regime 1 fraud rate threshold in test_tier3:
    # Requirement: "strictly preserves <0.1% fraud rate"
    # Line 48: self.assertLess(fraud_rate, 0.002) -> allows up to 0.2% (2x requirement)!
    test_tier3_regime1_assertion_loose = True
    
    # Audit 5.4: Latency benchmark in test_tier4 / benchmark_engine.py:
    # Single iteration without multiple trials causes intermittent failure on 1,000-entity extraction.
    suite = BenchmarkSuite()
    trials_1000_entity = []
    for _ in range(10):
        res = suite.run_entity_extraction_benchmark(n_entities=1000, n_warmup_events=8)
        trials_1000_entity.append(res.measured_value)
    
    failures_out_of_10 = sum(1 for val in trials_1000_entity if val >= 5.0)

    results = {
        "pr_auc_test_is_oracle_mock": is_pr_auc_a_pure_oracle_mock,
        "cost_matrix_test_is_synthetic_simulation": is_cost_matrix_synthetic_simulation,
        "tier3_regime1_assertion_allows_2x_ceiling": test_tier3_regime1_assertion_loose,
        "tier4_1000_entity_trials_ms": trials_1000_entity,
        "tier4_1000_entity_failures_out_of_10": failures_out_of_10,
        "tier4_flakiness_pct": failures_out_of_10 * 10.0,
    }
    print(f"PR-AUC test uses label oracle directly (no model/features): {is_pr_auc_a_pure_oracle_mock}")
    print(f"Cost matrix test uses synthetic beta generator (no model): {is_cost_matrix_synthetic_simulation}")
    print(f"Tier 3 Regime 1 test asserts <0.002 (0.2%), double the <0.1% spec: {test_tier3_regime1_assertion_loose}")
    print(f"Tier 4 1,000-entity extraction 10-trial timings: {trials_1000_entity}")
    print(f"Tier 4 flakiness rate (>5.0ms failure rate): {results['tier4_flakiness_pct']}%")
    return results


def main() -> None:
    print("=" * 80)
    print("STREAMGUARD ADVERSARIAL CHALLENGE HARNESS (teamwork_preview_challenger_m1_2)")
    print("=" * 80)
    
    r1 = probe_regime_1_baseline_fraud_rate()
    r2 = probe_regime_2_covariate_shift_and_label_leakage()
    r3 = probe_regime_3_concept_drift_and_timestamp_ordering()
    r4 = probe_delayed_feedback_queue()
    r5 = probe_multi_tier_test_assertions()
    
    print("\n" + "=" * 80)
    print("ALL PROBES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
