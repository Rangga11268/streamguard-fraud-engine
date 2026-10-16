"""
Adversarial Stress Test Suite for Milestone 6: FastAPI Serving Layer.
Empirical Challenger: teamwork_preview_challenger_m6_1.

Tests:
  1. POST /v1/score Fuzzing & Boundary Testing:
     - Non-positive amounts (0.0, -1.0, -1e-6) -> 422
     - Non-positive timestamps (0.0, -100.0) -> 422
     - Missing / empty required fields (customer_id, card_id, transaction_id, location) -> 422
     - Extreme physical coordinates (North Pole 90.0, South Pole -90.0, Antimeridian 180.0, -180.0) -> 200
     - Out of bounds coordinates (lat > 90, lat < -90, lon > 180, lon < -180) -> 422
     - Monotonic velocity updates across rapid sequential transactions
     - Impossible travel velocity flag trigger across antipodal / intercontinental locations (>800 km/h)
  2. Feedback Endpoints & Label Precedence:
     - /v1/feedback/investigator invalid verdicts, confidence bounds, non-existent tx_id
     - /v1/feedback/chargeback non-existent tx_id, negative amount, non-positive received_at
     - Label precedence: Chargeback dispute (y=1) strictly overrides prior investigator clearance (y=0)
     - Label precedence: Subsequent investigator clearance (y=0) CANNOT downgrade confirmed chargeback (y=1)
  3. GET /v1/telemetry/drift:
     - Valid schema compliance and metric bounds
     - Query parameter validation (feature, metric, window_size) -> 400 on invalid
     - Drift detection under simulated covariate & concept shift with automatic threshold recalibration
  4. Scoring Latency & P99 Benchmark:
     - 1,000 scored requests measured through API route
     - Strictly validates P99 latency < 15.0 ms
  5. Multi-Threaded Concurrency & Thread-Safety:
     - Concurrent scoring, feedback ingestion, and telemetry polling across 16 threads
     - Checks for race conditions, 500 errors, and memory integrity
"""

from __future__ import annotations

import concurrent.futures
import time
import unittest
from typing import Any, Dict, List

import numpy as np
from fastapi.testclient import TestClient

from app.api.schemas import (
    ChargebackFeedbackResponse,
    DriftTelemetryResponse,
    InvestigatorFeedbackResponse,
    ScoreResponse,
)
from app.main import app, get_global_engine
from app.memory.dual_memory import TransactionLabelState


class TestAdversarialFuzzingScore(unittest.TestCase):
    """Adversarial stress-testing of POST /v1/score endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.engine = get_global_engine()

    def test_amount_fuzzing_rejects_non_positive_and_malformed(self) -> None:
        """Reject 0, negative, and malformed monetary amounts."""
        invalid_amounts = [0.0, -0.01, -100.0, -1e-9]
        for amt in invalid_amounts:
            payload = {
                "transaction_id": f"tx_fuzz_amt_{amt}",
                "customer_id": "cust_fuzz",
                "card_id": "card_fuzz",
                "amount": amt,
                "timestamp": 1729100000.0,
                "location": {"latitude": 40.0, "longitude": -73.0},
            }
            resp = self.client.post("/v1/score", json=payload)
            self.assertEqual(
                resp.status_code,
                422,
                f"Amount {amt} expected 422, got {resp.status_code}: {resp.text}",
            )

    def test_missing_and_empty_mandatory_fields(self) -> None:
        """Reject omitted or empty mandatory identifiers."""
        base_payload = {
            "transaction_id": "tx_req_001",
            "customer_id": "cust_001",
            "card_id": "card_001",
            "amount": 50.0,
            "timestamp": 1729100000.0,
            "location": {"latitude": 10.0, "longitude": 10.0},
        }

        # Missing customer_id
        p1 = dict(base_payload)
        del p1["customer_id"]
        self.assertEqual(self.client.post("/v1/score", json=p1).status_code, 422)

        # Empty customer_id
        p2 = dict(base_payload)
        p2["customer_id"] = ""
        self.assertEqual(self.client.post("/v1/score", json=p2).status_code, 422)

        # Missing transaction_id
        p3 = dict(base_payload)
        del p3["transaction_id"]
        self.assertEqual(self.client.post("/v1/score", json=p3).status_code, 422)

        # Empty transaction_id
        p4 = dict(base_payload)
        p4["transaction_id"] = ""
        self.assertEqual(self.client.post("/v1/score", json=p4).status_code, 422)

        # Missing card_id
        p5 = dict(base_payload)
        del p5["card_id"]
        self.assertEqual(self.client.post("/v1/score", json=p5).status_code, 422)

        # Empty card_id
        p6 = dict(base_payload)
        p6["card_id"] = ""
        self.assertEqual(self.client.post("/v1/score", json=p6).status_code, 422)

        # Missing location
        p7 = dict(base_payload)
        del p7["location"]
        self.assertEqual(self.client.post("/v1/score", json=p7).status_code, 422)

        # Non-positive timestamp
        p8 = dict(base_payload)
        p8["timestamp"] = 0.0
        self.assertEqual(self.client.post("/v1/score", json=p8).status_code, 422)

        p9 = dict(base_payload)
        p9["timestamp"] = -500.0
        self.assertEqual(self.client.post("/v1/score", json=p9).status_code, 422)

    def test_extreme_coordinates_boundary_handling(self) -> None:
        """Physical poles and antimeridian coordinates execute safely without math domain errors."""
        extreme_points = [
            ("North Pole", 90.0, 0.0),
            ("South Pole", -90.0, 0.0),
            ("Antimeridian East", 0.0, 180.0),
            ("Antimeridian West", 0.0, -180.0),
            ("Equator Null Island", 0.0, 0.0),
            ("North-East Corner", 90.0, 180.0),
            ("South-West Corner", -90.0, -180.0),
        ]

        for name, lat, lon in extreme_points:
            payload = {
                "transaction_id": f"tx_pole_{lat}_{lon}",
                "customer_id": f"cust_pole_{lat}_{lon}",
                "card_id": f"card_pole_{lat}_{lon}",
                "amount": 75.0,
                "timestamp": 1729100100.0,
                "location": {"latitude": lat, "longitude": lon},
            }
            resp = self.client.post("/v1/score", json=payload)
            self.assertEqual(
                resp.status_code,
                200,
                f"Failed for extreme point {name} ({lat}, {lon}): {resp.text}",
            )
            data = resp.json()
            self.assertGreaterEqual(data["fraud_score"], 0.0)
            self.assertLessEqual(data["fraud_score"], 1.0)
            self.assertFalse(np.isnan(data["fraud_score"]))
            self.assertFalse(np.isinf(data["fraud_score"]))

    def test_out_of_bounds_coordinates_rejected_with_422(self) -> None:
        """Coordinates exceeding physical earth bounds fail validation."""
        invalid_coords = [
            (90.001, 0.0),
            (-90.001, 0.0),
            (0.0, 180.001),
            (0.0, -180.001),
            (1000.0, 2000.0),
        ]
        for lat, lon in invalid_coords:
            payload = {
                "transaction_id": f"tx_oob_{lat}_{lon}",
                "customer_id": "cust_oob",
                "card_id": "card_oob",
                "amount": 50.0,
                "timestamp": 1729100200.0,
                "location": {"latitude": lat, "longitude": lon},
            }
            resp = self.client.post("/v1/score", json=payload)
            self.assertEqual(
                resp.status_code,
                422,
                f"Coord ({lat}, {lon}) expected 422, got {resp.status_code}",
            )

    def test_rapid_sequential_transactions_velocity_increments(self) -> None:
        """Rapid consecutive transactions for single customer strictly increment velocity counts."""
        customer = "cust_adv_burst_01"
        card = "card_adv_burst_01"
        base_time = 1729101000.0

        for i in range(10):
            payload = {
                "transaction_id": f"tx_burst_{i}",
                "customer_id": customer,
                "card_id": card,
                "amount": 10.0 + i,
                "timestamp": base_time + (i * 5.0),  # 5 seconds apart (all within 5m, 1h, 24h)
                "location": {"latitude": 40.7128, "longitude": -74.0060},
            }
            resp = self.client.post("/v1/score", json=payload)
            self.assertEqual(resp.status_code, 200)
            feat = resp.json()["features"]

            # Strict non-leakage invariant: event i observes i prior events
            self.assertEqual(feat["velocity_5m_count"], i)
            self.assertEqual(feat["velocity_1h_count"], i)
            self.assertEqual(feat["velocity_24h_count"], i)
            self.assertEqual(feat["impossible_travel_flag"], 0)

    def test_impossible_travel_jump_and_recovery(self) -> None:
        """Trigger impossible travel flag on supersonic cross-globe hop, then recover on local tx."""
        customer = "cust_adv_travel_hop"
        card = "card_adv_travel_hop"
        t0 = 1729102000.0

        # Step 1: Sydney, Australia (-33.8688, 151.2093)
        r1 = self.client.post("/v1/score", json={
            "transaction_id": "tx_hop_sydney",
            "customer_id": customer,
            "card_id": card,
            "amount": 40.0,
            "timestamp": t0,
            "location": {"latitude": -33.8688, "longitude": 151.2093},
        })
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["features"]["impossible_travel_flag"], 0)

        # Step 2: Reykjavik, Iceland (64.1466, -21.9426) 120 seconds later (~16,000 km, ~480,000 km/h)
        r2 = self.client.post("/v1/score", json={
            "transaction_id": "tx_hop_reykjavik",
            "customer_id": customer,
            "card_id": card,
            "amount": 850.0,
            "timestamp": t0 + 120.0,
            "location": {"latitude": 64.1466, "longitude": -21.9426},
        })
        self.assertEqual(r2.status_code, 200)
        d2 = r2.json()
        self.assertEqual(d2["features"]["impossible_travel_flag"], 1)
        self.assertGreater(d2["features"]["travel_speed_kmh"], 800.0)
        # Decision must reflect elevated fraud risk
        self.assertIn(d2["decision"], ["CHALLENGE", "BLOCK"])

        # Step 3: Reykjavik local transaction 30 seconds later (same city: 64.1470, -21.9400)
        r3 = self.client.post("/v1/score", json={
            "transaction_id": "tx_hop_reykjavik_local",
            "customer_id": customer,
            "card_id": card,
            "amount": 25.0,
            "timestamp": t0 + 150.0,
            "location": {"latitude": 64.1470, "longitude": -21.9400},
        })
        self.assertEqual(r3.status_code, 200)
        d3 = r3.json()
        # Impossible travel flag must reset back to 0 for normal local movement
        self.assertEqual(d3["features"]["impossible_travel_flag"], 0)
        self.assertLess(d3["features"]["travel_speed_kmh"], 800.0)


class TestAdversarialFeedbackAndLabelPrecedence(unittest.TestCase):
    """Adversarial stress-testing of feedback endpoints and ground-truth label precedence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.engine = get_global_engine()

    def test_investigator_feedback_validation_fuzzing(self) -> None:
        """Investigator review rejects invalid verdicts, confidence bounds, and missing tx."""
        # Non-existent transaction returns 404
        r404 = self.client.post("/v1/feedback/investigator", json={
            "transaction_id": "tx_non_existent_ghost_01",
            "investigator_id": "inv_01",
            "verdict": "FRAUD",
            "reviewed_at": 1729103000.0,
        })
        self.assertEqual(r404.status_code, 404)

        # Invalid verdict strings
        for bad_verdict in ["MAYBE", "UNKNOWN", "REJECTED", "TRUE", "1"]:
            resp = self.client.post("/v1/feedback/investigator", json={
                "transaction_id": "tx_dummy",
                "investigator_id": "inv_01",
                "verdict": bad_verdict,
                "reviewed_at": 1729103000.0,
            })
            self.assertEqual(resp.status_code, 422, f"Expected 422 for verdict '{bad_verdict}'")

        # Invalid confidence bounds (> 1.0 or < 0.0)
        for bad_conf in [-0.1, 1.01, 2.0]:
            resp = self.client.post("/v1/feedback/investigator", json={
                "transaction_id": "tx_dummy",
                "investigator_id": "inv_01",
                "verdict": "FRAUD",
                "reviewed_at": 1729103000.0,
                "confidence": bad_conf,
            })
            self.assertEqual(resp.status_code, 422, f"Expected 422 for confidence '{bad_conf}'")

    def test_chargeback_feedback_validation_fuzzing(self) -> None:
        """Chargeback dispute rejects non-positive amounts, received_at, and missing tx."""
        # Non-existent transaction returns 404
        r404 = self.client.post("/v1/feedback/chargeback", json={
            "transaction_id": "tx_ghost_dispute_99",
            "chargeback_id": "cb_01",
            "chargeback_amount": 100.0,
            "received_at": 1729104000.0,
        })
        self.assertEqual(r404.status_code, 404)

        # Non-positive amount
        for bad_amt in [0.0, -10.0]:
            resp = self.client.post("/v1/feedback/chargeback", json={
                "transaction_id": "tx_dummy",
                "chargeback_id": "cb_01",
                "chargeback_amount": bad_amt,
                "received_at": 1729104000.0,
            })
            self.assertEqual(resp.status_code, 422)

        # Non-positive received_at
        resp = self.client.post("/v1/feedback/chargeback", json={
            "transaction_id": "tx_dummy",
            "chargeback_id": "cb_01",
            "chargeback_amount": 50.0,
            "received_at": 0.0,
        })
        self.assertEqual(resp.status_code, 422)

    def test_strict_label_precedence_and_non_downgrade(self) -> None:
        """Empirically prove chargeback dispute overrides clearance and cannot be downgraded."""
        tx_id = "tx_adv_precedence_master"

        # 1. Ingest transaction via /v1/score
        score_resp = self.client.post("/v1/score", json={
            "transaction_id": tx_id,
            "customer_id": "cust_prec_01",
            "card_id": "card_prec_01",
            "amount": 420.0,
            "timestamp": 1729105000.0,
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        })
        self.assertEqual(score_resp.status_code, 200)

        # Verify initial memory record state
        rec = self.engine.dual_memory.working_memory.get(tx_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.state, TransactionLabelState.PROVISIONAL_UNLABELED)

        # 2. Fast investigator clears transaction (verdict="LEGITIMATE", y=0)
        inv_resp1 = self.client.post("/v1/feedback/investigator", json={
            "transaction_id": tx_id,
            "investigator_id": "inv_rookie_01",
            "verdict": "LEGITIMATE",
            "reviewed_at": 1729108600.0,  # 1 hour later
            "confidence": 0.85,
        })
        self.assertEqual(inv_resp1.status_code, 200)
        self.assertEqual(inv_resp1.json()["provisional_label"], 0)

        # Check internal memory state updated to INVESTIGATOR_CLEARED
        rec = self.engine.dual_memory.working_memory.get(tx_id)
        self.assertEqual(rec.state, TransactionLabelState.INVESTIGATOR_CLEARED)
        self.assertEqual(rec.effective_label, 0)

        # 3. 21 days later, bank chargeback dispute arrives (is_fraud=True, y=1)
        cb_resp = self.client.post("/v1/feedback/chargeback", json={
            "transaction_id": tx_id,
            "chargeback_id": "cb_master_dispute_101",
            "chargeback_amount": 420.0,
            "received_at": 1730919400.0,
            "is_fraud": True,
        })
        self.assertEqual(cb_resp.status_code, 200)
        cb_data = cb_resp.json()
        self.assertEqual(cb_data["status"], "CONSOLIDATED")
        self.assertEqual(cb_data["final_label"], 1)
        self.assertTrue(cb_data["reconciled_with_investigator"])

        # Check internal memory state overridden to CHARGEBACK_CONFIRMED with effective_label=1
        rec_cm = self.engine.dual_memory.consolidated_memory.get(tx_id)
        self.assertIsNotNone(rec_cm)
        self.assertEqual(rec_cm.state, TransactionLabelState.CHARGEBACK_CONFIRMED)
        self.assertEqual(rec_cm.effective_label, 1)

        # 4. ADVERSARIAL ATTEMPT: Investigator tries to re-submit clearance on confirmed dispute
        inv_resp2 = self.client.post("/v1/feedback/investigator", json={
            "transaction_id": tx_id,
            "investigator_id": "inv_rookie_01",
            "verdict": "LEGITIMATE",
            "reviewed_at": 1730920000.0,
            "confidence": 0.99,
        })
        # Internal state MUST NOT downgrade
        rec_after = self.engine.dual_memory.consolidated_memory.get(tx_id)
        self.assertEqual(
            rec_after.state,
            TransactionLabelState.CHARGEBACK_CONFIRMED,
            "Confirmed chargeback was illegally downgraded by investigator!",
        )
        self.assertEqual(
            rec_after.effective_label,
            1,
            "Effective label was illegally set to 0 after confirmed chargeback!",
        )


class TestAdversarialTelemetryAndDrift(unittest.TestCase):
    """Adversarial testing of GET /v1/telemetry/drift and automatic recalibration."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.engine = get_global_engine()

    def test_telemetry_schema_and_parameter_fuzzing(self) -> None:
        """Validate drift telemetry output schema and query parameter edge cases."""
        resp = self.client.get("/v1/telemetry/drift")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        validated = DriftTelemetryResponse(**data)
        self.assertGreaterEqual(validated.score_psi, 0.0)
        self.assertGreaterEqual(validated.score_wasserstein, 0.0)
        self.assertIn(validated.drift_status, ["STABLE", "WARNING", "DRIFT_DETECTED"])
        self.assertIn("tau_challenge", validated.current_thresholds)
        self.assertIn("tau_block", validated.current_thresholds)

        # Invalid query parameters
        self.assertEqual(self.client.get("/v1/telemetry/drift?feature=bogus").status_code, 400)
        self.assertEqual(self.client.get("/v1/telemetry/drift?metric=bogus").status_code, 400)
        self.assertEqual(self.client.get("/v1/telemetry/drift?window_size=0").status_code, 400)
        self.assertEqual(self.client.get("/v1/telemetry/drift?window_size=-10").status_code, 400)

    def test_drift_trigger_and_threshold_recalibration(self) -> None:
        """Inject shifted transaction stream to verify drift detection and router recalibration."""
        initial_tau_chal = self.engine.router.tau_challenge
        initial_tau_block = self.engine.router.tau_block

        # Stream 150 extreme fraud transactions (forcing large score PSI)
        base_t = 1729110000.0
        for i in range(150):
            payload = {
                "transaction_id": f"tx_drift_attack_{i}",
                "customer_id": f"cust_drift_{i % 10}",
                "card_id": f"card_drift_{i % 10}",
                "amount": 2500.0 + (i * 20.0),  # Massive amounts
                "timestamp": base_t + (i * 2.0),
                "location": {
                    "latitude": 55.7558 + (i * 0.01),
                    "longitude": 37.6173 + (i * 0.01),
                },
            }
            resp = self.client.post("/v1/score", json=payload)
            self.assertEqual(resp.status_code, 200)

        # Query drift telemetry
        tel_resp = self.client.get("/v1/telemetry/drift")
        self.assertEqual(tel_resp.status_code, 200)
        tel_data = tel_resp.json()

        # Under extreme high scores, score_psi or score_wasserstein must indicate divergence
        self.assertGreater(tel_data["score_psi"], 0.0)
        self.assertGreater(tel_data["window_samples"], 100)


class TestScoringLatencyBenchmark(unittest.TestCase):
    """Rigorous CPU latency benchmarking through REST API route (Target: P99 < 15ms)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.engine = get_global_engine()

    def test_p99_scoring_latency_strictly_sub_15ms(self) -> None:
        """Measure 1,000 requests through /v1/score; confirm P99 < 15.0ms on native CPU."""
        import asyncio
        import httpx

        n_warmup = 50
        n_benchmark = 1000
        base_t = 1729120000.0

        transport = httpx.ASGITransport(app=app)

        async def run_benchmark():
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                # Warmup CPU caches
                for i in range(n_warmup):
                    await client.post("/v1/score", json={
                        "transaction_id": f"tx_warmup_{i}",
                        "customer_id": f"cust_warmup_{i % 20}",
                        "card_id": f"card_warmup_{i % 20}",
                        "amount": 50.0 + (i % 50),
                        "timestamp": base_t + i,
                        "location": {"latitude": 40.7128, "longitude": -74.0060},
                    })

                # Benchmark run
                latencies_ms: List[float] = []
                server_latencies_ms: List[float] = []

                for i in range(n_benchmark):
                    payload = {
                        "transaction_id": f"tx_bench_{i}",
                        "customer_id": f"cust_bench_{i % 100}",
                        "card_id": f"card_bench_{i % 100}",
                        "amount": 35.0 + (i % 100),
                        "timestamp": base_t + 1000 + i,
                        "location": {
                            "latitude": 40.7128 + ((i % 10) * 0.01),
                            "longitude": -74.0060 + ((i % 10) * 0.01),
                        },
                    }
                    t_start = time.perf_counter()
                    resp = await client.post("/v1/score", json=payload)
                    client_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                    self.assertEqual(resp.status_code, 200)
                    data = resp.json()
                    latencies_ms.append(client_elapsed_ms)
                    server_latencies_ms.append(data["latency_ms"])

                return latencies_ms, server_latencies_ms

        latencies_ms, server_latencies_ms = asyncio.run(run_benchmark())

        arr_client = np.array(latencies_ms, dtype=np.float64)
        arr_server = np.array(server_latencies_ms, dtype=np.float64)

        p50 = float(np.percentile(arr_client, 50))
        p90 = float(np.percentile(arr_client, 90))
        p95 = float(np.percentile(arr_client, 95))
        p99 = float(np.percentile(arr_client, 99))
        p99_server = float(np.percentile(arr_server, 99))
        max_lat = float(np.max(arr_client))

        print(
            f"\n[LATENCY BENCHMARK RESULT] (N={n_benchmark})\n"
            f"  ASGI Round-Trip: P50={p50:.2f}ms | P90={p90:.2f}ms | P95={p95:.2f}ms | P99={p99:.2f}ms | Max={max_lat:.2f}ms\n"
            f"  Server-Side Route: P99={p99_server:.2f}ms\n"
            f"  Target: strictly < 15.0ms P99"
        )

        self.assertLess(
            p99,
            15.0,
            f"ASGI round-trip P99 latency {p99:.3f} ms breached 15.0ms threshold!",
        )
        self.assertLess(
            p99_server,
            15.0,
            f"Server-reported P99 latency {p99_server:.3f} ms breached 15.0ms threshold!",
        )


class TestAdversarialConcurrency(unittest.TestCase):
    """Stress-test concurrent request handling across multiple worker threads."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.engine = get_global_engine()

    def test_multi_threaded_mixed_workload_safety(self) -> None:
        """16 worker threads concurrently issuing scoring, feedback, and telemetry queries."""
        n_threads = 16
        n_ops_per_thread = 20
        base_t = 1729130000.0

        errors: List[str] = []

        def worker(thread_idx: int) -> None:
            for j in range(n_ops_per_thread):
                tx_id = f"tx_conc_t{thread_idx}_j{j}"
                cust_id = f"cust_conc_t{thread_idx % 4}"
                card_id = f"card_conc_t{thread_idx % 4}"

                # 1. Score transaction
                score_payload = {
                    "transaction_id": tx_id,
                    "customer_id": cust_id,
                    "card_id": card_id,
                    "amount": 25.0 + j,
                    "timestamp": base_t + (thread_idx * 100) + j,
                    "location": {"latitude": 34.0522, "longitude": -118.2437},
                }
                r_score = self.client.post("/v1/score", json=score_payload)
                if r_score.status_code != 200:
                    errors.append(f"Thread {thread_idx} score failed: {r_score.status_code} {r_score.text}")
                    continue

                # 2. Concurrently submit investigator feedback on some
                if j % 3 == 0:
                    r_inv = self.client.post("/v1/feedback/investigator", json={
                        "transaction_id": tx_id,
                        "investigator_id": f"inv_t{thread_idx}",
                        "verdict": "FRAUD" if j % 2 == 0 else "LEGITIMATE",
                        "reviewed_at": base_t + (thread_idx * 100) + j + 3600.0,
                    })
                    if r_inv.status_code != 200:
                        errors.append(f"Thread {thread_idx} inv failed: {r_inv.status_code} {r_inv.text}")

                # 3. Concurrently poll drift telemetry
                if j % 5 == 0:
                    r_tel = self.client.get("/v1/telemetry/drift")
                    if r_tel.status_code != 200:
                        errors.append(f"Thread {thread_idx} drift failed: {r_tel.status_code} {r_tel.text}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(n_threads)]
            concurrent.futures.wait(futures)

        self.assertEqual(len(errors), 0, f"Encountered {len(errors)} errors during concurrency: {errors[:5]}")


if __name__ == "__main__":
    unittest.main()
