"""Tier 5: FastAPI Serving Layer Tests for StreamGuard Fraud Engine.

Verifies:
  1. POST /v1/score:
     - Returns HTTP 200 with valid schema, risk score, decision, risk tier, and sub-15ms latency.
     - Correctly computes multi-window velocity features across sequential customer events.
     - Flags impossible travel velocity when consecutive locations span distant coordinates.
  2. POST /v1/feedback/investigator:
     - Returns HTTP 200 with ACCEPTED status and provisional label for active transactions.
     - Returns HTTP 404 for unknown or expired transactions.
     - Enforces strict verdict validation ('FRAUD' or 'LEGITIMATE'), returning HTTP 422 otherwise.
  3. POST /v1/feedback/chargeback:
     - Returns HTTP 200 with CONSOLIDATED status and final ground-truth label.
     - Returns HTTP 404 for unknown transactions.
     - Enforces strict ground-truth precedence: chargeback dispute strictly overrides investigator clearance.
  4. GET /v1/telemetry/drift:
     - Returns HTTP 200 with valid schema, active thresholds, PSI, and Wasserstein metrics.
     - Validates query parameters, rejecting invalid features, metrics, or negative window sizes with HTTP 400.
  5. Validation & Edge Cases:
     - Negative or zero amounts return HTTP 422.
     - Invalid latitude/longitude coordinates return HTTP 422.
     - Missing mandatory fields return HTTP 422.
  6. Concurrency:
     - Multi-threaded concurrent scoring requests execute safely without state corruption or race conditions.
"""

from __future__ import annotations

import concurrent.futures
import time
import unittest
from typing import Any, Dict

from fastapi.testclient import TestClient

from app.api.schemas import (
    ChargebackFeedbackResponse,
    DriftTelemetryResponse,
    InvestigatorFeedbackResponse,
    ScoreResponse,
)
from app.main import app, get_global_engine


class TestFastAPIScoreEndpoint(unittest.TestCase):
    """Test suite for POST /v1/score endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.engine = get_global_engine()

    def test_valid_score_request_schema_and_latency(self) -> None:
        """Valid score request returns HTTP 200 and sub-15ms latency."""
        payload = {
            "transaction_id": "tx_score_001",
            "customer_id": "cust_score_alpha",
            "card_id": "card_score_alpha",
            "amount": 149.99,
            "currency": "USD",
            "timestamp": 1729080000.0,
            "merchant_id": "merch_electronics_101",
            "merchant_category_code": "5732",
            "channel": "e-commerce",
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
            },
            "device_id": "dev_test_01",
            "ip_address": "198.51.100.10",
        }

        response = self.client.post("/v1/score", json=payload)
        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}: {response.text}")

        data = response.json()
        validated = ScoreResponse(**data)

        self.assertEqual(validated.transaction_id, "tx_score_001")
        self.assertGreaterEqual(validated.fraud_score, 0.0)
        self.assertLessEqual(validated.fraud_score, 1.0)
        self.assertIn(validated.decision, ["APPROVE", "CHALLENGE", "BLOCK"])
        self.assertIn(validated.risk_tier, ["LOW", "MEDIUM", "HIGH"])
        self.assertLess(validated.latency_ms, 15.0, f"Latency {validated.latency_ms} ms exceeded 15ms target")
        self.assertIsNotNone(validated.features)
        self.assertIn("T", validated.evaluated_at)

    def test_sequential_transactions_update_sliding_windows(self) -> None:
        """Sequential transactions from the same customer increment window velocity counts."""
        cust_id = "cust_seq_test"
        card_id = "card_seq_test"
        t_base = 1729081000.0

        for i in range(3):
            payload = {
                "transaction_id": f"tx_seq_{i}",
                "customer_id": cust_id,
                "card_id": card_id,
                "amount": 50.0 + (i * 10.0),
                "timestamp": t_base + (i * 60.0),  # 1 minute apart (within 5m window)
                "location": {"latitude": 34.0522, "longitude": -118.2437},
            }
            resp = self.client.post("/v1/score", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()

            # The 0th transaction has 0 prior counts (strict non-leakage)
            # The 1st transaction has 1 prior count
            # The 2nd transaction has 2 prior counts
            expected_prior_5m = i
            self.assertEqual(
                data["features"]["velocity_5m_count"],
                expected_prior_5m,
                f"Expected prior 5m count {expected_prior_5m}, got {data['features']['velocity_5m_count']}",
            )

    def test_impossible_geographic_travel_detection(self) -> None:
        """Consecutive transactions across continents within 10 minutes flag impossible travel."""
        cust_id = "cust_travel_jump"
        card_id = "card_travel_jump"
        t0 = 1729082000.0

        # Event 1: New York (40.7128, -74.0060)
        resp1 = self.client.post("/v1/score", json={
            "transaction_id": "tx_ny_01",
            "customer_id": cust_id,
            "card_id": card_id,
            "amount": 45.0,
            "timestamp": t0,
            "location": {"latitude": 40.7128, "longitude": -74.0060},
        })
        self.assertEqual(resp1.status_code, 200)

        # Event 2: London (51.5074, -0.1278) only 10 minutes (600s) later (~5570 km jump, >33,000 km/h)
        resp2 = self.client.post("/v1/score", json={
            "transaction_id": "tx_london_02",
            "customer_id": cust_id,
            "card_id": card_id,
            "amount": 950.0,
            "timestamp": t0 + 600.0,
            "location": {"latitude": 51.5074, "longitude": -0.1278},
        })
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()

        self.assertEqual(data2["features"]["impossible_travel_flag"], 1)
        self.assertGreater(data2["features"]["travel_speed_kmh"], 800.0)


class TestFastAPIFeedbackEndpoints(unittest.TestCase):
    """Test suite for human investigator and bank chargeback feedback endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_investigator_feedback_success(self) -> None:
        """Fast human investigator review updates working memory state."""
        tx_id = "tx_inv_valid_01"
        # 1. Ingest transaction first
        score_resp = self.client.post("/v1/score", json={
            "transaction_id": tx_id,
            "customer_id": "cust_inv_01",
            "card_id": "card_inv_01",
            "amount": 250.0,
            "timestamp": 1729083000.0,
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        })
        self.assertEqual(score_resp.status_code, 200)

        # 2. Ingest investigator feedback
        inv_payload = {
            "transaction_id": tx_id,
            "investigator_id": "inv_agent_42",
            "verdict": "FRAUD",
            "reviewed_at": 1729086600.0,  # 1 hour later
            "confidence": 0.98,
            "notes": "Confirmed suspicious credential stuffing and immediate jewelry order.",
        }
        resp = self.client.post("/v1/feedback/investigator", json=inv_payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        validated = InvestigatorFeedbackResponse(**data)
        self.assertEqual(validated.status, "ACCEPTED")
        self.assertEqual(validated.transaction_id, tx_id)
        self.assertEqual(validated.memory_target, "FAST_WORKING_MEMORY")
        self.assertEqual(validated.provisional_label, 1)

    def test_investigator_feedback_not_found(self) -> None:
        """Investigator review on unknown transaction ID returns HTTP 404."""
        resp = self.client.post("/v1/feedback/investigator", json={
            "transaction_id": "tx_non_existent_99999",
            "investigator_id": "inv_agent_01",
            "verdict": "LEGITIMATE",
            "reviewed_at": 1729085000.0,
        })
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())

    def test_investigator_feedback_invalid_verdict_returns_422(self) -> None:
        """Verdict other than 'FRAUD' or 'LEGITIMATE' fails validation with HTTP 422."""
        resp = self.client.post("/v1/feedback/investigator", json={
            "transaction_id": "tx_some_id",
            "investigator_id": "inv_01",
            "verdict": "MAYBE_FRAUD",
            "reviewed_at": 1729085000.0,
        })
        self.assertEqual(resp.status_code, 422)

    def test_chargeback_feedback_success_and_label_precedence(self) -> None:
        """Chargeback feedback reconciles into consolidated memory with strict fraud precedence."""
        tx_id = "tx_cb_precedence_01"
        # 1. Ingest transaction
        self.client.post("/v1/score", json={
            "transaction_id": tx_id,
            "customer_id": "cust_cb_01",
            "card_id": "card_cb_01",
            "amount": 320.0,
            "timestamp": 1729084000.0,
            "location": {"latitude": 41.8781, "longitude": -87.6298},
        })

        # 2. Investigator preliminarily cleared transaction (verdict='LEGITIMATE')
        inv_resp = self.client.post("/v1/feedback/investigator", json={
            "transaction_id": tx_id,
            "investigator_id": "inv_agent_09",
            "verdict": "LEGITIMATE",
            "reviewed_at": 1729091200.0,
            "confidence": 0.85,
        })
        self.assertEqual(inv_resp.status_code, 200)
        self.assertEqual(inv_resp.json()["provisional_label"], 0)

        # 3. 20 days later, definitive cardholder chargeback dispute arrives
        cb_payload = {
            "transaction_id": tx_id,
            "chargeback_id": "cb_dispute_8841",
            "reason_code": "10.4_FRAUD_CARD_ABSENT",
            "chargeback_amount": 320.0,
            "received_at": 1730812000.0,
            "is_fraud": True,
        }
        cb_resp = self.client.post("/v1/feedback/chargeback", json=cb_payload)
        self.assertEqual(cb_resp.status_code, 200)

        data = cb_resp.json()
        validated = ChargebackFeedbackResponse(**data)
        self.assertEqual(validated.status, "CONSOLIDATED")
        self.assertEqual(validated.transaction_id, tx_id)
        self.assertEqual(validated.memory_target, "CONSOLIDATED_GROUND_TRUTH")
        self.assertEqual(validated.final_label, 1)
        self.assertTrue(validated.reconciled_with_investigator)

    def test_chargeback_feedback_not_found(self) -> None:
        """Chargeback notice on unknown transaction ID returns HTTP 404."""
        resp = self.client.post("/v1/feedback/chargeback", json={
            "transaction_id": "tx_ghost_77777",
            "chargeback_id": "cb_ghost_01",
            "chargeback_amount": 100.0,
            "received_at": 1730000000.0,
            "is_fraud": True,
        })
        self.assertEqual(resp.status_code, 404)


class TestFastAPITelemetryEndpoint(unittest.TestCase):
    """Test suite for GET /v1/telemetry/drift endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_get_drift_telemetry_schema(self) -> None:
        """Drift telemetry endpoint returns valid statistics and active thresholds."""
        resp = self.client.get("/v1/telemetry/drift")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        validated = DriftTelemetryResponse(**data)

        self.assertGreaterEqual(validated.window_samples, 0)
        self.assertGreaterEqual(validated.score_psi, 0.0)
        self.assertGreaterEqual(validated.score_wasserstein, 0.0)
        self.assertIn(validated.drift_status, ["STABLE", "WARNING", "DRIFT_DETECTED"])
        self.assertIsInstance(validated.drift_alert, bool)
        self.assertIn("tau_challenge", validated.current_thresholds)
        self.assertIn("tau_block", validated.current_thresholds)
        self.assertGreater(validated.current_thresholds["tau_block"], validated.current_thresholds["tau_challenge"])

    def test_drift_telemetry_query_param_validation(self) -> None:
        """Query parameters are strictly validated."""
        # 1. Invalid feature returns 400
        r1 = self.client.get("/v1/telemetry/drift?feature=unsupported_feature")
        self.assertEqual(r1.status_code, 400)

        # 2. Invalid metric returns 400
        r2 = self.client.get("/v1/telemetry/drift?metric=chi_square")
        self.assertEqual(r2.status_code, 400)

        # 3. Non-positive window size returns 400
        r3 = self.client.get("/v1/telemetry/drift?window_size=-10")
        self.assertEqual(r3.status_code, 400)

        # 4. Valid custom query params return 200
        r4 = self.client.get("/v1/telemetry/drift?feature=amount&metric=psi&window_size=500")
        self.assertEqual(r4.status_code, 200)


class TestFastAPIValidationAndEdgeCases(unittest.TestCase):
    """Test suite for payload validation, constraints, and error handling."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_negative_or_zero_amount_returns_422(self) -> None:
        """Non-positive transaction amounts fail Pydantic validation with HTTP 422."""
        base_payload = {
            "transaction_id": "tx_val_amt",
            "customer_id": "c1",
            "card_id": "k1",
            "timestamp": 1729080000.0,
            "location": {"latitude": 0.0, "longitude": 0.0},
        }

        # Amount = 0.0
        r_zero = self.client.post("/v1/score", json={**base_payload, "amount": 0.0})
        self.assertEqual(r_zero.status_code, 422)

        # Amount = -25.50
        r_neg = self.client.post("/v1/score", json={**base_payload, "amount": -25.50})
        self.assertEqual(r_neg.status_code, 422)

    def test_invalid_coordinates_return_422(self) -> None:
        """Out-of-bounds geographic coordinates fail validation with HTTP 422."""
        base_payload = {
            "transaction_id": "tx_val_geo",
            "customer_id": "c1",
            "card_id": "k1",
            "amount": 10.0,
            "timestamp": 1729080000.0,
        }

        # Latitude > 90
        r_lat_high = self.client.post("/v1/score", json={
            **base_payload,
            "location": {"latitude": 95.0, "longitude": 0.0},
        })
        self.assertEqual(r_lat_high.status_code, 422)

        # Latitude < -90
        r_lat_low = self.client.post("/v1/score", json={
            **base_payload,
            "location": {"latitude": -95.0, "longitude": 0.0},
        })
        self.assertEqual(r_lat_low.status_code, 422)

        # Longitude > 180
        r_lon_high = self.client.post("/v1/score", json={
            **base_payload,
            "location": {"latitude": 0.0, "longitude": 185.0},
        })
        self.assertEqual(r_lon_high.status_code, 422)

        # Longitude < -180
        r_lon_low = self.client.post("/v1/score", json={
            **base_payload,
            "location": {"latitude": 0.0, "longitude": -185.0},
        })
        self.assertEqual(r_lon_low.status_code, 422)

    def test_missing_mandatory_fields_return_422(self) -> None:
        """Omission of required fields produces HTTP 422."""
        # Missing location entirely
        r1 = self.client.post("/v1/score", json={
            "transaction_id": "tx_missing_loc",
            "customer_id": "c1",
            "card_id": "k1",
            "amount": 50.0,
            "timestamp": 1729080000.0,
        })
        self.assertEqual(r1.status_code, 422)

        # Empty transaction_id
        r2 = self.client.post("/v1/score", json={
            "transaction_id": "",
            "customer_id": "c1",
            "card_id": "k1",
            "amount": 50.0,
            "timestamp": 1729080000.0,
            "location": {"latitude": 10.0, "longitude": 10.0},
        })
        self.assertEqual(r2.status_code, 422)


class TestFastAPIConcurrency(unittest.TestCase):
    """Test suite for concurrent request safety."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_concurrent_scoring_requests(self) -> None:
        """50 concurrent transactions scored across 8 threads complete without race conditions."""
        n_requests = 50
        base_time = 1729090000.0

        def score_tx(idx: int) -> int:
            payload = {
                "transaction_id": f"tx_conc_{idx}",
                "customer_id": f"cust_conc_{idx % 5}",
                "card_id": f"card_conc_{idx % 5}",
                "amount": 20.0 + (idx * 2.0),
                "timestamp": base_time + idx,
                "location": {
                    "latitude": 40.7128 + (idx * 0.001),
                    "longitude": -74.0060 + (idx * 0.001),
                },
            }
            resp = self.client.post("/v1/score", json=payload)
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            status_codes = list(executor.map(score_tx, range(n_requests)))

        self.assertEqual(len(status_codes), n_requests)
        self.assertTrue(all(code == 200 for code in status_codes), f"Some requests failed: {status_codes}")


if __name__ == "__main__":
    unittest.main()
