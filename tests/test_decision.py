"""
Tests for StreamGuard 3-Tier Cost-Sensitive Decision Router (Milestone 5).

Verifies:
  1. Parameterized financial cost matrix equations for Approve, Challenge, and Block.
  2. Optimal action selection minimizing expected loss across amount and risk spaces.
  3. Action decisions (Approve, Challenge, Block) across different amount and risk ranges.
  4. Total financial loss reduction >= 25% compared to static tau = 0.5 threshold.
  5. Threshold optimization for analytical and empirical global boundaries.
  6. Sub-10ms decision execution latency contract.
  7. Contract-compliant ScoringResult output payload.
  8. Dynamic boundary recalibration on drift signals (tightening under attack, recovery on normal).
"""

from __future__ import annotations

import math
import time
import unittest
import numpy as np

from app.decision.cost_matrix import (
    ACTION_TO_RISK_TIER,
    CostMatrixConfig,
    DecisionAction,
    RiskTier,
    compute_amount_thresholds,
    compute_expected_losses,
    compute_expected_losses_batch,
    evaluate_policy_expected_loss,
    evaluate_policy_realized_loss,
    expected_loss_approve,
    expected_loss_block,
    expected_loss_challenge,
    optimal_action,
    optimal_action_batch,
    optimize_thresholds,
    ThresholdOptimizer,
)
from app.decision.router import (
    AdaptiveDecisionRouter,
    RecalibrationEvent,
    ScoringResult,
)


class TestCostMatrixFormulation(unittest.TestCase):
    """Unit tests for parameterized expected loss formulas and mathematical invariants."""

    def setUp(self) -> None:
        self.config = CostMatrixConfig(
            c_chargeback=25.0,
            eta=0.95,
            c_otp=0.05,
            c_friction=1.50,
            c_tp_review=2.0,
            c_fp_churn=50.0,
        )

    def test_expected_loss_approve_exact_match(self) -> None:
        """E[Loss(Approve | p, Amount)] = p * (Amount + C_chargeback)."""
        # Case 1: p = 0.0 (legitimate) -> $0 loss
        loss_p0 = expected_loss_approve(p=0.0, amount=100.0, config=self.config)
        self.assertEqual(loss_p0, 0.0)

        # Case 2: p = 1.0 (fraud) -> Amount + C_cb
        loss_p1 = expected_loss_approve(p=1.0, amount=200.0, config=self.config)
        self.assertEqual(loss_p1, 225.0)

        # Case 3: p = 0.10, amount = 75.0 -> 0.10 * (75 + 25) = 10.0
        loss_p01 = expected_loss_approve(p=0.10, amount=75.0, config=self.config)
        self.assertAlmostEqual(loss_p01, 10.0, places=5)

    def test_expected_loss_challenge_exact_match(self) -> None:
        """E[Loss(Challenge | p, Amount)] = p * [(1 - eta)*(Amount + C_cb) + C_otp] + (1 - p)*C_friction."""
        # Case 1: p = 0.0 -> C_friction = 1.50
        loss_p0 = expected_loss_challenge(p=0.0, amount=500.0, config=self.config)
        self.assertEqual(loss_p0, 1.50)

        # Case 2: p = 1.0 -> (1 - 0.95) * (100 + 25) + 0.05 = 0.05 * 125 + 0.05 = 6.25 + 0.05 = 6.30
        loss_p1 = expected_loss_challenge(p=1.0, amount=100.0, config=self.config)
        self.assertAlmostEqual(loss_p1, 6.30, places=5)

        # Case 3: p = 0.50, amount = 100.0
        # 0.5 * 6.30 + 0.5 * 1.50 = 3.15 + 0.75 = 3.90
        loss_p05 = expected_loss_challenge(p=0.50, amount=100.0, config=self.config)
        self.assertAlmostEqual(loss_p05, 3.90, places=5)

    def test_expected_loss_block_exact_match(self) -> None:
        """E[Loss(Block | p, Amount)] = p * C_tp_review + (1 - p) * C_fp_churn."""
        # Case 1: p = 0.0 -> C_fp_churn = 50.0
        loss_p0 = expected_loss_block(p=0.0, amount=100.0, config=self.config)
        self.assertEqual(loss_p0, 50.0)

        # Case 2: p = 1.0 -> C_tp_review = 2.0
        loss_p1 = expected_loss_block(p=1.0, amount=100.0, config=self.config)
        self.assertEqual(loss_p1, 2.0)

        # Case 3: p = 0.20 -> 0.20 * 2.0 + 0.80 * 50.0 = 0.40 + 40.0 = 40.40
        loss_p02 = expected_loss_block(p=0.20, amount=100.0, config=self.config)
        self.assertAlmostEqual(loss_p02, 40.40, places=5)

    def test_boundary_clamping_robustness(self) -> None:
        """Negative amounts or out-of-range probabilities are safely clamped without NaN."""
        losses = compute_expected_losses(p=-0.5, amount=-100.0, config=self.config)
        for act, l in losses.items():
            self.assertFalse(math.isnan(l))
            self.assertFalse(math.isinf(l))
            self.assertGreaterEqual(l, 0.0)

        losses_high = compute_expected_losses(p=1.5, amount=100.0, config=self.config)
        self.assertEqual(losses_high[DecisionAction.APPROVE], 125.0)

    def test_vectorized_batch_computation_parity(self) -> None:
        """Vectorized batch functions produce identical results to scalar evaluations."""
        np.random.seed(42)
        n = 200
        probs = np.random.uniform(0.0, 1.0, size=n)
        amts = np.random.uniform(10.0, 2000.0, size=n)

        loss_app, loss_chal, loss_blk = compute_expected_losses_batch(probs, amts, self.config)
        acts_batch, min_loss_batch = optimal_action_batch(probs, amts, self.config)

        for i in range(n):
            losses_scalar = compute_expected_losses(probs[i], amts[i], self.config)
            self.assertAlmostEqual(loss_app[i], losses_scalar[DecisionAction.APPROVE], places=5)
            self.assertAlmostEqual(loss_chal[i], losses_scalar[DecisionAction.CHALLENGE], places=5)
            self.assertAlmostEqual(loss_blk[i], losses_scalar[DecisionAction.BLOCK], places=5)

            best_act, best_l = optimal_action(probs[i], amts[i], self.config)
            self.assertEqual(acts_batch[i], best_act.value)
            self.assertAlmostEqual(min_loss_batch[i], best_l, places=5)


class TestActionDecisionsAcrossRiskAndAmount(unittest.TestCase):
    """Unit tests verifying Approve, Challenge, and Block routing across risk and amount spaces."""

    def setUp(self) -> None:
        self.config = CostMatrixConfig()

    def test_low_risk_transactions_approved(self) -> None:
        """Transactions with negligible risk (p < 0.005) must be Approved across all amounts."""
        for amt in [5.0, 50.0, 250.0, 1500.0]:
            action, loss = optimal_action(p=0.001, amount=amt, config=self.config)
            self.assertEqual(
                action,
                DecisionAction.APPROVE,
                f"Low risk transaction (amt={amt}) was not approved",
            )
            self.assertLess(loss, 2.0)

    def test_high_risk_transactions_blocked(self) -> None:
        """Transactions with high fraud probability (p > 0.90) must be Blocked."""
        for amt in [100.0, 500.0, 2000.0, 10000.0]:
            action, loss = optimal_action(p=0.95, amount=amt, config=self.config)
            self.assertEqual(
                action,
                DecisionAction.BLOCK,
                f"High risk transaction (amt={amt}) was not blocked",
            )

    def test_intermediate_risk_transactions_challenged(self) -> None:
        """Transactions with moderate risk (e.g. p = 0.15, amount = $250) must trigger Challenge."""
        action, loss = optimal_action(p=0.15, amount=250.0, config=self.config)
        self.assertEqual(action, DecisionAction.CHALLENGE)
        # Expected losses:
        # Approve: 0.15 * 275 = 41.25
        # Challenge: 0.15 * (0.05 * 275 + 0.05) + 0.85 * 1.50 = 2.07 + 1.275 = 3.345
        # Block: 0.15 * 2.0 + 0.85 * 50.0 = 0.30 + 42.50 = 42.80
        # Challenge is vastly cheaper than Approve (41.25) or Block (42.80)
        self.assertLess(loss, 10.0)

    def test_amount_sensitivity_shifts_challenge_threshold(self) -> None:
        """Higher transaction amounts lower the threshold at which Challenge becomes optimal."""
        p = 0.04  # 4% fraud risk
        # For small $20 transaction:
        # Approve loss = 0.04 * (20 + 25) = 1.80
        # Challenge loss = 0.04 * (0.05 * 45 + 0.05) + 0.96 * 1.50 = 0.092 + 1.44 = 1.532
        # For small $10 transaction:
        # Approve loss = 0.04 * (10 + 25) = 1.40 < Challenge (1.49) -> APPROVE
        act_small, _ = optimal_action(p=0.03, amount=10.0, config=self.config)
        self.assertEqual(act_small, DecisionAction.APPROVE)

        # For $1,000 transaction:
        # Approve loss = 0.03 * (1000 + 25) = 30.75
        # Challenge loss = 0.03 * (0.05 * 1025 + 0.05) + 0.97 * 1.50 = 1.539 + 1.455 = 2.99 -> CHALLENGE
        act_large, _ = optimal_action(p=0.03, amount=1000.0, config=self.config)
        self.assertEqual(act_large, DecisionAction.CHALLENGE)

    def test_amount_dependent_threshold_invariants(self) -> None:
        """Analytical amount thresholds must satisfy 0 < tau_challenge < tau_block < 1."""
        for amt in [10.0, 50.0, 200.0, 1000.0, 5000.0]:
            t_chal, t_blk = compute_amount_thresholds(amt, self.config)
            self.assertGreater(t_chal, 0.0)
            self.assertLess(t_chal, t_blk)
            self.assertLess(t_blk, 1.0)


class TestFinancialLossReductionOverStatic(unittest.TestCase):
    """
    Unit tests verifying that 3-tier adaptive routing reduces total financial losses
    by >= 25% compared to static tau = 0.5 threshold.
    """

    def setUp(self) -> None:
        self.config = CostMatrixConfig()

    def test_loss_reduction_exceeds_25_percent_on_synthetic_stream(self) -> None:
        """Realized financial loss reduction under 3-tier routing is >= 25% vs static tau = 0.5."""
        np.random.seed(42)
        n = 5000
        # Extreme class imbalance: 1.5% fraud rate
        true_labels = (np.random.rand(n) < 0.015).astype(np.int32)
        n_fraud = int(np.sum(true_labels))

        # Calibrated risk scores: legit centered near 0.01, fraud spread between 0.10 and 0.85
        probs = np.zeros(n, dtype=np.float64)
        probs[true_labels == 0] = np.random.beta(a=0.3, b=30.0, size=n - n_fraud)
        probs[true_labels == 1] = np.random.beta(a=2.0, b=4.0, size=n_fraud)

        # Log-normal transaction amounts
        amounts = np.exp(np.random.normal(loc=4.5, scale=1.0, size=n))
        amounts = np.clip(amounts, 10.0, 3000.0)

        # Evaluate policy
        results = evaluate_policy_realized_loss(
            probabilities=probs,
            amounts=amounts,
            true_labels=true_labels,
            config=self.config,
            use_amount_dependent_thresholds=True,
        )

        loss_static = results["loss_static"]
        loss_3tier = results["loss_3tier"]
        reduction = results["loss_reduction_pct"]

        self.assertGreater(loss_static, loss_3tier, "3-Tier loss was not strictly lower than static")
        self.assertGreaterEqual(
            reduction,
            25.0,
            f"Financial loss reduction was {reduction:.2f}%, which is below the 25.0% target!",
        )
        self.assertGreater(results["action_counts"][DecisionAction.APPROVE.value], 0)
        self.assertGreater(results["action_counts"][DecisionAction.CHALLENGE.value], 0)
        self.assertGreater(results["action_counts"][DecisionAction.BLOCK.value], 0)

    def test_expected_loss_reduction_exceeds_25_percent(self) -> None:
        """Expected financial loss reduction without true labels is >= 25% vs static tau = 0.5."""
        np.random.seed(123)
        n = 5000
        probs = np.random.beta(a=0.5, b=20.0, size=n)
        amounts = np.random.uniform(20.0, 1000.0, size=n)

        results = evaluate_policy_expected_loss(
            probabilities=probs,
            amounts=amounts,
            config=self.config,
            use_amount_dependent_thresholds=True,
        )

        reduction = results["loss_reduction_pct"]
        self.assertGreaterEqual(
            reduction,
            25.0,
            f"Expected financial loss reduction was {reduction:.2f}%, below target!",
        )

    def test_threshold_optimizer_discovers_optimal_cutoffs(self) -> None:
        """ThresholdOptimizer discovers boundaries that beat static threshold by >= 25%."""
        np.random.seed(77)
        n = 2000
        true_labels = (np.random.rand(n) < 0.02).astype(np.int32)
        probs = np.where(
            true_labels == 1,
            np.random.beta(a=3.0, b=4.0, size=n),
            np.random.beta(a=0.4, b=25.0, size=n),
        )
        amounts = np.random.uniform(50.0, 800.0, size=n)

        optimizer = ThresholdOptimizer(config=self.config)
        t_chal, t_blk, opt_loss = optimizer.optimize(
            probabilities=probs,
            amounts=amounts,
            true_labels=true_labels,
            grid_points=30,
        )

        self.assertLess(t_chal, t_blk)
        self.assertGreater(t_chal, 0.01)
        self.assertLess(t_blk, 0.95)

        # Check that loss with these fixed thresholds beats static
        res = evaluate_policy_realized_loss(
            probabilities=probs,
            amounts=amounts,
            true_labels=true_labels,
            tau_challenge=t_chal,
            tau_block=t_blk,
            config=self.config,
            use_amount_dependent_thresholds=False,
        )
        self.assertGreaterEqual(res["loss_reduction_pct"], 25.0)


class TestAdaptiveDecisionRouter(unittest.TestCase):
    """Unit tests for AdaptiveDecisionRouter execution, scoring results, and sub-10ms latency."""

    def setUp(self) -> None:
        self.router = AdaptiveDecisionRouter(
            default_tau_challenge=0.15,
            default_tau_block=0.70,
            mode="expected_loss",
        )

    def test_scoring_result_contract_and_structure(self) -> None:
        """ScoringResult must contain all contract fields and valid types."""
        result = self.router.route(
            transaction_id="tx_test_101",
            fraud_score=0.12,
            amount=150.0,
            features={"velocity_5m_count": 2, "amount_to_mean24h_ratio": 1.4},
        )

        self.assertIsInstance(result, ScoringResult)
        self.assertEqual(result.transaction_id, "tx_test_101")
        self.assertAlmostEqual(result.fraud_score, 0.12)
        self.assertIn(result.decision, ["APPROVE", "CHALLENGE", "BLOCK"])
        self.assertIn(result.risk_tier, ["LOW", "MEDIUM", "HIGH"])
        self.assertIsInstance(result.latency_ms, float)
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertIsNotNone(result.evaluated_at)
        self.assertEqual(result.features["velocity_5m_count"], 2)

        # Dictionary serialization
        d = result.to_dict()
        self.assertEqual(d["transaction_id"], "tx_test_101")
        self.assertEqual(d["decision"], result.decision)
        self.assertEqual(d["risk_tier"], result.risk_tier)

    def test_sub_10ms_scoring_latency_budget(self) -> None:
        """Single-transaction decision routing must strictly execute in < 10ms (target: < 0.1ms)."""
        latencies = []
        for i in range(1000):
            st = time.perf_counter()
            res = self.router.route(
                transaction_id=f"tx_perf_{i}",
                fraud_score=0.08,
                amount=85.0,
                start_time=st,
            )
            latencies.append(res.latency_ms)

        p50 = float(np.percentile(latencies, 50))
        p99 = float(np.percentile(latencies, 99))
        self.assertLess(
            p99,
            10.0,
            f"P99 routing latency {p99:.4f}ms exceeded strict 10.0ms budget!",
        )
        # Even stricter check: pure math routing should be sub-millisecond
        self.assertLess(p50, 1.0, f"P50 routing latency {p50:.4f}ms was unexpectedly high")

    def test_fixed_threshold_mode_routing(self) -> None:
        """In 'threshold' mode, routing strictly follows active tau_challenge and tau_block."""
        router = AdaptiveDecisionRouter(
            default_tau_challenge=0.20,
            default_tau_block=0.80,
            mode="threshold",
        )

        # Score < 0.20 -> APPROVE (LOW)
        res_app = router.route("tx_1", fraud_score=0.15, amount=100.0)
        self.assertEqual(res_app.decision, "APPROVE")
        self.assertEqual(res_app.risk_tier, "LOW")
        self.assertTrue(res_app.is_approved)

        # 0.20 <= Score < 0.80 -> CHALLENGE (MEDIUM)
        res_chal = router.route("tx_2", fraud_score=0.45, amount=100.0)
        self.assertEqual(res_chal.decision, "CHALLENGE")
        self.assertEqual(res_chal.risk_tier, "MEDIUM")
        self.assertTrue(res_chal.is_challenged)

        # Score >= 0.80 -> BLOCK (HIGH)
        res_blk = router.route("tx_3", fraud_score=0.85, amount=100.0)
        self.assertEqual(res_blk.decision, "BLOCK")
        self.assertEqual(res_blk.risk_tier, "HIGH")
        self.assertTrue(res_blk.is_blocked)

    def test_batch_routing_consistency(self) -> None:
        """route_batch produces identical results to individual route calls."""
        batch_input = [
            ("tx_b1", 0.01, 50.0, {"v": 1}),
            ("tx_b2", 0.35, 300.0, {"v": 4}),
            ("tx_b3", 0.92, 1200.0, {"v": 10}),
        ]
        batch_results = self.router.route_batch(batch_input)
        self.assertEqual(len(batch_results), 3)

        for i, (tx_id, score, amt, feats) in enumerate(batch_input):
            single = self.router.route(tx_id, score, amt, features=feats)
            self.assertEqual(batch_results[i].decision, single.decision)
            self.assertEqual(batch_results[i].risk_tier, single.risk_tier)

    def test_telemetry_reporting(self) -> None:
        """Router tracks and reports telemetry accurately."""
        router = AdaptiveDecisionRouter()
        router.route("tx_1", 0.001, 20.0)  # Approve
        router.route("tx_2", 0.20, 500.0)  # Challenge
        router.route("tx_3", 0.98, 2000.0) # Block

        telemetry = router.get_telemetry()
        self.assertEqual(telemetry["total_routed"], 3)
        self.assertEqual(telemetry["action_counts"]["APPROVE"], 1)
        self.assertEqual(telemetry["action_counts"]["CHALLENGE"], 1)
        self.assertEqual(telemetry["action_counts"]["BLOCK"], 1)
        self.assertGreater(telemetry["total_expected_loss"], 0.0)
        self.assertIn("tau_challenge", telemetry["current_thresholds"])


class TestDynamicRecalibrationOnDrift(unittest.TestCase):
    """Unit tests for dynamic boundary recalibration hook on drift signals."""

    def setUp(self) -> None:
        self.router = AdaptiveDecisionRouter(
            default_tau_challenge=0.15,
            default_tau_block=0.70,
            mode="threshold",
            defensive_drift_discount=0.60,
        )

    def test_manual_recalibration_updates_boundaries_and_audit_log(self) -> None:
        """Direct threshold update recalibrates boundaries and records audit log."""
        old_t1 = self.router.tau_challenge
        old_t2 = self.router.tau_block

        new_t1, new_t2 = self.router.recalibrate(
            new_tau_challenge=0.10,
            new_tau_block=0.60,
            reason="Investigator policy shift",
        )

        self.assertEqual(new_t1, 0.10)
        self.assertEqual(new_t2, 0.60)
        self.assertEqual(self.router.tau_challenge, 0.10)
        self.assertEqual(self.router.tau_block, 0.60)

        self.assertEqual(len(self.router.recalibration_history), 1)
        ev = self.router.recalibration_history[0]
        self.assertEqual(ev.old_tau_challenge, old_t1)
        self.assertEqual(ev.old_tau_block, old_t2)
        self.assertEqual(ev.new_tau_challenge, 0.10)
        self.assertEqual(ev.new_tau_block, 0.60)
        self.assertEqual(ev.reason, "Investigator policy shift")

    def test_drift_alert_tightens_thresholds_defensively(self) -> None:
        """When an unsupervised drift alert arrives (PSI >= 0.25), thresholds tighten defensively."""
        base_t1 = self.router.tau_challenge
        base_t2 = self.router.tau_block

        # Simulated drift telemetry payload from M4 drift monitor
        drift_alert_payload = {
            "drift_status": "DRIFT_DETECTED",
            "drift_alert": True,
            "score_psi": 0.354,
            "score_wasserstein": 0.082,
        }

        t1_alert, t2_alert = self.router.recalibrate(drift_signal=drift_alert_payload)

        # Thresholds must become strictly tighter to challenge and block more aggressively
        self.assertLess(t1_alert, base_t1, "Challenge threshold was not tightened during drift")
        self.assertLess(t2_alert, base_t2, "Block threshold was not tightened during drift")

        # Verify routing behavior changes: a transaction previously approved is now challenged
        # Score = 0.12 was < 0.15 (Approved). Under tightened threshold (e.g. 0.09), it is Challenged!
        res_under_drift = self.router.route("tx_drift_test", fraud_score=0.12, amount=100.0)
        self.assertEqual(
            res_under_drift.decision,
            "CHALLENGE",
            "Transaction should be challenged under tightened drift boundaries",
        )

    def test_drift_recovery_restores_normal_thresholds(self) -> None:
        """When drift clears (drift_alert=False), thresholds return to base levels."""
        # 1. Trigger drift
        self.router.recalibrate(drift_signal={"drift_alert": True, "score_psi": 0.40})
        self.assertLess(self.router.tau_challenge, self.router.base_tau_challenge)

        # 2. Recover from drift
        self.router.recalibrate(drift_signal={"drift_alert": False, "score_psi": 0.04})
        self.assertEqual(self.router.tau_challenge, self.router.base_tau_challenge)
        self.assertEqual(self.router.tau_block, self.router.base_tau_block)

    def test_recalibration_listener_invocation(self) -> None:
        """Subscribed callbacks receive RecalibrationEvent notifications."""
        received_events = []

        def on_recalibrate(event: RecalibrationEvent) -> None:
            received_events.append(event)

        self.router.register_recalibration_listener(on_recalibrate)
        self.router.recalibrate(new_tau_challenge=0.11, new_tau_block=0.65, reason="Listener test")

        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].new_tau_challenge, 0.11)
        self.assertEqual(received_events[0].new_tau_block, 0.65)
        self.assertEqual(received_events[0].reason, "Listener test")

    def test_reset_thresholds_restores_initial_values(self) -> None:
        """reset_thresholds() restores initial base boundaries."""
        self.router.recalibrate(new_tau_challenge=0.05, new_tau_block=0.40)
        self.router.reset_thresholds()

        self.assertEqual(self.router.tau_challenge, self.router.base_tau_challenge)
        self.assertEqual(self.router.tau_block, self.router.base_tau_block)


if __name__ == "__main__":
    unittest.main()
