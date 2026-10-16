"""Comprehensive Unit and Integration Tests for StreamGuard Memory Subsystem.

Verifies:
  1. Fast investigator review queue (1-4 hours review latency) with investigator verdict,
     confidence, and notes.
  2. Delayed bank chargeback queue (14-30 days dispute latency) with chargeback ID,
     reason code, amount, and true dispute flag.
  3. Event-time gating ensuring events are emitted only when sim_time >= scheduled_arrival_time
     (strict zero temporal leakage).
  4. Working Memory (M_fast) ingesting transactions from the active 30-day window, updating
     states (PROVISIONAL_UNLABELED, INVESTIGATOR_FLAGGED, INVESTIGATOR_CLEARED, CHARGEBACK_CONFIRMED).
  5. Consolidated Replay Buffer (M_cons) holding mature records (elapsed time >= 30 days or
     definitive chargeback received).
  6. Strict Label Precedence & Reconciliation: A chargeback dispute (y=1) strictly overrides
     a prior investigator clearance (y=0).
  7. Delayed-Feedback Debiaser: Eliminates the 50% false-negative contamination from naive
     label truncation (unreviewed transactions within 30 days are NOT assumed to be non-fraud).
"""

from __future__ import annotations

import random
import unittest
import numpy as np

from app.memory.feedback_queue import (
    ChargebackFeedback,
    DelayedFeedbackQueue,
    FeedbackChannel,
    FeedbackPriorityQueue,
    InvestigatorFeedback,
    InvestigatorVerdict,
)
from app.memory.dual_memory import (
    ConsolidatedMemory,
    DelayedFeedbackDebiaser,
    DualMemoryManager,
    MemoryRecord,
    TransactionLabelState,
    WorkingMemory,
)


class TestFeedbackPriorityQueue(unittest.TestCase):
    """Unit tests for feedback priority queue, scheduling, and event-time gating."""

    def setUp(self) -> None:
        self.queue = FeedbackPriorityQueue()
        self.t0 = 1729080000.0  # reference epoch timestamp

    def test_investigator_scheduling_and_temporal_gating(self) -> None:
        """Fast feedback scheduled at t0+2h is invisible at t0+1h and emitted at t0+2.5h."""
        delivery_time = self.queue.schedule_investigator_feedback(
            transaction_id="tx_inv_01",
            tx_timestamp=self.t0,
            is_fraud=True,
            delay_seconds=7200.0,  # 2 hours
            investigator_id="inv_test_01",
            verdict="FRAUD",
            confidence=0.98,
            notes="Suspicious impossible travel pattern",
        )
        self.assertEqual(delivery_time, self.t0 + 7200.0)
        self.assertEqual(self.queue.pending_count(), 1)
        self.assertEqual(self.queue.pending_investigator_count(), 1)
        self.assertEqual(self.queue.pending_chargeback_count(), 0)

        # Poll at t0 + 3600s (1 hour): event must NOT mature yet (Zero temporal leakage)
        inv_early, cb_early = self.queue.poll(self.t0 + 3600.0)
        self.assertEqual(len(inv_early), 0, "Temporal leakage: fast review emitted early!")
        self.assertEqual(len(cb_early), 0)
        self.assertEqual(self.queue.pending_count(), 1)

        # Poll at t0 + 7199.9s: strictly before scheduled delivery
        inv_before, _ = self.queue.poll(self.t0 + 7199.9)
        self.assertEqual(len(inv_before), 0, "Strict event-time boundary violated!")

        # Poll at t0 + 7200.0s: event matures exactly at delivery time
        inv_ready, cb_ready = self.queue.poll(self.t0 + 7200.0)
        self.assertEqual(len(inv_ready), 1)
        self.assertEqual(len(cb_ready), 0)
        fb = inv_ready[0]
        self.assertEqual(fb.transaction_id, "tx_inv_01")
        self.assertEqual(fb.investigator_id, "inv_test_01")
        self.assertEqual(fb.verdict, "FRAUD")
        self.assertTrue(fb.is_fraud)
        self.assertAlmostEqual(fb.confidence, 0.98)
        self.assertEqual(fb.notes, "Suspicious impossible travel pattern")
        self.assertEqual(self.queue.pending_count(), 0)

    def test_chargeback_scheduling_and_temporal_gating(self) -> None:
        """Delayed chargeback scheduled at t0+20d is invisible at t0+10d and emitted at t0+20d."""
        delay_20d = 20.0 * 86400.0
        delivery_time = self.queue.schedule_chargeback_feedback(
            transaction_id="tx_cb_01",
            tx_timestamp=self.t0,
            amount=750.50,
            is_fraud=True,
            delay_seconds=delay_20d,
            reason_code="10.4_FRAUD_CARD_ABSENT",
            chargeback_id="cb_999001",
        )
        self.assertEqual(delivery_time, self.t0 + delay_20d)
        self.assertEqual(self.queue.pending_chargeback_count(), 1)

        # Poll at t0 + 10 days: strictly zero leakage
        _, cb_10d = self.queue.poll(self.t0 + 10.0 * 86400.0)
        self.assertEqual(len(cb_10d), 0, "Temporal leakage: chargeback emitted early!")

        # Poll at t0 + 19.99 days
        _, cb_19d = self.queue.poll(self.t0 + 19.99 * 86400.0)
        self.assertEqual(len(cb_19d), 0)

        # Poll at t0 + 20 days: matured and emitted
        _, cb_20d = self.queue.poll(self.t0 + 20.0 * 86400.0)
        self.assertEqual(len(cb_20d), 1)
        fb = cb_20d[0]
        self.assertEqual(fb.transaction_id, "tx_cb_01")
        self.assertEqual(fb.chargeback_id, "cb_999001")
        self.assertEqual(fb.reason_code, "10.4_FRAUD_CARD_ABSENT")
        self.assertEqual(fb.chargeback_amount, 750.50)
        self.assertTrue(fb.is_fraud)
        self.assertEqual(self.queue.pending_count(), 0)

    def test_direct_enqueue_and_priority_ordering(self) -> None:
        """Directly enqueued events arrive in strict ascending scheduled time order."""
        inv1 = InvestigatorFeedback(
            transaction_id="tx_1",
            investigator_id="inv_1",
            verdict="LEGITIMATE",
            confidence=0.90,
            reviewed_at=self.t0 + 10000.0,
            scheduled_arrival_time=self.t0 + 10000.0,
        )
        inv2 = InvestigatorFeedback(
            transaction_id="tx_2",
            investigator_id="inv_2",
            verdict="FRAUD",
            confidence=0.95,
            reviewed_at=self.t0 + 5000.0,
            scheduled_arrival_time=self.t0 + 5000.0,
        )
        cb1 = ChargebackFeedback(
            transaction_id="tx_3",
            chargeback_id="cb_1",
            reason_code="10.4",
            chargeback_amount=100.0,
            received_at=self.t0 + 8000.0,
            scheduled_arrival_time=self.t0 + 8000.0,
        )

        self.queue.enqueue_investigator(inv1)
        self.queue.enqueue_investigator(inv2)
        self.queue.enqueue_chargeback(cb1)

        self.assertEqual(self.queue.pending_count(), 3)
        self.assertEqual(self.queue.peek_next_arrival_time(), self.t0 + 5000.0)

        # Step to t0 + 6000: only tx_2 should be emitted
        invs, cbs = self.queue.poll(self.t0 + 6000.0)
        self.assertEqual(len(invs), 1)
        self.assertEqual(invs[0].transaction_id, "tx_2")
        self.assertEqual(len(cbs), 0)

        # Step to t0 + 9000: cb1 emitted
        invs, cbs = self.queue.poll(self.t0 + 9000.0)
        self.assertEqual(len(invs), 0)
        self.assertEqual(len(cbs), 1)
        self.assertEqual(cbs[0].transaction_id, "tx_3")

        # Step to t0 + 12000: inv1 emitted
        invs, cbs = self.queue.poll(self.t0 + 12000.0)
        self.assertEqual(len(invs), 1)
        self.assertEqual(invs[0].transaction_id, "tx_1")
        self.assertEqual(self.queue.pending_count(), 0)

    def test_queue_stats_and_clear(self) -> None:
        """Stats reporting and clear resets work properly."""
        self.queue.schedule_investigator_feedback("tx_a", self.t0, False, delay_seconds=100.0)
        self.queue.schedule_chargeback_feedback("tx_b", self.t0, 50.0, delay_seconds=200.0)

        stats = self.queue.stats()
        self.assertEqual(stats["total_enqueued"], 2)
        self.assertEqual(stats["pending_investigators"], 1)
        self.assertEqual(stats["pending_chargebacks"], 1)
        self.assertEqual(stats["pending_total"], 2)

        self.queue.clear()
        self.assertEqual(self.queue.pending_count(), 0)
        self.assertIsNone(self.queue.peek_next_arrival_time())


class TestWorkingMemory(unittest.TestCase):
    """Unit tests for Working Memory active window tracking and state updates."""

    def setUp(self) -> None:
        self.wm = WorkingMemory(window_sec=30 * 86400.0)
        self.t0 = 1729080000.0

    def test_transaction_ingestion_provisional_state(self) -> None:
        """New transactions start in PROVISIONAL_UNLABELED state with effective_label None."""
        event = {
            "transaction_id": "tx_wm_01",
            "customer_id": "cust_100",
            "card_id": "card_200",
            "amount": 149.99,
            "timestamp": self.t0,
        }
        fv = np.ones(16, dtype=np.float32)
        rec = self.wm.add_transaction(event, feature_vector=fv, risk_score=0.12)

        self.assertEqual(rec.transaction_id, "tx_wm_01")
        self.assertEqual(rec.state, TransactionLabelState.PROVISIONAL_UNLABELED)
        self.assertIsNone(rec.effective_label)
        self.assertFalse(rec.is_mature)
        self.assertAlmostEqual(rec.amount, 149.99)
        self.assertEqual(self.wm.size(), 1)
        self.assertTrue(self.wm.contains("tx_wm_01"))

    def test_investigator_review_flagged_and_cleared(self) -> None:
        """Investigator review updates state to INVESTIGATOR_FLAGGED or INVESTIGATOR_CLEARED."""
        # Tx 1: Flagged as fraud
        self.wm.add_transaction({"transaction_id": "tx_flag", "timestamp": self.t0, "amount": 100.0})
        inv_fraud = InvestigatorFeedback(
            transaction_id="tx_flag",
            investigator_id="inv_01",
            verdict="FRAUD",
            confidence=0.96,
            reviewed_at=self.t0 + 3600.0,
            scheduled_arrival_time=self.t0 + 3600.0,
        )
        updated = self.wm.apply_investigator_feedback(inv_fraud)
        self.assertTrue(updated)
        rec_flag = self.wm.get("tx_flag")
        assert rec_flag is not None
        self.assertEqual(rec_flag.state, TransactionLabelState.INVESTIGATOR_FLAGGED)
        self.assertEqual(rec_flag.effective_label, 1)

        # Tx 2: Cleared as legitimate
        self.wm.add_transaction({"transaction_id": "tx_clear", "timestamp": self.t0, "amount": 50.0})
        inv_legit = InvestigatorFeedback(
            transaction_id="tx_clear",
            investigator_id="inv_02",
            verdict="LEGITIMATE",
            confidence=0.92,
            reviewed_at=self.t0 + 7200.0,
            scheduled_arrival_time=self.t0 + 7200.0,
        )
        updated_legit = self.wm.apply_investigator_feedback(inv_legit)
        self.assertTrue(updated_legit)
        rec_clear = self.wm.get("tx_clear")
        assert rec_clear is not None
        self.assertEqual(rec_clear.state, TransactionLabelState.INVESTIGATOR_CLEARED)
        self.assertEqual(rec_clear.effective_label, 0)

        # Check state counts breakdown
        counts = self.wm.state_counts()
        self.assertEqual(counts[TransactionLabelState.INVESTIGATOR_FLAGGED.value], 1)
        self.assertEqual(counts[TransactionLabelState.INVESTIGATOR_CLEARED.value], 1)


class TestLabelPrecedenceAndReconciliation(unittest.TestCase):
    """Rigorous tests for label precedence: chargeback strictly overrides investigator clearance."""

    def setUp(self) -> None:
        self.wm = WorkingMemory()
        self.t0 = 1729080000.0

    def test_chargeback_strictly_overrides_investigator_clearance(self) -> None:
        """A chargeback dispute (y=1) strictly overrides a prior investigator clearance (y=0)."""
        tx_id = "tx_precedence_01"
        self.wm.add_transaction({"transaction_id": tx_id, "timestamp": self.t0, "amount": 320.0})

        # Step 1: Fast human review at t0 + 2h marks it LEGITIMATE (human false negative)
        inv_clear = InvestigatorFeedback(
            transaction_id=tx_id,
            investigator_id="inv_human_error",
            verdict="LEGITIMATE",
            confidence=0.88,
            reviewed_at=self.t0 + 7200.0,
            scheduled_arrival_time=self.t0 + 7200.0,
        )
        self.wm.apply_investigator_feedback(inv_clear)
        rec = self.wm.get(tx_id)
        assert rec is not None
        self.assertEqual(rec.state, TransactionLabelState.INVESTIGATOR_CLEARED)
        self.assertEqual(rec.effective_label, 0)

        # Step 2: 21 days later, cardholder files bank dispute: chargeback arrives
        cb_dispute = ChargebackFeedback(
            transaction_id=tx_id,
            chargeback_id="cb_precedence_99",
            reason_code="10.4_FRAUD_CARD_ABSENT",
            chargeback_amount=320.0,
            received_at=self.t0 + 21.0 * 86400.0,
            scheduled_arrival_time=self.t0 + 21.0 * 86400.0,
            is_fraud=True,
        )
        overridden = self.wm.apply_chargeback_feedback(cb_dispute)
        self.assertTrue(overridden)

        # State MUST be CHARGEBACK_CONFIRMED and effective_label MUST be 1
        self.assertEqual(rec.state, TransactionLabelState.CHARGEBACK_CONFIRMED)
        self.assertEqual(rec.effective_label, 1)
        self.assertTrue(rec.is_mature)

    def test_investigator_cannot_downgrade_confirmed_chargeback(self) -> None:
        """Late investigator clearance cannot downgrade an already confirmed chargeback."""
        tx_id = "tx_precedence_02"
        self.wm.add_transaction({"transaction_id": tx_id, "timestamp": self.t0, "amount": 180.0})

        # Chargeback arrives first
        cb = ChargebackFeedback(
            transaction_id=tx_id,
            chargeback_id="cb_first",
            reason_code="10.4",
            chargeback_amount=180.0,
            received_at=self.t0 + 15.0 * 86400.0,
            scheduled_arrival_time=self.t0 + 15.0 * 86400.0,
            is_fraud=True,
        )
        self.wm.apply_chargeback_feedback(cb)
        rec = self.wm.get(tx_id)
        assert rec is not None
        self.assertEqual(rec.state, TransactionLabelState.CHARGEBACK_CONFIRMED)
        self.assertEqual(rec.effective_label, 1)

        # Late investigator review tries to clear it
        inv_late = InvestigatorFeedback(
            transaction_id=tx_id,
            investigator_id="inv_late",
            verdict="LEGITIMATE",
            confidence=0.95,
            reviewed_at=self.t0 + 16.0 * 86400.0,
            scheduled_arrival_time=self.t0 + 16.0 * 86400.0,
        )
        res = self.wm.apply_investigator_feedback(inv_late)
        self.assertFalse(res, "Clearance should not modify CHARGEBACK_CONFIRMED state")
        self.assertEqual(rec.state, TransactionLabelState.CHARGEBACK_CONFIRMED)
        self.assertEqual(rec.effective_label, 1)

    def test_reconciliation_in_consolidated_memory(self) -> None:
        """Late chargeback arriving after transaction already consolidated to y=0 updates label to y=1."""
        cm = ConsolidatedMemory()
        tx_id = "tx_late_dispute"
        rec = MemoryRecord(
            transaction_id=tx_id,
            customer_id="cust_1",
            card_id="card_1",
            amount=500.0,
            timestamp=self.t0,
            state=TransactionLabelState.MATURED_LEGITIMATE,
            effective_label=0,
            is_mature=True,
            matured_at=self.t0 + 30.0 * 86400.0,
        )
        cm.add_record(rec)
        self.assertEqual(cm.legitimate_count(), 1)
        self.assertEqual(cm.fraud_count(), 0)

        # Dispute arrives at day 35
        cb = ChargebackFeedback(
            transaction_id=tx_id,
            chargeback_id="cb_day35",
            reason_code="10.4",
            chargeback_amount=500.0,
            received_at=self.t0 + 35.0 * 86400.0,
            scheduled_arrival_time=self.t0 + 35.0 * 86400.0,
            is_fraud=True,
        )
        cm.apply_chargeback_feedback(cb)
        self.assertEqual(cm.legitimate_count(), 0)
        self.assertEqual(cm.fraud_count(), 1)
        self.assertEqual(rec.state, TransactionLabelState.CHARGEBACK_CONFIRMED)
        self.assertEqual(rec.effective_label, 1)


class TestConsolidatedMemory(unittest.TestCase):
    """Unit tests for Consolidated Memory replay buffer and sampling."""

    def setUp(self) -> None:
        self.cm = ConsolidatedMemory(max_capacity=1000)
        self.t0 = 1729080000.0

    def test_add_and_array_conversion(self) -> None:
        """Adding records and converting to NumPy float32 arrays preserves exact values."""
        for i in range(10):
            is_fraud = (i == 0)  # 1 fraud, 9 legit
            rec = MemoryRecord(
                transaction_id=f"tx_{i}",
                customer_id=f"c_{i}",
                card_id=f"card_{i}",
                amount=float(50.0 + i * 10),
                timestamp=self.t0 + i,
                feature_vector=np.full(16, fill_value=float(i), dtype=np.float32),
                state=TransactionLabelState.CHARGEBACK_CONFIRMED if is_fraud else TransactionLabelState.MATURED_LEGITIMATE,
                effective_label=1 if is_fraud else 0,
                is_mature=True,
            )
            self.cm.add_record(rec)

        self.assertEqual(self.cm.size(), 10)
        self.assertEqual(self.cm.fraud_count(), 1)
        self.assertEqual(self.cm.legitimate_count(), 9)

        X, y, amounts = self.cm.to_numpy_arrays(feature_dim=16)
        self.assertEqual(X.shape, (10, 16))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.dtype, np.int32)
        self.assertEqual(y.sum(), 1)
        self.assertAlmostEqual(amounts[0], 50.0)

    def test_stratified_batch_sampling(self) -> None:
        """Stratified sampling accurately produces specified fraud proportion."""
        # Populate 900 legitimate and 100 fraud records
        for i in range(1000):
            is_fraud = (i < 100)
            rec = MemoryRecord(
                transaction_id=f"tx_{i}",
                customer_id=f"c_{i}",
                card_id=f"card_{i}",
                amount=100.0,
                timestamp=self.t0 + i,
                effective_label=1 if is_fraud else 0,
                state=TransactionLabelState.CHARGEBACK_CONFIRMED if is_fraud else TransactionLabelState.MATURED_LEGITIMATE,
                is_mature=True,
            )
            self.cm.add_record(rec)

        # Sample batch of 100 with target fraud_ratio = 0.20
        rng = random.Random(42)
        batch = self.cm.sample_batch(batch_size=100, fraud_ratio=0.20, rng=rng)
        self.assertEqual(len(batch), 100)
        frauds_sampled = sum(1 for r in batch if r.effective_label == 1)
        self.assertEqual(frauds_sampled, 20)

    def test_capacity_management_preserves_rare_fraud(self) -> None:
        """When capacity is exceeded, legitimate records are evicted before rare fraud records."""
        cm_small = ConsolidatedMemory(max_capacity=5)
        # Add 2 frauds
        for i in range(2):
            cm_small.add_record(MemoryRecord(
                transaction_id=f"f_{i}",
                customer_id="c",
                card_id="c",
                amount=10.0,
                timestamp=self.t0,
                effective_label=1,
                state=TransactionLabelState.CHARGEBACK_CONFIRMED,
                is_mature=True,
            ))
        # Add 4 legit records (total 6 > capacity 5)
        for i in range(4):
            cm_small.add_record(MemoryRecord(
                transaction_id=f"l_{i}",
                customer_id="c",
                card_id="c",
                amount=10.0,
                timestamp=self.t0,
                effective_label=0,
                state=TransactionLabelState.MATURED_LEGITIMATE,
                is_mature=True,
            ))

        self.assertEqual(cm_small.size(), 5)
        self.assertEqual(cm_small.fraud_count(), 2, "Fraud records must not be evicted prematurely")
        self.assertEqual(cm_small.legitimate_count(), 3)


class TestDelayedFeedbackDebiaser(unittest.TestCase):
    """Tests for delayed feedback debiaser eliminating 50% FN contamination."""

    def setUp(self) -> None:
        self.debiaser = DelayedFeedbackDebiaser()
        self.t0 = 1729080000.0

    def test_naive_vs_debiased_fn_contamination(self) -> None:
        """Verify naive label truncation creates ~50% FN contamination, while debiaser achieves 0%."""
        records: List[MemoryRecord] = []
        ground_truth: Dict[str, int] = {}
        horizon = 30.0 * 86400.0
        current_time = self.t0 + 25.0 * 86400.0  # Day 25

        # Create 100 transactions over the last 25 days
        # 10 are true frauds with chargeback arrival simulated between 14 and 30 days
        for i in range(100):
            tx_id = f"tx_eval_{i}"
            is_fraud = (i < 10)
            ground_truth[tx_id] = 1 if is_fraud else 0
            tx_time = self.t0 + (i * 0.25) * 86400.0  # Distributed across days 0 to 25

            rec = MemoryRecord(
                transaction_id=tx_id,
                customer_id="c",
                card_id="c",
                amount=100.0,
                timestamp=tx_time,
                state=TransactionLabelState.PROVISIONAL_UNLABELED,
                effective_label=None,
            )

            if is_fraud:
                # 5 frauds had early chargebacks that arrived before current_time
                if i < 5:
                    rec.state = TransactionLabelState.CHARGEBACK_CONFIRMED
                    rec.effective_label = 1
                    rec.is_mature = True
                # 5 frauds have chargebacks scheduled AFTER current_time (pending dispute)
                else:
                    rec.state = TransactionLabelState.PROVISIONAL_UNLABELED
                    rec.effective_label = None

            records.append(rec)

        metrics = self.debiaser.measure_contamination_rates(
            records=records,
            current_sim_time=current_time,
            ground_truth_labels=ground_truth,
            maturity_horizon_sec=horizon,
        )

        self.assertEqual(metrics["true_frauds_in_window"], 10.0)
        self.assertEqual(metrics["naive_mislabeled_frauds"], 5.0)
        self.assertAlmostEqual(metrics["naive_fn_contamination_rate"], 0.50, delta=0.01)
        self.assertEqual(metrics["debiased_mislabeled_frauds"], 0.0)
        self.assertEqual(metrics["debiased_fn_contamination_rate"], 0.0)

    def test_compute_debiasing_weights_masking(self) -> None:
        """Pending unreviewed transactions receive sample weight 0.0 and mask False."""
        horizon = 30.0 * 86400.0
        current_time = self.t0 + 20.0 * 86400.0

        r_mature_cb = MemoryRecord(
            transaction_id="r1", customer_id="c", card_id="c", amount=10.0,
            timestamp=self.t0, state=TransactionLabelState.CHARGEBACK_CONFIRMED,
            effective_label=1, is_mature=True,
        )
        r_mature_legit = MemoryRecord(
            transaction_id="r2", customer_id="c", card_id="c", amount=10.0,
            timestamp=self.t0 - 35.0 * 86400.0, state=TransactionLabelState.MATURED_LEGITIMATE,
            effective_label=0, is_mature=True,
        )
        r_pending_unreviewed = MemoryRecord(
            transaction_id="r3", customer_id="c", card_id="c", amount=10.0,
            timestamp=self.t0 + 10.0 * 86400.0, state=TransactionLabelState.PROVISIONAL_UNLABELED,
            effective_label=None, is_mature=False,
        )

        res = self.debiaser.compute_debiasing_weights(
            records=[r_mature_cb, r_mature_legit, r_pending_unreviewed],
            current_sim_time=current_time,
            maturity_horizon_sec=horizon,
        )

        labels = res["labels"]
        weights = res["weights"]
        mask = res["mask"]

        # Confirmed chargeback
        self.assertEqual(labels[0], 1)
        self.assertEqual(weights[0], 1.0)
        self.assertTrue(mask[0])

        # Mature legitimate
        self.assertEqual(labels[1], 0)
        self.assertEqual(weights[1], 1.0)
        self.assertTrue(mask[1])

        # Pending unreviewed: weight=0, mask=False, preventing FN contamination
        self.assertEqual(weights[2], 0.0)
        self.assertFalse(mask[2])


class TestDualMemoryManagerIntegration(unittest.TestCase):
    """End-to-end integration tests for DualMemoryManager lifecycle."""

    def setUp(self) -> None:
        self.t0 = 1729080000.0
        self.manager = DualMemoryManager(maturity_horizon_sec=30.0 * 86400.0)

    def test_end_to_end_streaming_and_maturation(self) -> None:
        """Transactions stream in, receive asynchronous reviews, mature, and consolidate."""
        # Day 0: Ingest 3 transactions
        self.manager.ingest_transaction(
            {"transaction_id": "tx_normal_01", "timestamp": self.t0, "amount": 45.0},
            feature_vector=np.ones(16, dtype=np.float32),
            risk_score=0.01,
        )
        self.manager.ingest_transaction(
            {"transaction_id": "tx_fraud_reviewed", "timestamp": self.t0, "amount": 890.0},
            feature_vector=np.ones(16, dtype=np.float32),
            risk_score=0.85,
        )
        self.manager.ingest_transaction(
            {"transaction_id": "tx_fraud_delayed", "timestamp": self.t0, "amount": 1200.0},
            feature_vector=np.ones(16, dtype=np.float32),
            risk_score=0.60,
        )

        # Schedule reviews:
        # tx_fraud_reviewed gets fast review at t0 + 2h
        self.manager.schedule_investigator_review(
            transaction_id="tx_fraud_reviewed",
            tx_timestamp=self.t0,
            is_fraud=True,
            delay_seconds=7200.0,
            verdict="FRAUD",
            confidence=0.97,
        )

        # tx_fraud_delayed gets chargeback at t0 + 20 days
        self.manager.schedule_chargeback(
            transaction_id="tx_fraud_delayed",
            tx_timestamp=self.t0,
            amount=1200.0,
            is_fraud=True,
            delay_seconds=20.0 * 86400.0,
        )

        self.assertEqual(self.manager.working_memory.size(), 3)
        self.assertEqual(self.manager.consolidated_memory.size(), 0)

        # Step to Day 0.5 (t0 + 12h): Investigator review should be applied
        step_12h = self.manager.step(self.t0 + 43200.0)
        self.assertEqual(step_12h["investigator_reviews_applied"], 1)
        self.assertEqual(step_12h["chargebacks_applied"], 0)
        rec_rev = self.manager.working_memory.get("tx_fraud_reviewed")
        assert rec_rev is not None
        self.assertEqual(rec_rev.state, TransactionLabelState.INVESTIGATOR_FLAGGED)

        # Step to Day 21 (t0 + 21 days): Chargeback arrives
        step_21d = self.manager.step(self.t0 + 21.0 * 86400.0)
        self.assertEqual(step_21d["chargebacks_applied"], 1)
        # Confirmed chargebacks immediately enter consolidated memory
        self.assertEqual(self.manager.consolidated_memory.fraud_count(), 1)

        # Step to Day 32 (t0 + 32 days): Transaction window matures
        step_32d = self.manager.step(self.t0 + 32.0 * 86400.0)
        # Normal transaction without dispute matures to legitimate
        self.assertGreaterEqual(self.manager.consolidated_memory.legitimate_count(), 1)
        rec_norm = self.manager.consolidated_memory.get("tx_normal_01")
        assert rec_norm is not None
        self.assertEqual(rec_norm.state, TransactionLabelState.MATURED_LEGITIMATE)
        self.assertEqual(rec_norm.effective_label, 0)


if __name__ == "__main__":
    unittest.main()
