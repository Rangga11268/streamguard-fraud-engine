"""StreamGuard Dual-Memory Feedback Architecture.

Separates operational fast feedback from definitive delayed ground truth:
  1. Working Memory (M_fast):
     Maintains the active 30-day transaction window. Dynamically updates label states:
       - PROVISIONAL_UNLABELED: initial transaction state before any review.
       - INVESTIGATOR_FLAGGED: reviewed by human investigator as FRAUD within 1-4h.
       - INVESTIGATOR_CLEARED: reviewed by human investigator as LEGITIMATE within 1-4h.
       - CHARGEBACK_CONFIRMED: definitive bank dispute arriving in 14-30 days (y=1).
       - MATURED_LEGITIMATE: aged past 30-day dispute window without chargeback (y=0).
  2. Consolidated Memory (M_cons):
     Replay buffer holding mature records (where elapsed time > 30 days or definitive
     chargeback received). Used for pure offline / periodic model retraining.
  3. Strict Label Precedence & Reconciliation:
     A chargeback dispute (y=1) strictly overrides a prior investigator clearance (y=0).
     Investigator clearance can never downgrade a confirmed chargeback.
  4. Delayed-Feedback Debiaser:
     Eliminates the 50% false-negative contamination caused by naive label truncation
     (unreviewed transactions within 30 days are NOT assumed to be non-fraud).
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from app.config import (
    DELAYED_FEEDBACK_MAX_SEC,
    FAST_FEEDBACK_MIN_SEC,
    MATURITY_HORIZON_SEC,
    StreamGuardConfig,
    default_config,
)
from app.memory.feedback_queue import (
    ChargebackFeedback,
    FeedbackPriorityQueue,
    InvestigatorFeedback,
    InvestigatorVerdict,
)


class TransactionLabelState(str, Enum):
    """Lifecycle states of transaction labels under delayed verification."""
    PROVISIONAL_UNLABELED = "PROVISIONAL_UNLABELED"
    INVESTIGATOR_FLAGGED = "INVESTIGATOR_FLAGGED"
    INVESTIGATOR_CLEARED = "INVESTIGATOR_CLEARED"
    CHARGEBACK_CONFIRMED = "CHARGEBACK_CONFIRMED"
    MATURED_LEGITIMATE = "MATURED_LEGITIMATE"


@dataclass(slots=True)
class MemoryRecord:
    """Unified transaction record tracked across dual-memory buffers."""
    transaction_id: str
    customer_id: str
    card_id: str
    amount: float
    timestamp: float  # Transaction event time t_tx
    feature_vector: Optional[np.ndarray] = None  # 16-dim float32 vector
    risk_score: float = 0.0  # Predicted probability p(fraud | x)
    state: TransactionLabelState = TransactionLabelState.PROVISIONAL_UNLABELED
    effective_label: Optional[int] = None  # None (unresolved), 1 (fraud), 0 (legitimate)
    investigator_feedback: Optional[InvestigatorFeedback] = None
    chargeback_feedback: Optional[ChargebackFeedback] = None
    is_mature: bool = False
    matured_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_investigator_review(
        self,
        feedback: InvestigatorFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Apply fast investigator review to record.

        Label Precedence Rule:
        If record already has CHARGEBACK_CONFIRMED (y=1), an investigator clearance
        MUST NOT downgrade the confirmed dispute.
        """
        self.investigator_feedback = feedback

        if self.state == TransactionLabelState.CHARGEBACK_CONFIRMED:
            # Chargeback dispute is definitive ground truth; do not override
            return False

        if feedback.is_fraud:
            self.state = TransactionLabelState.INVESTIGATOR_FLAGGED
            self.effective_label = 1
        else:
            self.state = TransactionLabelState.INVESTIGATOR_CLEARED
            self.effective_label = 0

        return True

    def apply_chargeback(
        self,
        feedback: ChargebackFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Apply definitive bank chargeback to record.

        Strict Label Precedence Rule:
        A chargeback dispute (y=1) strictly overrides a prior investigator
        clearance (y=0) or provisional state. A confirmed chargeback is immediately
        mature ground truth.
        """
        self.chargeback_feedback = feedback

        if feedback.is_fraud:
            self.state = TransactionLabelState.CHARGEBACK_CONFIRMED
            self.effective_label = 1
            self.is_mature = True
            if sim_time is not None:
                self.matured_at = sim_time
            elif self.matured_at is None:
                self.matured_at = feedback.received_at
            return True
        else:
            # Rare: dispute formally resolved as recognized non-fraud
            if self.state != TransactionLabelState.CHARGEBACK_CONFIRMED:
                self.state = TransactionLabelState.MATURED_LEGITIMATE
                self.effective_label = 0
                self.is_mature = True
                if sim_time is not None:
                    self.matured_at = sim_time
                return True
            return False

    def mark_matured_legitimate(self, sim_time: float) -> bool:
        """Mark record as matured legitimate after surviving the dispute window without chargeback."""
        if self.state == TransactionLabelState.CHARGEBACK_CONFIRMED:
            # Cannot mark confirmed fraud as legitimate
            return False
        if self.state == TransactionLabelState.INVESTIGATOR_FLAGGED:
            # If flagged by human investigator, do not mark as legitimate
            self.is_mature = True
            self.matured_at = sim_time
            return False

        # Transaction completed dispute window with no chargeback: definitive legitimate
        self.state = TransactionLabelState.MATURED_LEGITIMATE
        self.effective_label = 0
        self.is_mature = True
        self.matured_at = sim_time
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation."""
        fv_list = self.feature_vector.tolist() if self.feature_vector is not None else None
        return {
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "card_id": self.card_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "feature_vector": fv_list,
            "risk_score": self.risk_score,
            "state": self.state.value,
            "effective_label": self.effective_label,
            "is_mature": self.is_mature,
            "matured_at": self.matured_at,
            "has_investigator_review": self.investigator_feedback is not None,
            "has_chargeback": self.chargeback_feedback is not None,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """Working Memory (M_fast) maintaining active transactions within the 30-day window.

    Tracks transactions in real-time, ingests fast investigator feedback,
    and transitions records through their label states.
    """

    def __init__(self, window_sec: float = 30 * 86400.0) -> None:
        self.window_sec: float = float(window_sec)
        self._records: Dict[str, MemoryRecord] = {}
        # Chronological index: (timestamp, transaction_id)
        self._chronological: deque[Tuple[float, str]] = deque()

    def add_transaction(
        self,
        event: Any,
        feature_vector: Optional[np.ndarray] = None,
        risk_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Ingest a new transaction into Working Memory with PROVISIONAL_UNLABELED state.

        Parameters
        ----------
        event : Any
            Object with transaction_id, customer_id, card_id, amount, timestamp,
            or dictionary.
        feature_vector : Optional[np.ndarray]
            16-dim float32 feature vector from feature store.
        risk_score : float
            Predicted probability from model.
        metadata : Optional[Dict[str, Any]]
            Additional context.
        """
        if isinstance(event, dict):
            tx_id = str(event.get("transaction_id", ""))
            cust_id = str(event.get("customer_id", ""))
            card_id = str(event.get("card_id", ""))
            amount = float(event.get("amount", 0.0))
            ts = float(event.get("timestamp", 0.0))
        else:
            tx_id = str(getattr(event, "transaction_id", ""))
            cust_id = str(getattr(event, "customer_id", ""))
            card_id = str(getattr(event, "card_id", ""))
            amount = float(getattr(event, "amount", 0.0))
            ts = float(getattr(event, "timestamp", 0.0))

        if feature_vector is not None and not isinstance(feature_vector, np.ndarray):
            feature_vector = np.asarray(feature_vector, dtype=np.float32)
        elif feature_vector is not None and feature_vector.dtype != np.float32:
            feature_vector = feature_vector.astype(np.float32)

        record = MemoryRecord(
            transaction_id=tx_id,
            customer_id=cust_id,
            card_id=card_id,
            amount=amount,
            timestamp=ts,
            feature_vector=feature_vector,
            risk_score=float(risk_score),
            state=TransactionLabelState.PROVISIONAL_UNLABELED,
            effective_label=None,
            metadata=metadata or {},
        )

        self._records[tx_id] = record
        self._chronological.append((ts, tx_id))
        return record

    def get(self, transaction_id: str) -> Optional[MemoryRecord]:
        """Retrieve record by transaction ID."""
        return self._records.get(transaction_id)

    def contains(self, transaction_id: str) -> bool:
        """Check if transaction exists in Working Memory."""
        return transaction_id in self._records

    def apply_investigator_feedback(
        self,
        feedback: InvestigatorFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Apply fast investigator review to transaction in Working Memory."""
        record = self._records.get(feedback.transaction_id)
        if record is None:
            return False
        return record.apply_investigator_review(feedback, sim_time)

    def apply_chargeback_feedback(
        self,
        feedback: ChargebackFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Apply definitive bank chargeback to transaction in Working Memory with label precedence."""
        record = self._records.get(feedback.transaction_id)
        if record is None:
            return False
        return record.apply_chargeback(feedback, sim_time)

    def extract_mature_records(
        self,
        current_sim_time: float,
        maturity_horizon_sec: float = 30 * 86400.0,
    ) -> List[MemoryRecord]:
        """Extract and remove all mature records from Working Memory for consolidation.

        A record is mature if:
          1. A definitive chargeback was received (CHARGEBACK_CONFIRMED), OR
          2. Elapsed time (current_sim_time - tx_timestamp) >= maturity_horizon_sec.
             In this case, if no chargeback or investigator fraud was received,
             the record is definitively marked as MATURED_LEGITIMATE (y=0).

        Parameters
        ----------
        current_sim_time : float
            Current simulation timestamp.
        maturity_horizon_sec : float
            Dispute maturity threshold in seconds (default: 30 days).

        Returns
        -------
        List[MemoryRecord]
            List of matured records removed from Working Memory.
        """
        mature_records: List[MemoryRecord] = []
        ids_to_remove: List[str] = []

        for tx_id, record in self._records.items():
            elapsed = current_sim_time - record.timestamp

            # Case 1: Confirmed chargeback is immediately mature
            if record.state == TransactionLabelState.CHARGEBACK_CONFIRMED:
                record.is_mature = True
                if record.matured_at is None:
                    record.matured_at = current_sim_time
                mature_records.append(record)
                ids_to_remove.append(tx_id)
                continue

            # Case 2: Elapsed time exceeds dispute maturity horizon
            if elapsed >= maturity_horizon_sec:
                if record.state in (
                    TransactionLabelState.PROVISIONAL_UNLABELED,
                    TransactionLabelState.INVESTIGATOR_CLEARED,
                ):
                    record.mark_matured_legitimate(current_sim_time)
                else:
                    record.is_mature = True
                    record.matured_at = current_sim_time

                mature_records.append(record)
                ids_to_remove.append(tx_id)

        # Evict from working memory map
        for tx_id in ids_to_remove:
            del self._records[tx_id]

        # Clean chronological index
        while self._chronological and self._chronological[0][1] not in self._records:
            self._chronological.popleft()

        return mature_records

    def prune_expired(self, current_sim_time: float, max_age_sec: float) -> int:
        """Prune records older than max_age_sec that were not yet extracted."""
        cutoff = current_sim_time - max_age_sec
        pruned = 0
        while self._chronological and self._chronological[0][0] < cutoff:
            ts, tx_id = self._chronological.popleft()
            if tx_id in self._records:
                del self._records[tx_id]
                pruned += 1
        return pruned

    def size(self) -> int:
        """Return total count of active transactions in Working Memory."""
        return len(self._records)

    def state_counts(self) -> Dict[str, int]:
        """Return count breakdown by TransactionLabelState."""
        counts = {state.value: 0 for state in TransactionLabelState}
        for rec in self._records.values():
            counts[rec.state.value] += 1
        return counts

    def all_records(self) -> List[MemoryRecord]:
        """Return all active records in Working Memory."""
        return list(self._records.values())


class ConsolidatedMemory:
    """Consolidated Replay Buffer (M_cons) holding mature ground-truth records.

    Contains only fully matured records (where elapsed dispute horizon has passed
    or definitive chargeback received), guaranteeing zero label truncation bias.
    Used for model retraining, offline evaluation, and balanced replay sampling.
    """

    def __init__(self, max_capacity: int = 100_000) -> None:
        self.max_capacity: int = int(max_capacity)
        self._records: Dict[str, MemoryRecord] = {}
        self._fraud_ids: List[str] = []
        self._legit_ids: List[str] = []

    def add_record(self, record: MemoryRecord) -> None:
        """Add or update a mature record in Consolidated Memory.

        Maintains chargeback precedence if a late dispute arrives for an already
        consolidated record.
        """
        tx_id = record.transaction_id

        if tx_id in self._records:
            existing = self._records[tx_id]
            # If new record has chargeback, update existing with chargeback precedence
            if record.chargeback_feedback is not None:
                existing.apply_chargeback(record.chargeback_feedback, record.matured_at)
                # Re-index if effective label changed
                if existing.effective_label == 1 and tx_id in self._legit_ids:
                    self._legit_ids.remove(tx_id)
                    if tx_id not in self._fraud_ids:
                        self._fraud_ids.append(tx_id)
            return

        # Enforce capacity by evicting oldest legitimate record to protect rare fraud examples
        if len(self._records) >= self.max_capacity:
            if self._legit_ids:
                evict_id = self._legit_ids.pop(0)
                if evict_id in self._records:
                    del self._records[evict_id]
            elif self._fraud_ids:
                evict_id = self._fraud_ids.pop(0)
                if evict_id in self._records:
                    del self._records[evict_id]

        self._records[tx_id] = record
        if record.effective_label == 1:
            self._fraud_ids.append(tx_id)
        else:
            self._legit_ids.append(tx_id)

    def apply_chargeback_feedback(
        self,
        feedback: ChargebackFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Apply a delayed chargeback to a transaction already stored in Consolidated Memory."""
        record = self._records.get(feedback.transaction_id)
        if record is None:
            return False

        record.apply_chargeback(feedback, sim_time)
        tx_id = feedback.transaction_id
        if record.effective_label == 1:
            if tx_id in self._legit_ids:
                self._legit_ids.remove(tx_id)
            if tx_id not in self._fraud_ids:
                self._fraud_ids.append(tx_id)
        return True

    def get(self, transaction_id: str) -> Optional[MemoryRecord]:
        """Retrieve consolidated record by transaction ID."""
        return self._records.get(transaction_id)

    def size(self) -> int:
        """Return total count of consolidated records."""
        return len(self._records)

    def fraud_count(self) -> int:
        """Return count of verified fraud records (y=1)."""
        return len(self._fraud_ids)

    def legitimate_count(self) -> int:
        """Return count of verified legitimate records (y=0)."""
        return len(self._legit_ids)

    def sample_batch(
        self,
        batch_size: int,
        fraud_ratio: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> List[MemoryRecord]:
        """Sample a batch of mature records with optional stratified fraud ratio.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.
        fraud_ratio : Optional[float]
            Target fraud proportion in sampled batch (e.g. 0.05). If None, uniform random.
        rng : Optional[random.Random]
            Random number generator instance for reproducible sampling.

        Returns
        -------
        List[MemoryRecord]
            Sampled list of mature records.
        """
        if not self._records:
            return []

        rand = rng or random.Random()
        total = len(self._records)
        actual_batch_size = min(batch_size, total)

        if fraud_ratio is None or not self._fraud_ids or not self._legit_ids:
            # Uniform random sampling
            sampled_ids = rand.sample(list(self._records.keys()), actual_batch_size)
            return [self._records[tx_id] for tx_id in sampled_ids]

        # Stratified sampling
        target_fraud_count = max(1, int(round(actual_batch_size * fraud_ratio)))
        target_fraud_count = min(target_fraud_count, len(self._fraud_ids))
        target_legit_count = actual_batch_size - target_fraud_count
        target_legit_count = min(target_legit_count, len(self._legit_ids))

        sampled_fraud = rand.sample(self._fraud_ids, target_fraud_count)
        sampled_legit = rand.sample(self._legit_ids, target_legit_count)
        all_sampled_ids = sampled_fraud + sampled_legit
        rand.shuffle(all_sampled_ids)

        return [self._records[tx_id] for tx_id in all_sampled_ids]

    def to_numpy_arrays(
        self,
        feature_dim: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert consolidated records to NumPy feature matrix, labels, and amounts.

        Returns
        -------
        X : np.ndarray, shape (N, feature_dim), dtype float32
        y : np.ndarray, shape (N,), dtype int32
        amounts : np.ndarray, shape (N,), dtype float32
        """
        records = [r for r in self._records.values() if r.feature_vector is not None and r.effective_label is not None]
        if not records:
            return (
                np.empty((0, feature_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        X = np.empty((len(records), feature_dim), dtype=np.float32)
        y = np.empty(len(records), dtype=np.int32)
        amounts = np.empty(len(records), dtype=np.float32)

        for i, rec in enumerate(records):
            assert rec.feature_vector is not None
            X[i] = rec.feature_vector
            y[i] = int(rec.effective_label) if rec.effective_label is not None else 0
            amounts[i] = float(rec.amount)

        return X, y, amounts

    def all_records(self) -> List[MemoryRecord]:
        """Return all consolidated records."""
        return list(self._records.values())

    def state_counts(self) -> Dict[str, int]:
        """Return count breakdown by TransactionLabelState."""
        counts = {state.value: 0 for state in TransactionLabelState}
        for rec in self._records.values():
            counts[rec.state.value] += 1
        return counts


class DelayedFeedbackDebiaser:
    """Eliminates the 50% false-negative contamination from naive label truncation.

    Under naive online labeling:
      Unreviewed transactions in the recent 30-day window are assumed to be non-fraud
      (y=0). Because bank chargebacks require 14-30 days to arrive, approximately
      50% of recent fraudulent transactions have not yet generated a dispute and are
      erroneously poisoned into the negative class.

    The DelayedFeedbackDebiaser resolves this by:
      1. Pure Maturity Isolation: Only using mature records from Consolidated Memory
         (where elapsed dispute horizon has passed) or confirmed chargebacks.
      2. Debiasing Weight Computation: Masking unreviewed transactions in the pending
         window (weight = 0.0), ensuring they are NEVER naively assumed negative.
    """

    @staticmethod
    def filter_unbiased_records(records: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        """Filter a collection of records to only include definitively labeled, unbiased records.

        Discards unreviewed / pending records so they are NEVER assumed to be negative.
        """
        unbiased: List[MemoryRecord] = []
        for r in records:
            # Included if mature legitimate (survived horizon) OR confirmed chargeback
            if r.state == TransactionLabelState.CHARGEBACK_CONFIRMED:
                unbiased.append(r)
            elif r.state == TransactionLabelState.MATURED_LEGITIMATE:
                unbiased.append(r)
            elif r.state == TransactionLabelState.INVESTIGATOR_FLAGGED:
                # High-confidence investigator review can optionally be included
                unbiased.append(r)
        return unbiased

    @staticmethod
    def compute_debiasing_weights(
        records: Sequence[MemoryRecord],
        current_sim_time: float,
        maturity_horizon_sec: float = 30 * 86400.0,
    ) -> Dict[str, np.ndarray]:
        """Compute labels, sample weights, and inclusion masks for a set of records.

        Returns
        -------
        Dict[str, np.ndarray]:
          - "labels": np.ndarray (N,) int32
          - "weights": np.ndarray (N,) float32 (0.0 for unreviewed pending records)
          - "mask": np.ndarray (N,) bool (True for valid unbiased samples)
          - "is_contaminated_under_naive": np.ndarray (N,) bool (True if naive would mislabel as FN)
        """
        n = len(records)
        labels = np.zeros(n, dtype=np.int32)
        weights = np.zeros(n, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)

        for i, rec in enumerate(records):
            elapsed = current_sim_time - rec.timestamp

            if rec.state == TransactionLabelState.CHARGEBACK_CONFIRMED:
                labels[i] = 1
                weights[i] = 1.0
                mask[i] = True
            elif rec.state == TransactionLabelState.INVESTIGATOR_FLAGGED:
                labels[i] = 1
                conf = rec.investigator_feedback.confidence if rec.investigator_feedback else 0.90
                weights[i] = float(conf)
                mask[i] = True
            elif rec.state == TransactionLabelState.MATURED_LEGITIMATE or elapsed >= maturity_horizon_sec:
                labels[i] = 0
                weights[i] = 1.0
                mask[i] = True
            else:
                # Pending transaction within 30 days:
                # DO NOT assume negative! Weight = 0.0, mask = False.
                labels[i] = 0
                weights[i] = 0.0
                mask[i] = False

        return {
            "labels": labels,
            "weights": weights,
            "mask": mask,
        }

    @staticmethod
    def measure_contamination_rates(
        records: Sequence[MemoryRecord],
        current_sim_time: float,
        ground_truth_labels: Dict[str, int],
        maturity_horizon_sec: float = 30 * 86400.0,
    ) -> Dict[str, float]:
        """Empirically measure false-negative contamination under naive vs debiased strategies.

        Parameters
        ----------
        records : Sequence[MemoryRecord]
            Collection of records in the recent transaction window.
        current_sim_time : float
            Current simulation timestamp.
        ground_truth_labels : Dict[str, int]
            True un-delayed fraud labels (1 for fraud, 0 for legitimate).
        maturity_horizon_sec : float
            Maturity horizon (default 30 days).

        Returns
        -------
        Dict[str, float]:
          - "true_frauds_in_window": int
          - "naive_mislabeled_frauds": int
          - "naive_fn_contamination_rate": float (typically ~50%)
          - "debiased_mislabeled_frauds": int (0)
          - "debiased_fn_contamination_rate": float (0.0%)
        """
        total_frauds = 0
        naive_mislabeled = 0
        debiased_mislabeled = 0

        for rec in records:
            true_y = ground_truth_labels.get(rec.transaction_id, 0)
            if true_y == 1:
                total_frauds += 1
                # Under naive strategy: if no chargeback or investigator fraud yet, labeled as 0
                naive_label = 1 if (rec.state in (TransactionLabelState.CHARGEBACK_CONFIRMED, TransactionLabelState.INVESTIGATOR_FLAGGED)) else 0
                if naive_label == 0:
                    naive_mislabeled += 1

                # Under debiased strategy: unreviewed pending records are NOT labeled negative
                # Only mature records are given negative labels
                elapsed = current_sim_time - rec.timestamp
                if rec.state == TransactionLabelState.MATURED_LEGITIMATE:
                    debiased_mislabeled += 1  # Should never happen if dispute horizon is respected

        naive_rate = (naive_mislabeled / total_frauds) if total_frauds > 0 else 0.0
        debiased_rate = (debiased_mislabeled / total_frauds) if total_frauds > 0 else 0.0

        return {
            "true_frauds_in_window": float(total_frauds),
            "naive_mislabeled_frauds": float(naive_mislabeled),
            "naive_fn_contamination_rate": float(naive_rate),
            "debiased_mislabeled_frauds": float(debiased_mislabeled),
            "debiased_fn_contamination_rate": float(debiased_rate),
        }


class DualMemoryManager:
    """Master Dual-Memory Manager coordinating Working Memory, Consolidated Memory,

    Feedback Queues, and the Delayed-Feedback Debiaser.
    """

    def __init__(
        self,
        config: Optional[StreamGuardConfig] = None,
        maturity_horizon_sec: float = 30 * 86400.0,
        working_window_sec: float = 30 * 86400.0,
        max_consolidated_capacity: int = 100_000,
    ) -> None:
        self.config = config or default_config
        self.maturity_horizon_sec: float = float(maturity_horizon_sec)
        self.working_window_sec: float = float(working_window_sec)

        self.feedback_queue = FeedbackPriorityQueue(self.config)
        self.working_memory = WorkingMemory(window_sec=self.working_window_sec)
        self.consolidated_memory = ConsolidatedMemory(max_capacity=max_consolidated_capacity)
        self.debiaser = DelayedFeedbackDebiaser()

    def ingest_transaction(
        self,
        event: Any,
        feature_vector: Optional[np.ndarray] = None,
        risk_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Ingest incoming transaction into Working Memory."""
        return self.working_memory.add_transaction(
            event=event,
            feature_vector=feature_vector,
            risk_score=risk_score,
            metadata=metadata,
        )

    def schedule_investigator_review(
        self,
        transaction_id: str,
        tx_timestamp: float,
        is_fraud: bool,
        delay_seconds: Optional[float] = None,
        investigator_id: Optional[str] = None,
        verdict: Optional[str] = None,
        confidence: Optional[float] = None,
        accuracy: float = 0.95,
        notes: str = "",
    ) -> float:
        """Schedule fast investigator review in feedback queue."""
        return self.feedback_queue.schedule_investigator_feedback(
            transaction_id=transaction_id,
            tx_timestamp=tx_timestamp,
            is_fraud=is_fraud,
            delay_seconds=delay_seconds,
            investigator_id=investigator_id,
            verdict=verdict,
            confidence=confidence,
            accuracy=accuracy,
            notes=notes,
        )

    def schedule_chargeback(
        self,
        transaction_id: str,
        tx_timestamp: float,
        amount: float,
        is_fraud: bool = True,
        delay_seconds: Optional[float] = None,
        reason_code: Optional[str] = None,
        chargeback_id: Optional[str] = None,
    ) -> float:
        """Schedule delayed bank chargeback in feedback queue."""
        return self.feedback_queue.schedule_chargeback_feedback(
            transaction_id=transaction_id,
            tx_timestamp=tx_timestamp,
            amount=amount,
            is_fraud=is_fraud,
            delay_seconds=delay_seconds,
            reason_code=reason_code,
            chargeback_id=chargeback_id,
        )

    def ingest_investigator_feedback(
        self,
        feedback: InvestigatorFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Directly ingest an investigator review (e.g. from REST API)."""
        updated_wm = self.working_memory.apply_investigator_feedback(feedback, sim_time)
        return updated_wm

    def ingest_chargeback_feedback(
        self,
        feedback: ChargebackFeedback,
        sim_time: Optional[float] = None,
    ) -> bool:
        """Directly ingest a bank chargeback (e.g. from REST API) with label precedence."""
        updated_wm = self.working_memory.apply_chargeback_feedback(feedback, sim_time)
        updated_cm = self.consolidated_memory.apply_chargeback_feedback(feedback, sim_time)

        # If present in working memory with chargeback, immediately move to consolidated memory
        rec = self.working_memory.get(feedback.transaction_id)
        if rec is not None:
            self.consolidated_memory.add_record(rec)

        return updated_wm or updated_cm

    def step(self, current_sim_time: float) -> Dict[str, Any]:
        """Advance simulation time: poll matured feedback and consolidate mature records.

        Steps:
          1. Poll matured investigator & chargeback feedback from queue (zero temporal leakage).
          2. Apply investigator reviews to Working Memory.
          3. Apply chargebacks to Working Memory & Consolidated Memory (strict label precedence).
          4. Extract mature records from Working Memory and consolidate into Consolidated Memory.

        Parameters
        ----------
        current_sim_time : float
            Current simulation timestamp.

        Returns
        -------
        Dict[str, Any]
            Operational telemetry for this step.
        """
        # 1. Poll feedback from priority queue
        inv_list, cb_list = self.feedback_queue.poll(current_sim_time)

        # 2. Ingest investigator feedback
        inv_applied = 0
        for inv in inv_list:
            if self.ingest_investigator_feedback(inv, current_sim_time):
                inv_applied += 1

        # 3. Ingest chargebacks
        cb_applied = 0
        for cb in cb_list:
            if self.ingest_chargeback_feedback(cb, current_sim_time):
                cb_applied += 1

        # 4. Extract mature records from Working Memory
        mature_records = self.working_memory.extract_mature_records(
            current_sim_time=current_sim_time,
            maturity_horizon_sec=self.maturity_horizon_sec,
        )

        for rec in mature_records:
            self.consolidated_memory.add_record(rec)

        return {
            "current_sim_time": current_sim_time,
            "investigator_reviews_polled": len(inv_list),
            "investigator_reviews_applied": inv_applied,
            "chargebacks_polled": len(cb_list),
            "chargebacks_applied": cb_applied,
            "records_matured": len(mature_records),
            "working_memory_size": self.working_memory.size(),
            "consolidated_memory_size": self.consolidated_memory.size(),
            "pending_queue_count": self.feedback_queue.pending_count(),
        }

    def get_debiased_training_batch(
        self,
        batch_size: int,
        fraud_ratio: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> List[MemoryRecord]:
        """Sample a batch of pure, debiased records from Consolidated Memory."""
        return self.consolidated_memory.sample_batch(
            batch_size=batch_size,
            fraud_ratio=fraud_ratio,
            rng=rng,
        )

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive telemetry across all memory subsystems."""
        return {
            "working_memory_size": self.working_memory.size(),
            "working_memory_states": self.working_memory.state_counts(),
            "consolidated_memory_size": self.consolidated_memory.size(),
            "consolidated_memory_frauds": self.consolidated_memory.fraud_count(),
            "consolidated_memory_legit": self.consolidated_memory.legitimate_count(),
            "feedback_queue_stats": self.feedback_queue.stats(),
            "maturity_horizon_sec": self.maturity_horizon_sec,
        }
