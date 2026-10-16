"""StreamGuard Asynchronous Delayed Feedback Queues.

Provides priority-queue and event-driven buffers for the two asynchronous
feedback channels in financial fraud detection:
  1. Fast Human Investigator Review (1-4 hours latency):
     Captures investigator ID, verdict (FRAUD / LEGITIMATE), confidence score,
     and review notes.
  2. Delayed Bank Chargebacks & Disputes (14-30 days latency):
     Captures chargeback ID, network reason codes (e.g. Visa 10.4, MC 4837),
     disputed amount, and definitive dispute ground truth.

Enforces strict event-time gating to guarantee ZERO temporal leakage:
feedback events are held in scheduled delivery min-heaps and can never be
emitted or observed until simulation time reaches or exceeds the delivery time.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import (
    FAST_FEEDBACK_MAX_SEC,
    FAST_FEEDBACK_MIN_SEC,
    DELAYED_FEEDBACK_MAX_SEC,
    DELAYED_FEEDBACK_MIN_SEC,
    StreamGuardConfig,
    default_config,
)


class InvestigatorVerdict(str, Enum):
    """Investigator review outcomes."""
    FRAUD = "FRAUD"
    LEGITIMATE = "LEGITIMATE"


class FeedbackChannel(str, Enum):
    """Feedback channel identifiers."""
    INVESTIGATOR = "INVESTIGATOR"
    CHARGEBACK = "CHARGEBACK"


@dataclass(slots=True)
class InvestigatorFeedback:
    """Fast human investigator review event arriving in 1-4 hours."""
    transaction_id: str
    investigator_id: str
    verdict: str  # "FRAUD" | "LEGITIMATE"
    confidence: float  # [0.0, 1.0]
    reviewed_at: float  # Event time when review occurred
    scheduled_arrival_time: float  # Event time when feedback becomes available
    notes: str = ""

    def __post_init__(self) -> None:
        # Normalize verdict string
        norm_v = str(self.verdict).upper().strip()
        if norm_v in ("1", "TRUE", "FRAUD"):
            norm_v = "FRAUD"
        elif norm_v in ("0", "FALSE", "LEGITIMATE", "CLEARED"):
            norm_v = "LEGITIMATE"
        object.__setattr__(self, "verdict", norm_v)
        object.__setattr__(self, "confidence", float(max(0.0, min(1.0, self.confidence))))

    @property
    def is_fraud(self) -> bool:
        """Boolean indicator if investigator deemed transaction fraudulent."""
        return self.verdict == InvestigatorVerdict.FRAUD.value

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigatorFeedback":
        """Instantiate from dictionary or JSON payload."""
        rev_at = float(data.get("reviewed_at", 0.0))
        sched_time = float(data.get("scheduled_arrival_time", rev_at))
        return cls(
            transaction_id=str(data.get("transaction_id", "")),
            investigator_id=str(data.get("investigator_id", "")),
            verdict=str(data.get("verdict", "LEGITIMATE")),
            confidence=float(data.get("confidence", 0.95)),
            reviewed_at=rev_at,
            scheduled_arrival_time=sched_time,
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class ChargebackFeedback:
    """Definitive bank chargeback dispute arriving in 14-30 days."""
    transaction_id: str
    chargeback_id: str
    reason_code: str
    chargeback_amount: float
    received_at: float  # Event time when chargeback notice was received
    scheduled_arrival_time: float  # Event time when chargeback is accessible
    is_fraud: bool = True  # Network dispute ground truth flag

    def __post_init__(self) -> None:
        object.__setattr__(self, "chargeback_amount", float(self.chargeback_amount))
        object.__setattr__(self, "is_fraud", bool(self.is_fraud))

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChargebackFeedback":
        """Instantiate from dictionary or JSON payload."""
        rec_at = float(data.get("received_at", 0.0))
        sched_time = float(data.get("scheduled_arrival_time", rec_at))
        return cls(
            transaction_id=str(data.get("transaction_id", "")),
            chargeback_id=str(data.get("chargeback_id", "")),
            reason_code=str(data.get("reason_code", "10.4_FRAUD_CARD_ABSENT")),
            chargeback_amount=float(data.get("chargeback_amount", data.get("amount", 0.0))),
            received_at=rec_at,
            scheduled_arrival_time=sched_time,
            is_fraud=bool(data.get("is_fraud", True)),
        )


class FeedbackPriorityQueue:
    """Asynchronous dual-channel delayed feedback queue with strict temporal gating.

    Ensures mathematical zero temporal leakage: feedback events are stored in a
    min-heap priority queue keyed by scheduled delivery timestamp and can ONLY be
    retrieved when `current_sim_time >= scheduled_arrival_time`.
    """

    def __init__(self, config: Optional[StreamGuardConfig] = None) -> None:
        self.config = config or default_config
        # Heap elements: (scheduled_arrival_time, sequence_id, channel_name, feedback_obj)
        self._heap: List[Tuple[float, int, str, Union[InvestigatorFeedback, ChargebackFeedback]]] = []
        self._counter: int = 0
        self._enqueued_investigators: int = 0
        self._enqueued_chargebacks: int = 0
        self._delivered_investigators: int = 0
        self._delivered_chargebacks: int = 0

    def schedule_investigator_feedback(
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
        """Schedule fast investigator review feedback with 1-4 hours review latency.

        Parameters
        ----------
        transaction_id : str
            Unique identifier of transaction.
        tx_timestamp : float
            Timestamp when the transaction originally occurred.
        is_fraud : bool
            Ground truth flag used to simulate human review verdict if not specified.
        delay_seconds : Optional[float]
            Review latency in seconds. Defaults to Uniform(3600, 14400) (1-4 hours).
        investigator_id : Optional[str]
            Investigator identifier.
        verdict : Optional[str]
            Explicit verdict ("FRAUD" | "LEGITIMATE"). If None, simulated with `accuracy`.
        confidence : Optional[float]
            Confidence score [0.0, 1.0].
        accuracy : float
            Simulated review accuracy if verdict not explicitly supplied.
        notes : str
            Reviewer commentary or rationale.

        Returns
        -------
        float
            Scheduled arrival timestamp (tx_timestamp + delay_seconds).
        """
        if delay_seconds is None:
            delay_seconds = random.uniform(
                self.config.fast_feedback_min_sec,
                self.config.fast_feedback_max_sec,
            )

        delivery_time = tx_timestamp + delay_seconds
        inv_id = investigator_id or f"inv_{random.randint(100, 999)}"

        if verdict is None:
            if is_fraud:
                resolved_verdict = "FRAUD" if random.random() < accuracy else "LEGITIMATE"
                default_conf = random.uniform(0.85, 0.99) if resolved_verdict == "FRAUD" else random.uniform(0.50, 0.70)
            else:
                resolved_verdict = "LEGITIMATE" if random.random() < accuracy else "FRAUD"
                default_conf = random.uniform(0.88, 0.99) if resolved_verdict == "LEGITIMATE" else random.uniform(0.50, 0.65)
        else:
            resolved_verdict = verdict
            default_conf = 0.95

        conf = confidence if confidence is not None else default_conf
        review_notes = notes or f"Investigator review completed after {delay_seconds/3600.0:.2f}h"

        fb = InvestigatorFeedback(
            transaction_id=transaction_id,
            investigator_id=inv_id,
            verdict=resolved_verdict,
            confidence=round(conf, 4),
            reviewed_at=delivery_time,
            scheduled_arrival_time=delivery_time,
            notes=review_notes,
        )

        self.enqueue_investigator(fb, scheduled_time=delivery_time)
        return delivery_time

    def schedule_chargeback_feedback(
        self,
        transaction_id: str,
        tx_timestamp: float,
        amount: float,
        is_fraud: bool = True,
        delay_seconds: Optional[float] = None,
        reason_code: Optional[str] = None,
        chargeback_id: Optional[str] = None,
    ) -> float:
        """Schedule delayed bank chargeback feedback with 14-30 days dispute latency.

        Parameters
        ----------
        transaction_id : str
            Unique identifier of transaction.
        tx_timestamp : float
            Timestamp when the transaction originally occurred.
        amount : float
            Disputed chargeback amount.
        is_fraud : bool
            True if chargeback represents confirmed fraud dispute.
        delay_seconds : Optional[float]
            Dispute latency in seconds. Defaults to Uniform(14*86400, 30*86400).
        reason_code : Optional[str]
            Card network reason code.
        chargeback_id : Optional[str]
            Chargeback reference identifier.

        Returns
        -------
        float
            Scheduled arrival timestamp (tx_timestamp + delay_seconds).
        """
        if delay_seconds is None:
            delay_seconds = random.uniform(
                self.config.delayed_feedback_min_sec,
                self.config.delayed_feedback_max_sec,
            )

        delivery_time = tx_timestamp + delay_seconds
        cb_id = chargeback_id or f"cb_{random.randint(100000, 999999)}"
        rc = reason_code or ("10.4_FRAUD_CARD_ABSENT" if is_fraud else "4837_DISPUTE_RECOGNIZED")

        fb = ChargebackFeedback(
            transaction_id=transaction_id,
            chargeback_id=cb_id,
            reason_code=rc,
            chargeback_amount=amount,
            received_at=delivery_time,
            scheduled_arrival_time=delivery_time,
            is_fraud=is_fraud,
        )

        self.enqueue_chargeback(fb, scheduled_time=delivery_time)
        return delivery_time

    def enqueue_investigator(
        self,
        feedback: InvestigatorFeedback,
        scheduled_time: Optional[float] = None,
    ) -> float:
        """Enqueue an investigator feedback item into the priority queue."""
        delivery_time = scheduled_time if scheduled_time is not None else feedback.scheduled_arrival_time
        self._counter += 1
        self._enqueued_investigators += 1
        heapq.heappush(
            self._heap,
            (delivery_time, self._counter, FeedbackChannel.INVESTIGATOR.value, feedback),
        )
        return delivery_time

    def enqueue_chargeback(
        self,
        feedback: ChargebackFeedback,
        scheduled_time: Optional[float] = None,
    ) -> float:
        """Enqueue a chargeback feedback item into the priority queue."""
        delivery_time = scheduled_time if scheduled_time is not None else feedback.scheduled_arrival_time
        self._counter += 1
        self._enqueued_chargebacks += 1
        heapq.heappush(
            self._heap,
            (delivery_time, self._counter, FeedbackChannel.CHARGEBACK.value, feedback),
        )
        return delivery_time

    def poll(
        self,
        current_sim_time: float,
    ) -> Tuple[List[InvestigatorFeedback], List[ChargebackFeedback]]:
        """Poll and return all feedback events whose scheduled arrival time has matured.

        Strictly enforces ZERO temporal leakage: any event where
        scheduled_arrival_time > current_sim_time remains safely in the heap.

        Parameters
        ----------
        current_sim_time : float
            Current simulation epoch timestamp.

        Returns
        -------
        Tuple[List[InvestigatorFeedback], List[ChargebackFeedback]]
            Two lists containing matured investigator and chargeback feedback events.
        """
        investigators: List[InvestigatorFeedback] = []
        chargebacks: List[ChargebackFeedback] = []

        while self._heap and self._heap[0][0] <= current_sim_time:
            delivery_time, _, channel, feedback = heapq.heappop(self._heap)
            if channel == FeedbackChannel.INVESTIGATOR.value:
                assert isinstance(feedback, InvestigatorFeedback)
                investigators.append(feedback)
                self._delivered_investigators += 1
            elif channel == FeedbackChannel.CHARGEBACK.value:
                assert isinstance(feedback, ChargebackFeedback)
                chargebacks.append(feedback)
                self._delivered_chargebacks += 1

        return investigators, chargebacks

    def peek_next_arrival_time(self) -> Optional[float]:
        """Return the scheduled arrival time of the earliest pending event, or None."""
        if not self._heap:
            return None
        return self._heap[0][0]

    def pending_count(self) -> int:
        """Return total count of pending feedback events in queue."""
        return len(self._heap)

    def pending_investigator_count(self) -> int:
        """Return count of pending investigator events."""
        return sum(1 for item in self._heap if item[2] == FeedbackChannel.INVESTIGATOR.value)

    def pending_chargeback_count(self) -> int:
        """Return count of pending chargeback events."""
        return sum(1 for item in self._heap if item[2] == FeedbackChannel.CHARGEBACK.value)

    def stats(self) -> Dict[str, Any]:
        """Return telemetry and queue operational statistics."""
        return {
            "total_enqueued": self._counter,
            "enqueued_investigators": self._enqueued_investigators,
            "enqueued_chargebacks": self._enqueued_chargebacks,
            "pending_total": len(self._heap),
            "pending_investigators": self.pending_investigator_count(),
            "pending_chargebacks": self.pending_chargeback_count(),
            "delivered_investigators": self._delivered_investigators,
            "delivered_chargebacks": self._delivered_chargebacks,
            "earliest_pending_time": self.peek_next_arrival_time(),
        }

    def clear(self) -> None:
        """Clear all pending feedback queues and reset telemetry counters."""
        self._heap.clear()
        self._counter = 0
        self._enqueued_investigators = 0
        self._enqueued_chargebacks = 0
        self._delivered_investigators = 0
        self._delivered_chargebacks = 0


# Backward-compatible alias for existing tests and synthetic stream generator
DelayedFeedbackQueue = FeedbackPriorityQueue
