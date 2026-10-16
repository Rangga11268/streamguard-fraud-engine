"""
REST API Routes for StreamGuard Ultra-Low-Latency Fraud Engine.

Provides high-performance async endpoints:
  - POST /v1/score: Real-time feature extraction, model scoring, 3-tier routing, drift tracking
  - POST /v1/feedback/investigator: Fast human investigator review ingestion (1-4h delay)
  - POST /v1/feedback/chargeback: Delayed bank chargeback dispute ingestion (14-30d ground truth)
  - GET /v1/telemetry/drift: Real-time PSI, Wasserstein, alert status, and dynamic thresholds
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.schemas import (
    ChargebackFeedbackRequest,
    ChargebackFeedbackResponse,
    DriftTelemetryResponse,
    InvestigatorFeedbackRequest,
    InvestigatorFeedbackResponse,
    ScoreFeatures,
    ScoreRequest,
    ScoreResponse,
)
from app.features.feature_store import TransactionEntityEvent
from app.memory.feedback_queue import ChargebackFeedback, InvestigatorFeedback

router = APIRouter(prefix="/v1", tags=["StreamGuard Fraud Engine"])


def get_engine_state(request: Request):
    """Retrieve StreamGuardEngine instance from application state."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not getattr(engine, "initialized", False):
        from app.main import get_global_engine
        engine = get_global_engine()
        request.app.state.engine = engine
    if engine is None or not getattr(engine, "initialized", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="StreamGuard Fraud Engine is not initialized",
        )
    return engine


@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-time transaction scoring and 3-tier routing decision",
)
async def score_transaction(
    payload: ScoreRequest,
    engine=Depends(get_engine_state),
) -> ScoreResponse:
    """Score incoming transaction event, extract features, route action, and update telemetry."""
    t0 = time.perf_counter()

    # 1. Convert to TransactionEntityEvent & extract sliding features
    tx_event = TransactionEntityEvent(
        transaction_id=payload.transaction_id,
        customer_id=payload.customer_id,
        card_id=payload.card_id,
        amount=payload.amount,
        timestamp=payload.timestamp,
        merchant_category_code=payload.merchant_category_code,
        latitude=payload.location.latitude,
        longitude=payload.location.longitude,
    )
    features = engine.feature_store.extract_features(tx_event, commit=True)

    # 2. Real-time CPU model inference
    pred = engine.classifier.predict(features.feature_vector)
    fraud_prob = pred.calibrated_prob

    # 3. 3-tier cost-sensitive decision routing
    routing_result = engine.router.route(
        transaction_id=payload.transaction_id,
        fraud_score=fraud_prob,
        amount=payload.amount,
        features=features.to_dict(),
        start_time=t0,
    )

    # 4. Ingest score observation into unsupervised drift monitor
    telemetry = engine.drift_monitor.record_prediction(
        score=fraud_prob,
        features=features.to_dict(),
        timestamp=payload.timestamp,
    )

    # Trigger automatic threshold recalibration if drift monitor signaled alert
    if telemetry.drift_alert or telemetry.recalibration_triggered:
        engine.router.recalibrate(telemetry.to_dict())

    # 5. Ingest transaction into Working Memory
    engine.dual_memory.ingest_transaction(
        event=payload.model_dump(),
        feature_vector=features.feature_vector,
        risk_score=fraud_prob,
        metadata={
            "decision": routing_result.decision,
            "risk_tier": routing_result.risk_tier,
            "latency_ms": routing_result.latency_ms,
        },
    )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ScoreResponse(
        transaction_id=payload.transaction_id,
        fraud_score=round(fraud_prob, 4),
        decision=routing_result.decision,
        risk_tier=routing_result.risk_tier,
        latency_ms=round(routing_result.latency_ms, 3),
        features=ScoreFeatures(**features.to_dict()),
        evaluated_at=routing_result.evaluated_at or now_iso,
    )


@router.post(
    "/feedback/investigator",
    response_model=InvestigatorFeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest fast human investigator review verdict",
)
async def ingest_investigator_feedback(
    payload: InvestigatorFeedbackRequest,
    engine=Depends(get_engine_state),
) -> InvestigatorFeedbackResponse:
    """Ingest human review verdict into fast working memory buffer."""
    # Check if transaction exists in active working memory window
    rec = engine.dual_memory.working_memory.get(payload.transaction_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{payload.transaction_id}' not found in active working window",
        )

    # Instantiate feedback event and update working memory
    feedback = InvestigatorFeedback(
        transaction_id=payload.transaction_id,
        investigator_id=payload.investigator_id,
        verdict=payload.verdict,
        confidence=payload.confidence,
        reviewed_at=payload.reviewed_at,
        scheduled_arrival_time=payload.reviewed_at,
        notes=payload.notes or "",
    )
    engine.dual_memory.ingest_investigator_feedback(feedback, sim_time=payload.reviewed_at)

    provisional_label = 1 if payload.verdict.upper() == "FRAUD" else 0
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return InvestigatorFeedbackResponse(
        status="ACCEPTED",
        transaction_id=payload.transaction_id,
        memory_target="FAST_WORKING_MEMORY",
        provisional_label=provisional_label,
        ingested_at=now_iso,
    )


@router.post(
    "/feedback/chargeback",
    response_model=ChargebackFeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest delayed bank chargeback dispute ground truth",
)
async def ingest_chargeback_feedback(
    payload: ChargebackFeedbackRequest,
    engine=Depends(get_engine_state),
) -> ChargebackFeedbackResponse:
    """Ingest definitive bank dispute into consolidated ground-truth replay memory."""
    rec_wm = engine.dual_memory.working_memory.get(payload.transaction_id)
    rec_cm = engine.dual_memory.consolidated_memory.get(payload.transaction_id)

    if rec_wm is None and rec_cm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{payload.transaction_id}' not found in system history",
        )

    # Check reconciliation status with prior investigator verdict
    target_rec = rec_cm or rec_wm
    reconciled = target_rec.investigator_feedback is not None

    # Instantiate chargeback feedback and ingest with strict label precedence
    feedback = ChargebackFeedback(
        transaction_id=payload.transaction_id,
        chargeback_id=payload.chargeback_id,
        reason_code=payload.reason_code,
        chargeback_amount=payload.chargeback_amount,
        received_at=payload.received_at,
        scheduled_arrival_time=payload.received_at,
        is_fraud=payload.is_fraud,
    )
    engine.dual_memory.ingest_chargeback_feedback(feedback, sim_time=payload.received_at)

    final_label = 1 if payload.is_fraud else 0
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ChargebackFeedbackResponse(
        status="CONSOLIDATED",
        transaction_id=payload.transaction_id,
        memory_target="CONSOLIDATED_GROUND_TRUTH",
        final_label=final_label,
        reconciled_with_investigator=reconciled,
        consolidated_at=now_iso,
    )


@router.get(
    "/telemetry/drift",
    response_model=DriftTelemetryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query real-time unsupervised PSI and Wasserstein drift metrics",
)
async def get_drift_telemetry(
    window_size: int = Query(default=1000, description="Sliding evaluation window size"),
    feature: str = Query(default="score", description="Target feature or 'all'"),
    metric: str = Query(default="all", description="Metric type ('psi', 'wasserstein', 'all')"),
    engine=Depends(get_engine_state),
) -> DriftTelemetryResponse:
    """Expose real-time Population Stability Index (PSI) and Wasserstein drift metrics."""
    # Parameter validation per specification
    allowed_features = {
        "score",
        "amount",
        "velocity_5m",
        "velocity_5m_count",
        "velocity_1h_count",
        "all",
    }
    allowed_metrics = {"psi", "wasserstein", "all"}

    feat_clean = feature.lower().strip()
    met_clean = metric.lower().strip()

    if feat_clean not in allowed_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feature parameter: '{feature}'. Allowed: {sorted(allowed_features)}",
        )
    if met_clean not in allowed_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metric parameter: '{metric}'. Allowed: {sorted(allowed_metrics)}",
        )
    if window_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="window_size must be a positive integer",
        )

    tel = engine.drift_monitor.get_telemetry()
    current_thresh = engine.router.current_thresholds

    return DriftTelemetryResponse(
        window_samples=tel.window_samples,
        score_psi=round(float(tel.score_psi), 4),
        score_wasserstein=round(float(tel.score_wasserstein), 4),
        feature_psi={k: round(float(v), 4) for k, v in tel.feature_psi.items()},
        drift_status=tel.drift_status,
        drift_alert=tel.drift_alert,
        sample_count_since_drift_onset=tel.sample_count_since_drift_onset,
        last_recalibration_timestamp=tel.last_recalibration_timestamp,
        current_thresholds={
            "tau_challenge": round(float(current_thresh.get("tau_challenge", 0.15)), 4),
            "tau_block": round(float(current_thresh.get("tau_block", 0.70)), 4),
        },
        feature_wasserstein={k: round(float(v), 4) for k, v in tel.feature_wasserstein.items()},
        recalibration_triggered=tel.recalibration_triggered,
    )
