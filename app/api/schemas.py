"""
Pydantic v2 schemas for StreamGuard Ultra-Low-Latency Fraud Engine REST API.

Enforces strict validation, range constraints, and serialization matching the
authoritative OpenAPI contracts defined in PROJECT.md and the survey specifications.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class Location(BaseModel):
    """Geographic coordinate representation."""
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Geographic latitude coordinate between -90.0 and 90.0 degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Geographic longitude coordinate between -180.0 and 180.0 degrees",
    )

    @field_validator("latitude", "longitude")
    @classmethod
    def validate_finite_coordinates(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Geographic coordinate must be a finite real number (cannot be NaN or Inf)")
        return v


class ScoreRequest(BaseModel):
    """Payload for real-time transaction scoring via POST /v1/score."""
    transaction_id: str = Field(..., min_length=1, description="Unique transaction event identifier")
    customer_id: str = Field(..., min_length=1, description="Customer entity identifier")
    card_id: str = Field(..., min_length=1, description="Payment card identifier")
    amount: float = Field(..., gt=0.0, description="Monetary transaction amount (strictly > 0.0)")
    currency: str = Field(default="USD", description="Three-letter ISO currency code")
    timestamp: float = Field(..., gt=0.0, description="Epoch seconds timestamp of transaction event (> 0.0)")
    merchant_id: str = Field(default="", description="Merchant identifier")
    merchant_category_code: str = Field(default="5411", description="4-digit Merchant Category Code (MCC)")
    channel: str = Field(default="e-commerce", description="Transaction channel ('pos', 'e-commerce', 'atm')")
    location: Location = Field(..., description="Geographic location of transaction")
    device_id: str = Field(default="", description="Originating client device identifier")
    ip_address: str = Field(default="", description="Client IP address")

    @field_validator("amount", "timestamp")
    @classmethod
    def validate_finite_numerics(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Numeric value must be a finite real number (cannot be NaN or Inf)")
        return v



class ScoreFeatures(BaseModel):
    """Summary sliding-window and behavioral features included in ScoreResponse."""
    velocity_5m_count: int = Field(default=0, description="Customer transaction count in 5-minute sliding window")
    velocity_1h_count: int = Field(default=0, description="Customer transaction count in 1-hour sliding window")
    velocity_24h_count: int = Field(default=0, description="Customer transaction count in 24-hour sliding window")
    amount_to_mean24h_ratio: float = Field(default=1.0, description="Ratio of current amount to 24h mean spending")
    travel_speed_kmh: float = Field(default=0.0, description="Implied Haversine ground speed from previous location (km/h)")
    impossible_travel_flag: int = Field(default=0, description="Binary flag indicating impossible geographic velocity (>800 km/h)")


class ScoreResponse(BaseModel):
    """Response payload returned by POST /v1/score."""
    transaction_id: str = Field(..., description="Unique transaction event identifier")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated fraud posterior probability P(fraud|x)")
    decision: str = Field(..., description="Operational action ('APPROVE', 'CHALLENGE', 'BLOCK')")
    risk_tier: str = Field(..., description="Categorical risk tier ('LOW', 'MEDIUM', 'HIGH')")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end evaluation latency in milliseconds")
    features: ScoreFeatures = Field(..., description="Extracted streaming features")
    evaluated_at: str = Field(..., description="ISO 8601 UTC timestamp of evaluation")


class InvestigatorFeedbackRequest(BaseModel):
    """Payload for human investigator fast feedback review via POST /v1/feedback/investigator."""
    transaction_id: str = Field(..., min_length=1, description="Transaction identifier being reviewed")
    investigator_id: str = Field(..., min_length=1, description="Identity of the reviewing investigator agent")
    verdict: str = Field(..., description="Review outcome verdict: 'FRAUD' or 'LEGITIMATE'")
    reviewed_at: float = Field(..., gt=0.0, description="Epoch seconds timestamp when review was performed")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Review confidence score in [0.0, 1.0]")
    notes: Optional[str] = Field(default="", description="Investigator qualitative audit notes")

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        upper = v.upper().strip()
        if upper not in ("FRAUD", "LEGITIMATE"):
            raise ValueError("Verdict must be either 'FRAUD' or 'LEGITIMATE'")
        return upper


class InvestigatorFeedbackResponse(BaseModel):
    """Response payload returned by POST /v1/feedback/investigator."""
    status: str = Field(default="ACCEPTED", description="Processing status")
    transaction_id: str = Field(..., description="Transaction identifier")
    memory_target: str = Field(default="FAST_WORKING_MEMORY", description="Target memory tier")
    provisional_label: int = Field(..., description="Provisional binary label (1=FRAUD, 0=LEGITIMATE)")
    ingested_at: str = Field(..., description="ISO 8601 UTC timestamp of feedback ingestion")


class ChargebackFeedbackRequest(BaseModel):
    """Payload for bank dispute / chargeback delayed ground truth via POST /v1/feedback/chargeback."""
    transaction_id: str = Field(..., min_length=1, description="Transaction identifier being disputed")
    chargeback_id: str = Field(..., min_length=1, description="Unique bank chargeback case reference")
    reason_code: str = Field(default="10.4_FRAUD_CARD_ABSENT", description="Network dispute reason code")
    chargeback_amount: float = Field(..., gt=0.0, description="Disputed transaction monetary amount")
    received_at: float = Field(..., gt=0.0, description="Epoch seconds timestamp when chargeback was served")
    is_fraud: bool = Field(default=True, description="Ground truth fraud confirmation flag")


class ChargebackFeedbackResponse(BaseModel):
    """Response payload returned by POST /v1/feedback/chargeback."""
    status: str = Field(default="CONSOLIDATED", description="Processing status")
    transaction_id: str = Field(..., description="Transaction identifier")
    memory_target: str = Field(default="CONSOLIDATED_GROUND_TRUTH", description="Target memory tier")
    final_label: int = Field(..., description="Final ground-truth binary label (1=FRAUD, 0=LEGITIMATE)")
    reconciled_with_investigator: bool = Field(..., description="True if prior investigator review was reconciled")
    consolidated_at: str = Field(..., description="ISO 8601 UTC timestamp of memory consolidation")


class DriftTelemetryResponse(BaseModel):
    """Response payload returned by GET /v1/telemetry/drift."""
    window_samples: int = Field(..., description="Number of observations in the sliding evaluation window")
    score_psi: float = Field(..., description="Population Stability Index on risk scores")
    score_wasserstein: float = Field(..., description="Wasserstein-1 Earth Mover's Distance on risk scores")
    feature_psi: Dict[str, float] = Field(default_factory=dict, description="Per-feature Population Stability Index")
    drift_status: str = Field(..., description="Stability classification: 'STABLE', 'WARNING', 'DRIFT_DETECTED'")
    drift_alert: bool = Field(..., description="True if active drift alert threshold is breached")
    sample_count_since_drift_onset: int = Field(default=0, description="Count of streaming samples since onset")
    last_recalibration_timestamp: Optional[str] = Field(default=None, description="ISO timestamp of last recalibration")
    current_thresholds: Dict[str, float] = Field(..., description="Active decision thresholds (tau_challenge, tau_block)")
    feature_wasserstein: Optional[Dict[str, float]] = Field(default=None, description="Per-feature Wasserstein-1 metrics")
    recalibration_triggered: Optional[bool] = Field(default=False, description="True if recalibration fired on last sample")
