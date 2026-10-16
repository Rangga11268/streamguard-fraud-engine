"""
System-wide configuration, cost parameters, and latency defaults for StreamGuard.
"""

from dataclasses import dataclass, field
from typing import Tuple, List

# Standard feature names in exact vector order (16 dimensions)
FEATURE_NAMES: List[str] = [
    "amount",
    "velocity_5m_count",
    "velocity_1h_count",
    "velocity_24h_count",
    "spending_5m_sum",
    "spending_1h_sum",
    "spending_24h_sum",
    "amount_to_mean24h_ratio",
    "velocity_5m_to_24h_ratio",
    "card_velocity_5m_count",
    "card_velocity_1h_count",
    "card_velocity_24h_count",
    "card_spending_24h_sum",
    "card_distinct_mcc_count",
    "travel_speed_kmh",
    "impossible_travel_flag",
]

# Window durations in seconds
WINDOW_5M: float = 300.0
WINDOW_1H: float = 3600.0
WINDOW_24H: float = 86400.0
WINDOW_DURATIONS: Tuple[float, float, float] = (WINDOW_5M, WINDOW_1H, WINDOW_24H)

# Latency and throughput targets (milliseconds)
MAX_SCORING_LATENCY_MS: float = 15.0
FEATURE_EXTRACT_TARGET_MS: float = 5.0

# Geographic velocity parameters
IMPOSSIBLE_TRAVEL_SPEED_KMH: float = 800.0
EARTH_RADIUS_KM: float = 6371.0
MIN_TIME_DELTA_HOURS: float = 1e-4

# Numeric stability epsilon
EPSILON: float = 1e-4

# Inactive entity TTL eviction (seconds)
ENTITY_TTL_SEC: float = 86400.0

# Cost matrix parameters ($)
COST_FP_FRICTION: float = 5.0           # Customer friction cost for false positive OTP/challenge
COST_FN_FIXED: float = 25.0             # Bank chargeback processing fee added to amount
COST_OTP_CHALLENGE: float = 0.05        # SMS / OTP provider cost per challenge
COST_CHALLENGE_FRICTION: float = 1.50   # Abandonment friction penalty on challenge
OTP_INTERCEPTION_RATE: float = 0.95     # Success rate of challenge catching fraud
COST_BLOCK_CHURN: float = 50.0          # Customer lifetime value churn cost on false decline
COST_BLOCK_INVESTIGATION: float = 2.0   # Operational manual review cost for blocked fraud

# Feedback latency parameters (seconds)
FAST_FEEDBACK_MIN_SEC: float = 3600.0           # 1 hour
FAST_FEEDBACK_MAX_SEC: float = 14400.0          # 4 hours
DELAYED_FEEDBACK_MIN_SEC: float = 14 * 86400.0  # 14 days
DELAYED_FEEDBACK_MAX_SEC: float = 30 * 86400.0  # 30 days
MATURITY_HORIZON_SEC: float = 20 * 86400.0      # 20 days maturity horizon

# Unsupervised drift monitoring defaults
DRIFT_WINDOW_SIZE: int = 150
REFERENCE_WINDOW_SIZE: int = 1000
PSI_WARNING_THRESHOLD: float = 0.10
PSI_ALERT_THRESHOLD: float = 0.25
WASSERSTEIN_ALERT_THRESHOLD: float = 0.05

# Decision thresholds defaults
DEFAULT_TAU_CHALLENGE: float = 0.15
DEFAULT_TAU_BLOCK: float = 0.70


@dataclass(slots=True)
class StreamGuardConfig:
    """Config dataclass encapsulating all runtime settings and thresholds."""
    # Window durations
    window_5m: float = WINDOW_5M
    window_1h: float = WINDOW_1H
    window_24h: float = WINDOW_24H

    # Latencies
    max_scoring_latency_ms: float = MAX_SCORING_LATENCY_MS
    feature_extract_target_ms: float = FEATURE_EXTRACT_TARGET_MS

    # Geo velocity
    impossible_travel_speed_kmh: float = IMPOSSIBLE_TRAVEL_SPEED_KMH
    earth_radius_km: float = EARTH_RADIUS_KM
    min_time_delta_hours: float = MIN_TIME_DELTA_HOURS

    # Numerical
    epsilon: float = EPSILON
    entity_ttl_sec: float = ENTITY_TTL_SEC

    # Costs
    cost_fp_friction: float = COST_FP_FRICTION
    cost_fn_fixed: float = COST_FN_FIXED
    cost_otp_challenge: float = COST_OTP_CHALLENGE
    cost_challenge_friction: float = COST_CHALLENGE_FRICTION
    otp_interception_rate: float = OTP_INTERCEPTION_RATE
    cost_block_churn: float = COST_BLOCK_CHURN
    cost_block_investigation: float = COST_BLOCK_INVESTIGATION

    # Delays
    fast_feedback_min_sec: float = FAST_FEEDBACK_MIN_SEC
    fast_feedback_max_sec: float = FAST_FEEDBACK_MAX_SEC
    delayed_feedback_min_sec: float = DELAYED_FEEDBACK_MIN_SEC
    delayed_feedback_max_sec: float = DELAYED_FEEDBACK_MAX_SEC
    maturity_horizon_sec: float = MATURITY_HORIZON_SEC

    # Drift
    drift_window_size: int = DRIFT_WINDOW_SIZE
    reference_window_size: int = REFERENCE_WINDOW_SIZE
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD
    psi_alert_threshold: float = PSI_ALERT_THRESHOLD
    wasserstein_alert_threshold: float = WASSERSTEIN_ALERT_THRESHOLD

    # Thresholds
    default_tau_challenge: float = DEFAULT_TAU_CHALLENGE
    default_tau_block: float = DEFAULT_TAU_BLOCK

    # Feature definitions
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    feature_dim: int = 16


# Global default configuration instance
default_config = StreamGuardConfig()
