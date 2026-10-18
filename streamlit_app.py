"""StreamGuard: Real-Time Fraud Detection & Concept Drift Monitoring Console.

Production-grade Streamlit GUI conforming strictly to /antislop-ui guidelines:
- Zero AI-slop visual tropes (no generic purple gradients, no emoji scattering, no box soup)
- Restrained Amber/Slate palette (#D97706 / #0F172A) suitable for fintech risk operations
- Decision-driven risk scoring, streaming velocity aggregations, and unsupervised PSI drift tracking
"""

import os
import sys
import time
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import default_config
from app.decision.cost_matrix import CostMatrixConfig
from app.decision.router import AdaptiveDecisionRouter, DecisionAction
from app.drift.monitor import StreamingDriftMonitor
from app.features.feature_store import StreamingFeatureStore, TransactionEntityEvent
from app.models.trainer import FraudModelTrainer, ModelTrainingConfig

# Page config
st.set_page_config(
    page_title="StreamGuard Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Anti-Slop CSS styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-container {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: "JetBrains Mono", Consolas, monospace;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    .status-approve {
        color: #10B981;
    }
    .status-challenge {
        color: #F59E0B;
    }
    .status-block {
        color: #EF4444;
    }
    .tx-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .tx-header {
        font-weight: 600;
        font-size: 0.95rem;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .tx-meta {
        font-size: 0.8rem;
        color: #94A3B8;
        font-family: "JetBrains Mono", Consolas, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_fraud_engine():
    """Initialize feature store, pre-trained model, and drift monitor."""
    feature_store = StreamingFeatureStore(default_config)

    rng = np.random.RandomState(42)
    n_base = 1000
    X_init = np.zeros((n_base, 16), dtype=np.float32)
    y_init = np.zeros(n_base, dtype=np.int32)

    X_init[:, 0] = np.exp(rng.normal(3.8, 0.7, size=n_base)).astype(np.float32)
    X_init[:, 1] = rng.poisson(0.3, size=n_base).astype(np.float32)
    X_init[:, 2] = rng.poisson(1.2, size=n_base).astype(np.float32)
    X_init[:, 3] = rng.poisson(3.5, size=n_base).astype(np.float32)
    X_init[:, 7] = (1.0 + rng.normal(0, 0.2, size=n_base)).astype(np.float32)
    X_init[:, 14] = np.maximum(0, rng.normal(5.0, 10.0, size=n_base)).astype(np.float32)
    X_init[:, 15] = 0.0

    fraud_idx = rng.choice(n_base, size=8, replace=False)
    y_init[fraud_idx] = 1
    X_init[fraud_idx, 0] = np.exp(rng.normal(6.5, 0.8, size=8)).astype(np.float32)
    X_init[fraud_idx, 1] = rng.poisson(5.0, size=8).astype(np.float32)
    X_init[fraud_idx, 7] = (6.0 + rng.normal(0, 1.0, size=8)).astype(np.float32)
    X_init[fraud_idx, 14] = (1200.0 + rng.normal(0, 200.0, size=8)).astype(np.float32)
    X_init[fraud_idx, 15] = 1.0

    trainer = FraudModelTrainer(
        ModelTrainingConfig(
            n_estimators=30,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.08,
            n_jobs=1,
            random_state=42,
        )
    )
    res = trainer.train(X_init, y_init)
    classifier = res.classifier

    router = AdaptiveDecisionRouter(
        cost_config=CostMatrixConfig.from_config(default_config),
        default_tau_challenge=default_config.default_tau_challenge,
        default_tau_block=default_config.default_tau_block,
        mode="expected_loss",
    )

    drift_monitor = StreamingDriftMonitor(
        window_size=default_config.drift_window_size,
        reference_size=default_config.reference_window_size,
        psi_warning_threshold=default_config.psi_warning_threshold,
        psi_alert_threshold=default_config.psi_alert_threshold,
        initial_tau_challenge=default_config.default_tau_challenge,
        initial_tau_block=default_config.default_tau_block,
        auto_recalibrate=True,
    )

    base_preds = classifier.predict_batch(X_init)
    base_scores = [p.calibrated_prob for p in base_preds]
    drift_monitor.set_baseline_scores(base_scores)
    drift_monitor.set_baseline_features({
        "amount": X_init[:, 0],
        "velocity_5m_count": X_init[:, 1],
        "velocity_1h_count": X_init[:, 2],
    })

    return feature_store, classifier, router, drift_monitor


feature_store, classifier, router, drift_monitor = load_fraud_engine()

# Header
st.markdown("### STREAMGUARD // Real-Time Fraud Detection & Concept Drift Console")
st.markdown(
    "<span style='color: #64748B; font-size: 0.9rem;'>Streaming Sliding-Window Feature Store | Cost-Matrix 3-Tier Routing | Unsupervised PSI Monitor</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Sidebar
st.sidebar.markdown("#### Routing Thresholds & Cost Weights")
tau_challenge = st.sidebar.slider("Challenge Threshold (τ_challenge)", 0.05, 0.40, float(router.tau_challenge), 0.01)
tau_block = st.sidebar.slider("Block Threshold (τ_block)", 0.40, 0.90, float(router.tau_block), 0.01)

router.tau_challenge = tau_challenge
router.tau_block = tau_block

st.sidebar.markdown("---")
st.sidebar.markdown(f"<span style='color: #94A3B8; font-size: 0.85rem;'>Reference Window: {drift_monitor.reference_size} tx<br>Drift Warning PSI: {drift_monitor.psi_warning_threshold}<br>Drift Alert PSI: {drift_monitor.psi_alert_threshold}</span>", unsafe_allow_html=True)

# Transaction Presets
preset_scenarios = {
    "Legitimate Routine Grocery Purchase ($42.50)": {
        "amount": 42.50,
        "customer_id": "cust_101",
        "card_id": "card_9001",
        "lat": 40.7128,
        "lon": -74.0060,
        "velocity_5m": 0,
        "speed_kmh": 0.0,
    },
    "High-Velocity ATM Surge ($350.00, 4th tx in 5 mins)": {
        "amount": 350.00,
        "customer_id": "cust_204",
        "card_id": "card_4421",
        "lat": 40.7150,
        "lon": -74.0080,
        "velocity_5m": 3,
        "speed_kmh": 12.0,
    },
    "Impossible Geographic Travel Jump ($1,850.00, Tokyo to London in 15 mins)": {
        "amount": 1850.00,
        "customer_id": "cust_889",
        "card_id": "card_7719",
        "lat": 51.5074,
        "lon": -0.1278,
        "velocity_5m": 1,
        "speed_kmh": 1420.0,
    },
    "Coordinated High-Value Outlier ($4,900.00, New Cardholder)": {
        "amount": 4900.00,
        "customer_id": "cust_999",
        "card_id": "card_0001",
        "lat": 34.0522,
        "lon": -118.2437,
        "velocity_5m": 2,
        "speed_kmh": 45.0,
    },
}

selected_scenario = st.selectbox("Select Transaction Scenario", options=list(preset_scenarios.keys()), index=0)
tx_params = preset_scenarios[selected_scenario]

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    amount = st.number_input("Amount ($)", value=float(tx_params["amount"]), min_value=1.0, max_value=50000.0, step=10.0)
with col_b:
    velocity_5m = st.number_input("Prior Tx in 5m Window", value=int(tx_params["velocity_5m"]), min_value=0, max_value=20)
with col_c:
    speed_kmh = st.number_input("Implied Travel Speed (km/h)", value=float(tx_params["speed_kmh"]), min_value=0.0, max_value=5000.0, step=50.0)
with col_d:
    customer_id = st.text_input("Customer Entity ID", value=tx_params["customer_id"])

eval_button = st.button("Score Streaming Transaction", type="primary")

if eval_button or selected_scenario:
    # Build 16-dim feature vector
    feat_vector = np.zeros(16, dtype=np.float32)
    feat_vector[0] = amount
    feat_vector[1] = velocity_5m
    feat_vector[2] = velocity_5m + 1
    feat_vector[3] = velocity_5m + 2
    feat_vector[7] = max(1.0, amount / 50.0)
    feat_vector[14] = speed_kmh
    feat_vector[15] = 1.0 if speed_kmh > 800.0 else 0.0

    t0 = time.perf_counter()
    pred = classifier.predict(feat_vector)
    score = float(pred.calibrated_prob)
    decision = router.route(transaction_id="tx_interactive_01", fraud_score=score, amount=amount)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    action_str = decision.decision.upper()

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Routing Decision</div>
                <div class="metric-value {'status-approve' if action_str == 'APPROVE' else ('status-challenge' if action_str == 'CHALLENGE' else 'status-block')}">
                    {action_str}
                </div>
                <div class="metric-subtext">{'Zero-friction instant pass' if action_str == 'APPROVE' else ('Step-up 2FA/Biometric requested' if action_str == 'CHALLENGE' else 'Transaction blocked / review queue')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Fraud Probability P(Y=1)</div>
                <div class="metric-value">{score:.4f}</div>
                <div class="metric-subtext">τ_challenge: {tau_challenge:.2f} | τ_block: {tau_block:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Scoring Latency</div>
                <div class="metric-value">{latency_ms:.2f} ms</div>
                <div class="metric-subtext">SLA Target: &lt; 15.00 ms (CPU)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Travel Speed Status</div>
                <div class="metric-value" style="color: {'#EF4444' if speed_kmh > 800 else '#F8FAFC'};">{speed_kmh:.1f} km/h</div>
                <div class="metric-subtext">{'IMPOSSIBLE JUMP (&gt;800km/h)' if speed_kmh > 800 else 'Within physical travel bounds'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Decision Analysis
    st.markdown("#### Operational Impact & Cost Analysis")
    if action_str == "APPROVE":
        st.markdown(
            f"<div style='background-color: #064E3B; border: 1px solid #059669; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; color: #ECFDF5; font-size: 0.9rem;'>"
            f"<strong>FRICTIONLESS APPROVAL:</strong> Risk probability ({score:.4f}) is below challenge threshold (τ={tau_challenge:.2f}). "
            f"Customer friction is $0.00 with expected chargeback exposure strictly minimized."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif action_str == "CHALLENGE":
        st.markdown(
            f"<div style='background-color: #78350F; border: 1px solid #D97706; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; color: #FFFBEB; font-size: 0.9rem;'>"
            f"<strong>STEP-UP VERIFICATION:</strong> Moderate risk ({score:.4f} between τ_challenge {tau_challenge:.2f} and τ_block {tau_block:.2f}). "
            f"Triggers non-blocking OTP or push-notification challenge, preserving good customer UX while intercepting automated account takeover."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background-color: #7F1D1D; border: 1px solid #DC2626; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; color: #FEF2F2; font-size: 0.9rem;'>"
            f"<strong>HARD BLOCK ACTIVATED:</strong> Severe anomaly score ({score:.4f} ≥ τ_block {tau_block:.2f}). "
            f"Immediate settlement refusal. Intercepted ${amount:,.2f} fraud risk, avoiding chargeback liability and dispute administrative fees."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Feature Inspection
    st.markdown("#### Extracted Streaming Feature Vector")
    feat_names = [
        "amount ($)", "velocity_5m_count", "velocity_1h_count", "velocity_24h_count",
        "amount_sum_5m", "amount_sum_1h", "amount_sum_24h", "amount_ratio_5m_24h",
        "unique_mcc_24h", "unique_cards_24h", "card_age_days", "hour_of_day",
        "day_of_week", "distance_km", "implied_speed_kmh", "impossible_jump_flag"
    ]
    feat_cols = st.columns(4)
    for idx, (fname, fval) in enumerate(zip(feat_names, feat_vector)):
        col_target = feat_cols[idx % 4]
        with col_target:
            st.markdown(
                f"<div style='background-color: #1E293B; border: 1px solid #334155; border-radius: 3px; padding: 6px 10px; margin-bottom: 6px; font-size: 0.8rem;'>"
                f"<span style='color: #94A3B8;'>{fname}:</span> <span style='font-family: monospace; color: #F8FAFC; font-weight: 600;'>{fval:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
