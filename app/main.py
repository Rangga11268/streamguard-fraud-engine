"""
StreamGuard Ultra-Low-Latency Financial Fraud Engine Main Application.

FastAPI application serving:
  - POST /v1/score
  - POST /v1/feedback/investigator
  - POST /v1/feedback/chargeback
  - GET  /v1/telemetry/drift
  - GET  /health
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import math

# Windows CPU multiprocessing safety
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from app.api.routes import router as api_v1_router
from app.config import StreamGuardConfig, default_config
from app.decision.cost_matrix import CostMatrixConfig
from app.decision.router import AdaptiveDecisionRouter
from app.drift.monitor import StreamingDriftMonitor
from app.features.feature_store import StreamingFeatureStore
from app.memory.dual_memory import DualMemoryManager
from app.models.classifier import FraudClassifier
from app.models.trainer import FraudModelTrainer, ModelTrainingConfig


class StreamGuardEngine:
    """Production service container wiring all StreamGuard analytical subsystems."""

    def __init__(self, config: Optional[StreamGuardConfig] = None) -> None:
        self.config: StreamGuardConfig = config or default_config
        self.feature_store: StreamingFeatureStore = StreamingFeatureStore(self.config)
        self.classifier: Optional[FraudClassifier] = None
        self.router: AdaptiveDecisionRouter = AdaptiveDecisionRouter(
            cost_config=CostMatrixConfig.from_config(self.config),
            default_tau_challenge=self.config.default_tau_challenge,
            default_tau_block=self.config.default_tau_block,
            mode="expected_loss",
        )
        self.drift_monitor: StreamingDriftMonitor = StreamingDriftMonitor(
            window_size=self.config.drift_window_size,
            reference_size=self.config.reference_window_size,
            psi_warning_threshold=self.config.psi_warning_threshold,
            psi_alert_threshold=self.config.psi_alert_threshold,
            initial_tau_challenge=self.config.default_tau_challenge,
            initial_tau_block=self.config.default_tau_block,
            auto_recalibrate=True,
        )
        self.dual_memory: DualMemoryManager = DualMemoryManager(
            config=self.config,
            maturity_horizon_sec=self.config.maturity_horizon_sec,
            working_window_sec=self.config.delayed_feedback_max_sec,
        )
        self.initialized: bool = False

    def initialize(self) -> None:
        """Establish baseline models, reference distributions, and recalibration hooks."""
        if self.initialized:
            return

        t0 = time.perf_counter()

        # 1. Train initial baseline LightGBM classifier with asymmetric loss on synthetic baseline
        rng = np.random.RandomState(42)
        n_baseline = 1000
        n_features = 16

        X_base = np.zeros((n_baseline, n_features), dtype=np.float32)
        y_base = np.zeros(n_baseline, dtype=np.int32)

        # Baseline legitimate distribution (typical amounts $20-$120, low velocity)
        X_base[:, 0] = np.exp(rng.normal(3.8, 0.7, size=n_baseline)).astype(np.float32)
        X_base[:, 1] = rng.poisson(0.3, size=n_baseline).astype(np.float32)
        X_base[:, 2] = rng.poisson(1.2, size=n_baseline).astype(np.float32)
        X_base[:, 3] = rng.poisson(3.5, size=n_baseline).astype(np.float32)
        X_base[:, 7] = (1.0 + rng.normal(0, 0.2, size=n_baseline)).astype(np.float32)
        X_base[:, 14] = np.maximum(0, rng.normal(5.0, 10.0, size=n_baseline)).astype(np.float32)
        X_base[:, 15] = 0.0

        # Baseline extreme class imbalance (~0.8% fraud)
        n_fraud = 8
        fraud_indices = rng.choice(n_baseline, size=n_fraud, replace=False)
        y_base[fraud_indices] = 1

        # Induce realistic signal for fraud baseline
        X_base[fraud_indices, 0] = np.exp(rng.normal(6.5, 0.8, size=n_fraud)).astype(np.float32)
        X_base[fraud_indices, 1] = rng.poisson(5.0, size=n_fraud).astype(np.float32)
        X_base[fraud_indices, 7] = (6.0 + rng.normal(0, 1.0, size=n_fraud)).astype(np.float32)
        X_base[fraud_indices, 14] = (1200.0 + rng.normal(0, 200.0, size=n_fraud)).astype(np.float32)
        X_base[fraud_indices, 15] = 1.0

        # Fit cost-sensitive LightGBM model
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
        training_result = trainer.train(X_base, y_base)
        self.classifier = training_result.classifier

        # 2. Establish drift monitor reference baseline scores and features
        predictions = self.classifier.predict_batch(X_base)
        reference_scores = [p.calibrated_prob for p in predictions]
        self.drift_monitor.set_baseline_scores(reference_scores)
        self.drift_monitor.set_baseline_features({
            "amount": X_base[:, 0],
            "velocity_5m_count": X_base[:, 1],
            "velocity_1h_count": X_base[:, 2],
        })

        # 3. Wire automatic dynamic threshold recalibration hook
        def on_drift_recalibration(telemetry, current_thresholds):
            # Tighten router thresholds when drift alert triggers
            self.router.recalibrate(drift_signal=telemetry.to_dict())
            return {
                "tau_challenge": self.router.tau_challenge,
                "tau_block": self.router.tau_block,
            }

        self.drift_monitor.register_recalibration_hook(on_drift_recalibration)
        self.initialized = True
        init_duration = round(time.perf_counter() - t0, 3)


import threading

# Windows CPU multiprocessing safety
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

_init_lock = threading.Lock()
_global_engine: Optional[StreamGuardEngine] = None


def get_global_engine() -> StreamGuardEngine:
    """Retrieve or initialize the global StreamGuardEngine singleton in a thread-safe manner."""
    global _global_engine
    if _global_engine is None or not _global_engine.initialized:
        with _init_lock:
            if _global_engine is None or not _global_engine.initialized:
                engine = StreamGuardEngine()
                engine.initialize()
                _global_engine = engine
    return _global_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan managing startup and shutdown."""
    engine = get_global_engine()
    app.state.engine = engine
    yield


def create_app() -> FastAPI:
    """FastAPI application factory for StreamGuard Fraud Engine."""
    application = FastAPI(
        title="StreamGuard Fraud Engine",
        description="Production-grade, ultra-low-latency financial fraud detection engine for non-stationary streams.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware for dashboards / local clients
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @application.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "ok",
            "service": "StreamGuard Fraud Engine",
            "version": "0.1.0",
            "timestamp": time.time(),
        }

    # Root metadata endpoint
    @application.get("/", tags=["Root"])
    async def root_index():
        return {
            "name": "StreamGuard Ultra-Low-Latency Fraud Engine",
            "version": "0.1.0",
            "documentation": "/docs",
            "endpoints": [
                "/v1/score",
                "/v1/feedback/investigator",
                "/v1/feedback/chargeback",
                "/v1/telemetry/drift",
                "/health",
            ],
        }

    # Ensure engine state is initialized on request if lifespan didn't fire (e.g. raw TestClient)
    @application.middleware("http")
    async def ensure_engine_initialized(request: Request, call_next):
        if not hasattr(request.app.state, "engine") or request.app.state.engine is None:
            request.app.state.engine = get_global_engine()
        return await call_next(request)

    # Register API v1 routes
    application.include_router(api_v1_router)

    # Sanitize RequestValidationError to ensure non-finite inputs (NaN, Inf) and exceptions are JSON-safe
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        from fastapi.encoders import jsonable_encoder
        raw_errors = jsonable_encoder(exc.errors())

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, float):
                if not math.isfinite(obj):
                    return str(obj)
                return obj
            elif isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        sanitized_errors = _sanitize(raw_errors)
        return JSONResponse(status_code=422, content={"detail": sanitized_errors})

    return application



# Top-level ASGI application instance
app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
