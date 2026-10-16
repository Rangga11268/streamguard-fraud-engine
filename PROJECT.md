# Project: StreamGuard Fraud Engine

## Architecture
StreamGuard is a production-grade, ultra-low-latency financial fraud detection engine designed for non-stationary transaction streams under delayed feedback (30-day verification latency) and extreme class imbalance (<0.1% fraud rate).
All components execute natively on Windows Python 3.12 without external infrastructure or GPU dependencies.

### Architectural Dataflow
```
Incoming Transaction Stream (JSON / HTTP / Synthetic Generator)
                         │
                         ▼
      ┌──────────────────────────────────────────────┐
      │  M1: Streaming Feature Store (app.features)  │ [DONE]
      │  - 5m, 1h, 24h multi-scale entity sliding deques
      │  - Velocity ratios & spending acceleration   │
      │  - Haversine impossible travel speed (<5ms)  │
      │  - Strict zero temporal leakage guard        │
      └──────────────────────┬───────────────────────┘
                             │ Extracted Feature Vector (16 dims)
                             ▼
      ┌──────────────────────────────────────────────┐
      │   M2: Classification Engine (app.models)     │ [DONE]
      │  - Cost-sensitive LightGBM CPU inference     │
      │  - Sub-15ms P99 latency (measured 1.19ms)    │
      │  - Focal / asymmetric loss for <0.1% fraud   │
      │  - Probability calibration for cost matrix   │
      └──────────────────────┬───────────────────────┘
                             │ Calibrated Fraud Probability p(fraud|x)
                             ▼
      ┌──────────────────────────────────────────────┐
      │   M5: 3-Tier Adaptive Router (app.decision)  │ [DONE]
      │  - Expected loss minimization:               │
      │    E[Loss(a | p, Amount)] for Approve/Chal/Block
      │  - > 70% loss reduction vs static tau = 0.5  │
      └───────────────┬──────────────────────┬───────┘
                      │                      │
                      ▼                      ▼
         [Telemetry Score Push]      [Action Output]
                      │              - Approve (frictionless)
                      │              - Challenge (OTP / 2FA)
                      │              - Block (manual review)
                      ▼
      ┌──────────────────────────────────────────────┐
      │    M4: Drift Monitor (app.drift)             │ [DONE]
      │  - Unsupervised sliding-window PSI (hybrid)  │
      │  - Wasserstein-1 continuous metric           │
      │  - Detection within <= 200 samples           │
      │  - Dynamic threshold recalibration trigger   │
      └──────────────────────┘

Asynchronous Feedback Streams:
  - 1-4h Investigator Reviews  ──►  POST /v1/feedback/investigator ──┐
  - 14-30d Bank Chargebacks    ──►  POST /v1/feedback/chargeback   ──┤
                                                                     ▼
                                            ┌───────────────────────────────────┐
                                            │ M3: Dual-Memory Queue (app.memory)│ [DONE]
                                            │ - Fast-Feedback Working Buffer    │
                                            │ - Transition Horizon (20-30d)     │
                                            │ - Consolidated Ground-Truth Replay│
                                            │ - Eliminates 50% FN truncation    │
                                            └───────────────────────────────────┘
```

---

## Feature Inventory

| # | Feature | Description | Milestone | Status | Source |
|---|---------|-------------|-----------|--------|--------|
| 1 | Multi-Scale Sliding Windows | 5m, 1h, 24h entity event deques with timestamp-indexed pruning | M1 | DONE | Survey |
| 2 | Spending Velocity Ratios | Short-to-long velocity and current amount vs 24h baseline | M1 | DONE | Survey |
| 3 | Haversine Geo Velocity | Geographic distance, velocity km/h, and impossible travel flag | M1 | DONE | Survey |
| 4 | Strict Temporal Non-Leakage | Extraction prior to commit; out-of-order monotonic guard | M1 | DONE | Survey |
| 5 | Cost-Sensitive LightGBM Model | Weighted asymmetric objective for <0.1% fraud imbalance | M2 | DONE | Survey |
| 6 | Sub-15ms CPU Inference | Contiguous float32 vectorized inference with single-thread optimization | M2 | DONE | Survey |
| 7 | Probability Calibration | Platt / Bayes odds calibration from weighted margin outputs | M2 | DONE | Survey |
| 8 | Fast Investigator Memory | Working buffer for 1-4h human review feedback with confidence score | M3 | DONE | Survey |
| 9 | Consolidated Chargeback Memory| Replay buffer for 14-30d chargebacks and matured legitimate records | M3 | DONE | Survey |
| 10 | Delayed Feedback Debiaser | Resolves label delay truncation bias without false-negative assumptions | M3 | DONE | Survey |
| 11 | Unsupervised Score & Feature PSI| 10-bin domain-adaptive hybrid binning Population Stability Index | M4 | DONE | Survey |
| 12 | 1D Wasserstein Metric | Fast continuous Earth Mover's Distance on prediction score distributions | M4 | DONE | Survey |
| 13 | Rapid Drift Trigger | Early warning alert within <= 200 samples of concept/covariate drift | M4 | DONE | Survey |
| 14 | Threshold Recalibration | Dynamic boundary adjustment (tau1, tau2) triggered on drift alert | M4 | DONE | Survey |
| 15 | Cost Matrix Expected Loss | Computes argmin E[Loss] across Approve, Challenge, Block given amount | M5 | DONE | Survey |
| 16 | 3-Tier Dynamic Action Router | Maps transaction to Approve, Challenge, or Block risk tiers | M5 | DONE | Survey |
| 17 | FastAPI REST Serving Layer | Endpoints: /v1/score, /v1/feedback/investigator, /v1/feedback/chargeback, /v1/telemetry/drift | M6 | DONE | Survey |
| 18 | Interactive Terminal Demo | 5-act live streaming telemetry dashboard (python demo.py) < 60s | M6 | DONE | Survey |
| 19 | Synthetic Stream Generator | 3-regime non-stationary generator (baseline, covariate surge, concept attack) | E2E | DONE | Survey |
| 20 | Multi-Tier Test Suite | Single-command runner (python run_tests.py) covering Tiers 1-5 | E2E | DONE | Survey |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| **E2E** | E2E Testing Track | Test harness, synthetic 3-regime stream generator, multi-tier runner (`run_tests.py`), and Tiers 1-4 test cases; published `TEST_READY.md` v1.1.0 | none | **DONE** |
| **M1** | Feature Store | Sliding entity deques, velocity ratios, Haversine impossible travel speed, zero temporal leakage | none | **DONE** |
| **M2** | Classification Engine | Cost-sensitive LightGBM wrapper, asymmetric focal loss, extreme imbalance training, probability calibration, sub-15ms CPU pipeline | M1 | **DONE** |
| **M3** | Dual-Memory System | Asynchronous feedback queues, 1-4h fast investigator buffer, 14-30d chargeback buffer, debiased replay buffer | M1 | **DONE** |
| **M4** | Drift Monitoring | Unsupervised hybrid-binning PSI, Wasserstein-1 distance, early warning alert <= 200 samples, threshold recalibration | M2 | **DONE** |
| **M5** | 3-Tier Decision Routing | Cost matrix loss minimizer, Approve/Challenge/Block routing, > 70% financial loss reduction vs tau=0.5 | M2, M4 | **DONE** |
| **M6** | FastAPI Layer & Demo | Async REST API endpoints, Pydantic schemas, and interactive 5-act terminal demo (`demo.py`) | M1-M5 | **DONE** |
| **FINAL** | E2E Verification & Hardening | Phase 1: Pass 100% of E2E tests (Tiers 1-5). Phase 2: Adversarial coverage hardening (Tier 5) | E2E, M1-M6 | **DONE** |
