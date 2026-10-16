# StreamGuard: Developer & AI Agent Guide (AGENTS.md)

## Project Overview
**StreamGuard** is a production-grade, ultra-low-latency financial fraud detection engine designed for non-stationary transaction streams under delayed feedback (30-day verification latency) and extreme class imbalance (<0.1% fraud rate).

---

## Tech Stack & Architecture
- **Language & Runtime:** Python 3.12+ (100% Native Windows, Zero-Docker).
- **Core ML Framework:** LightGBM CPU with custom asymmetric cost-sensitive loss.
- **Probability Calibration:** Platt Sigmoid & Bayes Odds Adjustment.
- **Streaming Feature Store:** In-memory sliding-window entity deques (5m, 1h, 24h) + Haversine velocity.
- **Drift Detection:** Hybrid Dirichlet-smoothed Population Stability Index (PSI) & Wasserstein-1 continuous metric.
- **Decision Engine:** Expected cost-matrix optimization for 3-tier action routing (Approve, Challenge, Block).
- **Serving Layer:** Async FastAPI REST endpoints (`/v1/score`, `/v1/feedback/investigator`, `/v1/feedback/chargeback`, `/v1/telemetry/drift`).

---

## Essential Commands

```bash
# Run complete test suite across all 5 tiers
python run_tests.py --tier all

# Run adversarial stress suite
python run_tests.py --adversarial

# Run interactive 60-second live demonstration (5 acts)
python demo.py

# Launch FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# API docs available at http://127.0.0.1:8000/docs
```

---

## Architectural Boundaries & Non-Negotiables
1. **Zero-Docker Requirement:** 100% native execution on Windows Python 3.12 without Docker, WSL, or external job schedulers.
2. **Zero Temporal Lookahead Leakage:** Features must never access events from future timestamps.
3. **Delayed-Feedback Reality:** Unreviewed pending transactions must be debiased rather than treated as negative ground-truth.
4. **Latency Budget:** P99 scoring latency must remain strictly below 15ms on CPU (measured ~1.19ms).
5. **Testing Guarantee:** 100% pass rate across all tiers and adversarial probes.
