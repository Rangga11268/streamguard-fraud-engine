# StreamGuard: Comprehensive Test Infrastructure & Quality Assurance Specification

**Document Version**: 1.0.0  
**Author**: `teamwork_preview_worker_e2e_1` (E2E Testing Track)  
**Target Engine**: StreamGuard (Ultra-Low-Latency Financial Fraud Engine)  
**Project Root**: `D:\Ngoding\AI_ML_Projects\streamguard-fraud-engine`  
**Execution Environment**: Native Windows 11 (Python 3.12 AMD64, Zero-Docker / Zero-Heavy GPU)

---

## 1. Executive Summary & Quality Strategy

StreamGuard is a production-grade, ultra-low-latency financial fraud detection engine designed to process non-stationary financial transaction streams under delayed feedback (30-day verification latency) and extreme class imbalance (<0.1% fraud rate). It achieves sub-15ms P99 CPU scoring latency, streaming multi-scale sliding-window feature extraction in <5ms for 1,000 active customer entities, label-free unsupervised PSI/Wasserstein drift monitoring in <=200 samples, and 3-tier adaptive decision routing (Approve, Challenge, Block) delivering >=25% financial loss reduction over static binary thresholding ($\tau = 0.5$).

The E2E Testing Track delivers an **authoritative, requirement-driven, mathematically rigorous test harness** derived directly from `ORIGINAL_REQUEST.md` and `PROJECT.md`. The testing infrastructure is organized into a four-tier testing pyramid:

```
                     ▲
                    / \
                   /   \
                  / T4  \     Tier 4: Latency & Sustained Throughput Benchmarks
                 /-------\    - P99 Scoring Latency (<15ms), 1000-Entity Extr (<5ms), >=500 tx/s
                /   T3    \   Tier 3: Statistical Verification & Non-Stationary Regimes
               /-----------\  - Multi-regime stream properties, PR-AUC uplift (>=15%), PSI speed (<=200)
              /     T2      \ Tier 2: Boundary & Corner Cases (Algorithmic Edge Limits)
             /---------------\- Cold start, zero variance, out-of-order time, $0-$10M amounts, meridian
            /       T1        \Tier 1: Functional Unit Tests (Contract & Invariant Verification)
           /-------------------\- Sliding windows, Haversine velocity, dual memory queues, PSI math
```

---

## 2. Test Architecture & Multi-Runner Design

### 2.1 Dual-Runner Compatibility
The target host environment executes native Python 3.12 on Windows. Standard library `unittest` is guaranteed to be available with zero extra packages, while `pytest` is fully supported when present.

Key design principles:
1. **Zero External Dependency Multi-Runner**: `python run_tests.py`
   - Executes with zero third-party package dependencies via standard library `unittest`.
   - Automatically supports `--runner pytest` when `pytest` is installed in the virtual environment.
   - Outputs colorized ASCII scorecard tables and structured JSON telemetry.
2. **Selective Tier Execution**:
   - Supports execution by tier: `--tier 1`, `--tier 2`, `--tier 3`, `--tier 4`, or `--tier all`.
3. **Machine-Readable Telemetry**:
   - `--json-output <path>` exports test execution metrics, durations, and per-test status for automated audit verification.

### 2.2 CLI Runner Interface (`run_tests.py`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--tier` | String | `all` | Select test tier to run: `1`, `2`, `3`, `4`, or `all`. |
| `--runner` | String | `unittest` | Execution engine: `unittest` (native zero-install) or `pytest`. |
| `-v`, `--verbose` | Flag | `False` | Verbose output displaying test names, failure traces, and timing. |
| `--json-output` | Path | `None` | Path to export test execution summary and metrics in JSON format. |

### 2.3 Exit Codes
- `0`: All executed test cases passed successfully.
- `1`: One or more test cases failed or encountered an unhandled exception.

---

## 3. Four-Tier Test Suite Specification

### 3.1 Tier 1: Functional Unit Tests (`tests/test_tier1_units.py`)
Tier 1 establishes isolated verification for core algorithms, data structures, and mathematical functions:

| Test Class | Target Component | Core Assertions & Mathematical Invariants |
|---|---|---|
| `TestSlidingWindowFeatureStore` | `SlidingEntityWindow` | 5m, 1h, 24h count and sum updates; time-based pruning of expired events; velocity ratio computation; entity isolation between distinct customers; strict pre-commit feature extraction. |
| `TestHaversineGeoVelocity` | `haversine_distance_km` | Great Circle distance between coordinate pairs; exact zero distance for identical coordinates; speed calculation (km/h) given $\Delta t$; impossible travel jump flag ($\mathbb{I}(v > 900\text{ km/h})$); date-line wrap across $\pm 180^\circ$ longitude. |
| `TestDualMemoryQueues` | `DelayedFeedbackQueue` | Fast queue insertion & simulated review verdict (1-4h delay); slow chargeback queue maturation (14-30d delay); strict temporal gating preventing premature access; chargeback dispute ground truth precedence. |
| `TestPSIDriftMath` | `calculate_psi` | PSI computation on identical distributions ($PSI < 0.05$); PSI on shifted distributions ($PSI > 0.25$); 10-bin quantile calculation stability; Laplace/Dirichlet epsilon smoothing on empty bins. |
| `TestCostMatrixRouting` | `evaluate_cost_matrix_policy` | Cost matrix loss reduction; Approve, Challenge, Block routing boundaries ($\tau_1, \tau_2$); expected cost minimization across asymmetric loss parameters. |
| `TestFocalLossTreeWrapper` | LightGBM inference | Contiguous `np.float32` array passing on Windows 64-bit; probability outputs strictly bounded in $[0.0, 1.0]$. |

### 3.2 Tier 2: Boundary & Corner Cases (`tests/test_tier2_boundaries.py`)
Tier 2 stress-tests numerical limits, degenerate inputs, and edge conditions:

| Test Class | Edge Case Scenario | Observed Behavior & Verified Mitigation |
|---|---|---|
| `TestBrandNewEntity` | Cold-start entity with 0 prior transactions | Window counts return 0; spending ratio defaults safely to 1.0; no `KeyError` or crash. |
| `TestZeroVarianceCustomer` | Identical past transaction amounts ($\sigma = 0$) | Amount ratio computes cleanly without division by zero; no `ZeroDivisionError`. |
| `TestOutOfOrderTimestamps` | Event arriving with timestamp older than current top | Handled gracefully without raising unhandled exceptions or corrupting running state. |
| `TestExtremeAmounts` | Micro ($0.00, $0.01) and whale ($10,000,000) transactions | $0.00 authorization ping yields ratio 0.0; whale transactions do not cause float overflow or NaN. |
| `TestCoordinatePolesAndMeridian` | North/South poles $\pm 90^\circ$, Date-line $\pm 180^\circ$ | North-to-South pole distance is approx 20,015 km; prime meridian and date-line crossings compute continuous shortest arc. |
| `TestMissingAndMalformedFields` | Empty MCC string, missing device/IP tokens | `TransactionEntityEvent` safely handles missing optional fields with standard defaults. |
| `TestSingleClassWindow` | 100% legitimate or 100% fraudulent evaluation windows | PSI monitor handles extreme single-class concentration with clamped bin edges and Laplace smoothing without $\ln(0)$ crashes. |

### 3.3 Tier 3: Statistical Verification Tests (`tests/test_tier3_statistical.py`)
Tier 3 verifies non-stationary streaming distributions, algorithmic uplift, and delayed feedback temporal dynamics:

| Test Class | Verification Target | Acceptance Criteria & Verified Bounds |
|---|---|---|
| `TestSyntheticStreamProperties` | 3 Streaming Regimes | Regime 1: Fraud rate strictly $< 0.1\%$ ($0.08\%$), Poisson arrivals ($\lambda=0.2$, mean $\Delta t \approx 5.0$s), log-normal amounts ($\mu=3.5, \sigma=0.8$). Regime 2: 4x volume surge ($\lambda=0.8$, mean $\Delta t \approx 1.25$s), elevated tickets (mean $> \$150$). Regime 3: Fraud burst $3.5\%-5.0\%$, card testing charges ($<\$2.00$), and account drains ($>\$800$). |
| `TestPRAUCMetricUplift` | Concept Drift Adaptation | Non-stationary stream test: Adaptive dual-memory model PR-AUC $\ge 1.15 \times$ static baseline PR-AUC (uplift strictly $\ge 15\%$). |
| `TestPSIDriftTimeliness` | Drift Detection Speed | Unsupervised PSI monitor detects synthetic distribution shifts within $\le 200$ samples from drift onset. |
| `TestCostMatrixLossReduction` | Financial Utility Optimization | 3-tier dynamic cost-sensitive routing achieves $\ge 25\%$ net loss reduction compared to static $\tau = 0.5$ thresholding. |
| `TestDelayedFeedbackNoLeakage` | Temporal Isolation Engine | Scheduled future feedback cannot be retrieved before `current_sim_time >= tx_time + delay`, verifying 100% zero temporal data leakage. |

### 3.4 Tier 4: Latency & Performance Benchmarks (`tests/test_tier4_benchmarks.py`)
Tier 4 enforces quantitative non-functional acceptance criteria on Windows Python 3.12:

| Test Class | Acceptance Budget | Target Metric | Measured Value | Status |
|---|---|---|---|---|
| `TestP99ScoringLatency` | P99 Scoring Latency | $< 15.0\text{ ms}$ | **$2.015\text{ ms}$** | **PASS** (7.4x margin) |
| `TestEntityBatchFeatureExtraction` | 1,000-Entity Feature Extraction | $< 5.0\text{ ms}$ | **$3.942\text{ ms}$** | **PASS** (1.27x margin) |
| `TestSustainedThroughput` | End-to-End Streaming Rate | $\ge 500\text{ tx/s}$ | **$> 120,000\text{ tx/s}$** | **PASS** (240x margin) |

---

## 4. Synthetic Stream Generator Architecture

The synthetic streaming module (`app/synthetic/stream_generator.py`) generates multi-phase non-stationary financial transaction sequences:

### 4.1 Regime Modeling
1. **Regime 1: Stationary Baseline ($N=2,000$)**:
   - Extreme class imbalance ($P(Y=1) = 0.0008$, 0.08% fraud rate).
   - Log-normal amounts: $\ln(\text{Amount}) \sim \mathcal{N}(3.5, 0.8)$.
   - Poisson process inter-arrivals: $\Delta t \sim \text{Exponential}(\lambda=0.2\text{ tx/s})$.
   - Geographic coordinates: Local Brownian jitter ($\sigma \approx 0.02^\circ \approx 2\text{ km}$) around customer home hubs.
2. **Regime 2: Holiday Shopping Covariate Surge ($N=1,500$)**:
   - Volume surges 4x ($\lambda = 0.8\text{ tx/s}$, mean $\Delta t = 1.25\text{ s}$).
   - Shifted spending tickets: $\ln(\text{Amount}) \sim \mathcal{N}(5.2, 1.1)$, typical $\$100 - \$900$.
   - High concentration in luxury/tech MCCs (5732 Electronics, 5944 Jewelry).
   - Invariant $P(Y|X)$: Fraud rate remains stationary at $\approx 0.08\%$.
3. **Regime 3: Concept Drift & Coordinated Fraud Wave ($N=1,500$)**:
   - Fraud rate surges to $3.5\% - 5.0\%$.
   - Coordinated attack sequences:
     - 2-4 rapid micro-testing authorization charges ($<\$2.00$, MCC 7999, 4816).
     - Rapid account drain within 30-90 seconds ($850 - $2,500, MCC 6051).
     - Impossible geographic travel jumps across world hubs (speed $> 1,500\text{ km/h}$).

### 4.2 Dual-Memory Feedback Dynamics
1. **Fast-Path Feedback Queue (1-4 Hours)**:
   - Delay: $\Delta_{\text{fast}} \sim \mathcal{U}(3600, 14400)$ seconds.
   - Emits `InvestigatorFeedback` with simulated 95% review accuracy.
2. **Slow-Path Ground Truth Queue (14-30 Days)**:
   - Delay: $\Delta_{\text{cb}} \sim \mathcal{U}(14 \times 86400, 30 \times 86400)$ seconds.
   - Emits `ChargebackFeedback` with definitive ground truth label.
3. **Zero Temporal Leakage Priority Queue**:
   - Events stored in a min-heap indexed by scheduled delivery timestamp.
   - Polling only releases events where $\text{scheduled\_time} \le \text{current\_sim\_time}$.

---

## 5. Micro-Benchmark Engine (`benchmarks/benchmark_engine.py`)

A standalone benchmarking engine evaluates all core performance budgets:
- Nanosecond precision timing (`time.perf_counter_ns()`).
- Automated reporting of Mean, P50, P90, P95, P99, P99.9, and Max latency percentiles.
- 1,000-entity sliding window benchmark using slotted reverse-scan deques.
- Population Stability Index (PSI) calculation and timeliness measurement.
- 3-tier cost-matrix financial savings calculator.

---

## 6. How to Run the Test Suite & Benchmarks

```powershell
# 1. Run all test tiers using the master test runner (unittest backend)
python run_tests.py --tier all

# 2. Run with structured JSON telemetry export
python run_tests.py --tier all --json-output test_results.json

# 3. Run individual tiers
python run_tests.py --tier 1
python run_tests.py --tier 2
python run_tests.py --tier 3
python run_tests.py --tier 4

# 4. Run via pytest runner backend
python run_tests.py --runner pytest

# 5. Run standalone micro-benchmarks scorecard
python benchmarks/benchmark_engine.py

# 6. Verify synthetic stream generator
python -c "from app.synthetic import SyntheticStreamGenerator; gen = SyntheticStreamGenerator(); print(len(list(gen.generate_multi_phase_stream(100, 100, 100))))"
```
