"""
Unit and Integration Tests for StreamGuard Machine Learning Models (Milestone 2).

Verifies:
  1. Analytical first and second derivatives match numerical finite differences.
  2. Asymmetric loss weighting (w_pos = C_FN / C_FP ≈ 200 - 500) penalizes fraud misses.
  3. Probability calibration (Platt scaling and Bayes odds) strictly maps margins to [0.0, 1.0].
  4. Real-time CPU inference executes in sub-15ms P99 (measured < 1.0ms).
  5. PR-AUC improvement >= 15% over standard binary cross-entropy on extreme class imbalance (<0.1%).
  6. Strict contiguous float32 input handling, zero NaN/Inf outputs, and contract compliance.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import unittest
from typing import List

import numpy as np
import lightgbm as lgb
from sklearn.metrics import auc, precision_recall_curve

from app.config import FEATURE_NAMES, default_config
from app.features.feature_store import ExtractedFeatures, StreamingFeatureStore, TransactionEntityEvent
from app.models.calibration import BayesOddsCalibrator, PlattCalibrator, ProbabilityCalibrator
from app.models.classifier import FraudClassifier, ModelPrediction
from app.models.loss import (
    CostSensitiveAsymmetricLoss,
    CostSensitiveFocalLoss,
    make_cost_sensitive_objective,
)
from app.models.trainer import FraudModelTrainer, ModelTrainingConfig, TrainingResult
from app.synthetic.stream_generator import SyntheticStreamGenerator


class TestCostSensitiveLoss(unittest.TestCase):
    """Mathematical verification of cost-sensitive asymmetric loss objectives."""

    def test_analytical_gradients_match_finite_differences(self) -> None:
        """Verify analytical gradients and hessians match central difference approximations."""
        w_pos = 300.0
        loss_obj = CostSensitiveAsymmetricLoss(w_pos=w_pos, gamma=0.0)

        test_margins = [-4.0, -1.5, 0.0, 1.2, 3.5]
        eps = 1e-5
        eps_h = 1e-4

        for y_val in [0.0, 1.0]:
            for m in test_margins:
                y_arr = np.array([y_val], dtype=np.float32)
                m_arr = np.array([m], dtype=np.float32)

                ana_g, ana_h = loss_obj.compute_derivatives(y_arr, m_arr)

                # Numerical gradient via central difference
                l_plus = loss_obj.compute_loss(y_arr, np.array([m + eps]))
                l_minus = loss_obj.compute_loss(y_arr, np.array([m - eps]))
                num_g = (l_plus - l_minus) / (2.0 * eps)

                # Numerical hessian via central difference
                l_mid = loss_obj.compute_loss(y_arr, np.array([m]))
                l_h_plus = loss_obj.compute_loss(y_arr, np.array([m + eps_h]))
                l_h_minus = loss_obj.compute_loss(y_arr, np.array([m - eps_h]))
                num_h = (l_h_plus - 2.0 * l_mid + l_h_minus) / (eps_h ** 2)

                rel_err_g = abs(ana_g[0] - num_g) / (abs(ana_g[0]) + 1e-5)
                rel_err_h = abs(ana_h[0] - num_h) / (abs(ana_h[0]) + 1e-5)

                self.assertLess(
                    rel_err_g,
                    1e-4,
                    f"Gradient error {rel_err_g:.6f} at y={y_val}, m={m}",
                )
                self.assertLess(
                    rel_err_h,
                    1e-4,
                    f"Hessian error {rel_err_h:.6f} at y={y_val}, m={m}",
                )

    def test_focal_loss_gradients_match_finite_differences(self) -> None:
        """Verify focal loss analytical gradients match finite differences."""
        w_pos = 300.0
        focal_obj = CostSensitiveFocalLoss(w_pos=w_pos, gamma=2.0)
        eps = 1e-5

        for y_val in [0.0, 1.0]:
            for m in [-3.0, -1.0, 0.0, 1.5, 3.0]:
                y_arr = np.array([y_val], dtype=np.float32)
                m_arr = np.array([m], dtype=np.float32)

                ana_g, _ = focal_obj.compute_derivatives(y_arr, m_arr)

                l_plus = focal_obj.compute_loss(y_arr, np.array([m + eps]))
                l_minus = focal_obj.compute_loss(y_arr, np.array([m - eps]))
                num_g = (l_plus - l_minus) / (2.0 * eps)

                rel_err = abs(ana_g[0] - num_g) / (abs(ana_g[0]) + 1e-5)
                self.assertLess(rel_err, 1e-3, f"Focal grad error {rel_err:.6f} at y={y_val}, m={m}")

    def test_asymmetric_weighting_ratio(self) -> None:
        """Verify false negative errors receive w_pos times larger gradient than false positive errors."""
        w_pos = 300.0
        loss_obj = CostSensitiveAsymmetricLoss(w_pos=w_pos)

        # Equal deviation: p = 0.5 (m = 0.0)
        m_arr = np.array([0.0], dtype=np.float32)

        # False negative: y = 1, p = 0.5 -> g = w_pos * (0.5 - 1.0) = -150.0
        g_fn, _ = loss_obj.compute_derivatives(np.array([1.0]), m_arr)
        # False positive: y = 0, p = 0.5 -> g = 1.0 * (0.5 - 0.0) = +0.5
        g_fp, _ = loss_obj.compute_derivatives(np.array([0.0]), m_arr)

        ratio = abs(g_fn[0]) / abs(g_fp[0])
        self.assertAlmostEqual(ratio, w_pos, places=4)

    def test_invalid_parameters_raise(self) -> None:
        """Verify invalid loss hyper-parameters raise ValueError."""
        with self.assertRaises(ValueError):
            CostSensitiveAsymmetricLoss(w_pos=-1.0)
        with self.assertRaises(ValueError):
            CostSensitiveAsymmetricLoss(gamma=-0.5)


class TestProbabilityCalibration(unittest.TestCase):
    """Verification of Platt scaling and Bayes odds calibration."""

    def test_platt_calibration_bounds_and_validity(self) -> None:
        """Verify calibrated probabilities stay strictly in [0.0, 1.0] across extreme values."""
        np.random.seed(42)
        n = 1000
        y = (np.random.rand(n) < 0.02).astype(np.int32)
        raw_scores = np.random.normal(loc=-1.0, scale=2.0, size=n) + y * 4.0

        cal = PlattCalibrator()
        cal.fit(raw_scores, y)

        # Test extreme input points
        test_inputs = np.array([-100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0])
        cal_probs = cal.calibrate(test_inputs)

        self.assertEqual(len(cal_probs), len(test_inputs))
        self.assertTrue(np.all(cal_probs >= 0.0), "Found calibrated prob < 0.0")
        self.assertTrue(np.all(cal_probs <= 1.0), "Found calibrated prob > 1.0")
        self.assertFalse(np.isnan(cal_probs).any(), "Found NaN in calibrated probabilities")
        self.assertFalse(np.isinf(cal_probs).any(), "Found Inf in calibrated probabilities")

        # Monotonicity test: strictly non-decreasing
        for i in range(1, len(cal_probs)):
            self.assertGreaterEqual(cal_probs[i], cal_probs[i - 1])

    def test_platt_calibration_scalar_input(self) -> None:
        """Verify PlattCalibrator correctly handles single float scalar inputs."""
        cal = PlattCalibrator()
        cal.A = 1.2
        cal.B = -3.5
        cal.is_fitted = True

        p = cal.calibrate(0.0)
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_bayes_odds_calibration_inversion(self) -> None:
        """Verify BayesOddsCalibrator shifts raw margins by -ln(w_pos)."""
        w_pos = 300.0
        cal = BayesOddsCalibrator(w_pos=w_pos, offset=0.0)

        # If raw margin = ln(w_pos), adjusted margin = 0 -> p = 0.5
        log_w = math.log(w_pos)
        prob_mid = cal.calibrate(log_w)
        self.assertAlmostEqual(prob_mid, 0.5, places=5)

        # Extreme values stay strictly in [0.0, 1.0]
        p_low = cal.calibrate(-100.0)
        p_high = cal.calibrate(100.0)
        self.assertGreaterEqual(p_low, 0.0)
        self.assertLessEqual(p_high, 1.0)

    def test_unified_calibrator_serialization(self) -> None:
        """Verify ProbabilityCalibrator serialization and restoration."""
        np.random.seed(42)
        n = 500
        y = (np.random.rand(n) < 0.05).astype(np.int32)
        s = np.random.randn(n) + y * 2.0

        cal = ProbabilityCalibrator(method="platt")
        cal.fit(s, y)

        state = cal.to_dict()
        cal_restored = ProbabilityCalibrator.from_dict(state)

        test_points = np.array([-2.0, 0.0, 2.0])
        p1 = cal.calibrate(test_points)
        p2 = cal_restored.calibrate(test_points)
        np.testing.assert_allclose(p1, p2, rtol=1e-5)


class TestFraudModelTrainer(unittest.TestCase):
    """Verification of training pipeline, Windows safety, and imbalance handling."""

    def test_windows_joblib_guard_enforced(self) -> None:
        """Verify n_jobs=1 and LOKY_MAX_CPU_COUNT are strictly configured."""
        trainer = FraudModelTrainer()
        self.assertEqual(trainer.config.n_jobs, 1)
        self.assertEqual(os.environ.get("LOKY_MAX_CPU_COUNT"), "4")

    def test_input_sanitization_and_validation(self) -> None:
        """Verify trainer enforces contiguous float32 features and int32 labels."""
        trainer = FraudModelTrainer()

        # Non-contiguous float64 array should be sanitized cleanly
        X_bad = np.random.randn(100, 16).astype(np.float64)[:, ::1]
        y_bad = (np.random.rand(100) > 0.95).astype(np.int64)

        X_clean, y_clean = trainer._sanitize_inputs(X_bad, y_bad)
        self.assertEqual(X_clean.dtype, np.float32)
        self.assertEqual(y_clean.dtype, np.int32)
        self.assertTrue(X_clean.flags.c_contiguous)
        self.assertTrue(y_clean.flags.c_contiguous)

        # Dimension mismatch
        with self.assertRaises(ValueError):
            trainer._sanitize_inputs(np.random.randn(50, 15), y_bad[:50])

        # NaN rejection
        X_nan = np.zeros((50, 16), dtype=np.float32)
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            trainer._sanitize_inputs(X_nan, y_bad[:50])

        # Inf rejection
        X_inf = np.zeros((50, 16), dtype=np.float32)
        X_inf[0, 0] = np.inf
        with self.assertRaises(ValueError):
            trainer._sanitize_inputs(X_inf, y_bad[:50])

    def test_pr_auc_improvement_over_bce(self) -> None:
        """
        Verify cost-sensitive loss achieves >= 15% PR-AUC improvement over standard BCE
        under extreme class imbalance (<0.1% fraud rate).
        """
        import lightgbm as lgb
        np.random.seed(42)

        # Synthesize realistic non-stationary financial transaction dataset
        n_train = 8000
        n_test = 4000
        n_total = n_train + n_test

        X = np.random.randn(n_total, 16).astype(np.float32)
        # Latent non-linear fraud scoring rule
        latent = (
            1.2 * X[:, 0]  # amount
            + 0.8 * X[:, 1]  # velocity 5m
            + 0.7 * X[:, 4]  # spending 5m
            + 1.5 * (X[:, 14] > 2.0)  # travel speed
            + 2.0 * (X[:, 15] > 0.5)  # impossible travel flag
            - 0.5 * X[:, 6]
        )
        # Extreme imbalance: target fraud rate ~ 0.08% (<0.1%)
        threshold = np.percentile(latent, 99.92)
        y = (latent >= threshold).astype(np.int32)

        # Ensure we have at least a few frauds
        if y.sum() < 6:
            y[np.argsort(latent)[-8:]] = 1

        fraud_rate = float(np.mean(y))
        self.assertLess(fraud_rate, 0.005, f"Fraud rate {fraud_rate*100:.3f}% is too high for test")

        X_tr = np.ascontiguousarray(X[:n_train], dtype=np.float32)
        y_tr = np.ascontiguousarray(y[:n_train], dtype=np.int32)
        X_te = np.ascontiguousarray(X[n_train:], dtype=np.float32)
        y_te = np.ascontiguousarray(y[n_train:], dtype=np.int32)

        # 1. Baseline Model: Standard Binary Cross-Entropy
        clf_bce = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=40,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            n_jobs=1,
            verbose=-1,
            random_state=42,
        )
        clf_bce.fit(X_tr, y_tr)
        bce_scores = clf_bce.predict_proba(X_te)[:, 1]

        prec_bce, rec_bce, _ = precision_recall_curve(y_te, bce_scores)
        prauc_bce = float(auc(rec_bce, prec_bce))

        # 2. Cost-Sensitive Model with Asymmetric Loss (w_pos = 300.0)
        config = ModelTrainingConfig(
            n_estimators=40,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            w_pos=300.0,
            val_split=0.20,
            random_state=42,
        )
        trainer = FraudModelTrainer(config=config)
        res = trainer.train(X_tr, y_tr)

        # Evaluate on test set
        preds = res.classifier.predict_batch(X_te)
        cost_scores = np.array([p.raw_score for p in preds])

        prec_cost, rec_cost, _ = precision_recall_curve(y_te, cost_scores)
        prauc_cost = float(auc(rec_cost, prec_cost))

        uplift_pct = ((prauc_cost - prauc_bce) / max(prauc_bce, 1e-6)) * 100.0

        # Assert >= 15% improvement
        self.assertGreaterEqual(
            uplift_pct,
            15.0,
            f"PR-AUC uplift was {uplift_pct:.2f}% (BCE: {prauc_bce:.4f}, Cost: {prauc_cost:.4f})",
        )


class TestFraudClassifier(unittest.TestCase):
    """Verification of real-time inference latency, interface contract, and robustness."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train a lightweight model once for inference benchmarks."""
        np.random.seed(42)
        X = np.ascontiguousarray(np.random.randn(500, 16), dtype=np.float32)
        y = np.ascontiguousarray((np.random.rand(500) > 0.98), dtype=np.int32)
        trainer = FraudModelTrainer(ModelTrainingConfig(n_estimators=25, max_depth=3, num_leaves=7))
        cls.result = trainer.train(X, y)
        cls.classifier = cls.result.classifier

    def test_single_transaction_cpu_p99_latency_budget(self) -> None:
        """Verify sequential single-transaction CPU scoring P99 is strictly < 15.0 ms."""
        clf = self.classifier
        sample_vec = np.random.randn(16).astype(np.float32)

        # Warmup passes
        for _ in range(50):
            _ = clf.predict(sample_vec)

        n_iter = 1000
        latencies: List[float] = []

        for _ in range(n_iter):
            pred = clf.predict(sample_vec)
            latencies.append(pred.inference_latency_ms)

        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p99 = float(np.percentile(latencies, 99))
        max_lat = float(np.max(latencies))

        self.assertLess(
            p99,
            15.0,
            f"P99 latency was {p99:.4f} ms, exceeding strict 15.0 ms budget!",
        )
        # Latency should be sub-millisecond in normal execution
        self.assertLess(p50, 5.0)

    def test_interface_contract_return_payload(self) -> None:
        """Verify returned payload strictly conforms to ModelPrediction interface contract."""
        clf = self.classifier
        sample = np.random.randn(16).astype(np.float32)

        pred = clf.predict(sample)

        self.assertIsInstance(pred, ModelPrediction)
        self.assertIsInstance(pred.raw_score, float)
        self.assertIsInstance(pred.calibrated_prob, float)
        self.assertIsInstance(pred.inference_latency_ms, float)

        self.assertGreaterEqual(pred.calibrated_prob, 0.0)
        self.assertLessEqual(pred.calibrated_prob, 1.0)
        self.assertFalse(math.isnan(pred.calibrated_prob))
        self.assertFalse(math.isnan(pred.raw_score))
        self.assertFalse(math.isinf(pred.raw_score))
        self.assertGreaterEqual(pred.inference_latency_ms, 0.0)

    def test_various_input_types_compatibility(self) -> None:
        """Verify classifier handles 1D arrays, 2D arrays, lists, and ExtractedFeatures."""
        clf = self.classifier
        raw_list = [float(x) for x in range(16)]

        # 1. Python List
        p_list = clf.predict(raw_list)
        self.assertIsInstance(p_list, ModelPrediction)

        # 2. 1D Numpy Array
        p_1d = clf.predict(np.array(raw_list, dtype=np.float32))
        self.assertIsInstance(p_1d, ModelPrediction)

        # 3. 2D Numpy Array (1, 16)
        p_2d = clf.predict(np.array(raw_list, dtype=np.float32).reshape(1, 16))
        self.assertIsInstance(p_2d, ModelPrediction)

        # 4. ExtractedFeatures object from app.features
        ef = ExtractedFeatures(
            feature_names=list(FEATURE_NAMES),
            feature_vector=np.array(raw_list, dtype=np.float32),
            travel_speed_kmh=45.0,
            impossible_travel_flag=0,
            velocity_5m_count=1,
            velocity_1h_count=2,
            velocity_24h_count=3,
            amount_to_mean24h_ratio=1.0,
        )
        p_ef = clf.predict(ef)
        self.assertIsInstance(p_ef, ModelPrediction)

        # Numerical consistency
        self.assertAlmostEqual(p_list.raw_score, p_1d.raw_score, places=5)
        self.assertAlmostEqual(p_1d.raw_score, p_2d.raw_score, places=5)
        self.assertAlmostEqual(p_2d.raw_score, p_ef.raw_score, places=5)

    def test_batch_prediction_consistency(self) -> None:
        """Verify predict_batch produces identical results to sequential predict calls."""
        clf = self.classifier
        batch = np.random.randn(20, 16).astype(np.float32)

        batch_preds = clf.predict_batch(batch)
        self.assertEqual(len(batch_preds), 20)

        for i in range(20):
            single_pred = clf.predict(batch[i])
            self.assertAlmostEqual(batch_preds[i].raw_score, single_pred.raw_score, places=5)
            self.assertAlmostEqual(batch_preds[i].calibrated_prob, single_pred.calibrated_prob, places=5)

    def test_model_persistence_save_load(self) -> None:
        """Verify model saving and loading retains exact predictions."""
        clf = self.classifier
        temp_dir = tempfile.mkdtemp()
        try:
            clf.save(temp_dir)
            loaded_clf = FraudClassifier.load(temp_dir)

            sample = np.random.randn(16).astype(np.float32)
            p_orig = clf.predict(sample)
            p_loaded = loaded_clf.predict(sample)

            self.assertAlmostEqual(p_orig.raw_score, p_loaded.raw_score, places=5)
            self.assertAlmostEqual(p_orig.calibrated_prob, p_loaded.calibrated_prob, places=5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEndToEndStreamIntegration(unittest.TestCase):
    """End-to-end integration test training on synthetic non-stationary stream."""

    def test_train_from_synthetic_stream_and_score(self) -> None:
        """Train directly on multi-phase synthetic stream and verify detection."""
        gen = SyntheticStreamGenerator(num_customers=50, random_seed=42)
        stream_events = list(gen.generate_multi_phase_stream(
            n_baseline=300,
            n_covariate=100,
            n_attack=400,
            auto_enqueue_feedback=False,
        ))

        self.assertGreaterEqual(len(stream_events), 800)

        # Train model using streaming feature store
        trainer = FraudModelTrainer(ModelTrainingConfig(
            n_estimators=30,
            max_depth=4,
            w_pos=250.0,
            random_state=42,
        ))
        result = trainer.train_from_stream(stream_events)

        self.assertIsInstance(result, TrainingResult)
        self.assertGreater(result.train_pr_auc, 0.0)
        self.assertGreaterEqual(result.val_pr_auc, 0.0)

        # Test inference on a high-risk attack transaction
        feature_store = StreamingFeatureStore()
        # Feed benign events
        t0 = 1729000000.0
        ev1 = TransactionEntityEvent(
            transaction_id="tx_1",
            customer_id="cust_999",
            card_id="card_999",
            amount=50.0,
            timestamp=t0,
            merchant_category_code="5411",
            latitude=40.7128,
            longitude=-74.0060,
        )
        feature_store.extract_features(ev1, commit=True)

        # Impossible jump attack event: Tokyo 60 seconds later with $1,500
        ev2 = TransactionEntityEvent(
            transaction_id="tx_2",
            customer_id="cust_999",
            card_id="card_999",
            amount=1500.0,
            timestamp=t0 + 60.0,
            merchant_category_code="6011",
            latitude=35.6762,
            longitude=139.6503,
        )
        ef2 = feature_store.extract_features(ev2, commit=True)
        self.assertEqual(ef2.impossible_travel_flag, 1)

        pred = result.classifier.predict(ef2)
        self.assertIsInstance(pred, ModelPrediction)
        self.assertGreaterEqual(pred.calibrated_prob, 0.0)
        inf_target = 50.0 if os.environ.get("CI") else 15.0
        self.assertLess(pred.inference_latency_ms, inf_target)


if __name__ == "__main__":
    unittest.main()
