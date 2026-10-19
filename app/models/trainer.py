"""
Cost-Sensitive Fraud Model Training Pipeline for StreamGuard.

Designed for native Windows Python 3.12 execution:
  - Strictly enforces n_jobs=1 and os.environ["LOKY_MAX_CPU_COUNT"] = "4" to eliminate
    Windows joblib WinError 2 / core discovery deadlocks.
  - Strictly enforces contiguous np.float32 feature arrays and np.int32 label arrays.
  - Integrates asymmetric cost-sensitive loss (w_pos = C_FN / C_FP ≈ 200 - 500) and
    probability calibration under extreme class imbalance (<0.1% fraud).
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import auc, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Windows multiprocessing safety
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from app.config import FEATURE_NAMES, default_config
from app.models.calibration import ProbabilityCalibrator
from app.models.classifier import FraudClassifier
from app.models.loss import CostSensitiveAsymmetricLoss


@dataclass(slots=True)
class ModelTrainingConfig:
    """Hyperparameters and configuration settings for FraudModelTrainer."""
    n_estimators: int = 60
    max_depth: int = 5
    num_leaves: int = 20
    learning_rate: float = 0.05
    min_child_samples: int = 15
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    w_pos: float = 300.0
    gamma: float = 0.0
    calibration_method: str = "platt"
    val_split: float = 0.20
    time_based_split: bool = False  # Set True for chronological streaming logs to avoid temporal look-ahead leakage
    random_state: int = 42
    n_jobs: int = 1  # STRICT Windows single-process requirement
    verbose: int = -1
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))


@dataclass(slots=True)
class TrainingResult:
    """Summary of training metrics, validation results, and fitted classifier."""
    classifier: FraudClassifier
    train_pr_auc: float
    val_pr_auc: float
    val_roc_auc: float
    val_brier_score: float
    val_log_loss: float
    training_duration_sec: float
    n_train_samples: int
    n_val_samples: int
    fraud_rate_train: float
    fraud_rate_val: float
    evaluation_protocol: str = "stratified"
    calibration_evaluation_independent: bool = True
    is_pure_temporal: bool = False
    val_fpr: float = 0.0
    val_cost_loss: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    optimal_threshold: float = 0.5
    review_queue_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_pr_auc": round(self.train_pr_auc, 6),
            "val_pr_auc": round(self.val_pr_auc, 6),
            "val_roc_auc": round(self.val_roc_auc, 6),
            "val_brier_score": round(self.val_brier_score, 6),
            "val_log_loss": round(self.val_log_loss, 6),
            "val_fpr": round(self.val_fpr, 6),
            "val_cost_loss": round(self.val_cost_loss, 4),
            "val_precision": round(self.val_precision, 6),
            "val_recall": round(self.val_recall, 6),
            "val_f1": round(self.val_f1, 6),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "review_queue_rate": round(self.review_queue_rate, 6),
            "training_duration_sec": round(self.training_duration_sec, 4),
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "fraud_rate_train": round(self.fraud_rate_train, 6),
            "fraud_rate_val": round(self.fraud_rate_val, 6),
            "evaluation_protocol": self.evaluation_protocol,
            "calibration_evaluation_independent": self.calibration_evaluation_independent,
            "is_pure_temporal": self.is_pure_temporal,
        }


class FraudModelTrainer:
    """
    Automated training orchestrator for LightGBM fraud models.

    Handles data sanitization, cost-sensitive asymmetric objective configuration,
    model fitting with Windows joblib isolation, and probability calibration.
    """

    def __init__(self, config: Optional[ModelTrainingConfig] = None) -> None:
        self.config = config or ModelTrainingConfig()
        # Guarantee Windows environment settings
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"

    def _sanitize_inputs(
        self,
        X: Any,
        y: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs into contiguous float32 features and int32 labels."""
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        y_arr = np.ascontiguousarray(y, dtype=np.int32).ravel()

        if X_arr.ndim != 2:
            raise ValueError(f"Feature matrix X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[1] != len(self.config.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.config.feature_names)}, got {X_arr.shape[1]}"
            )
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Sample count mismatch: X has {len(X_arr)} rows, y has {len(y_arr)} elements")

        if np.isnan(X_arr).any():
            raise ValueError("Input feature matrix contains NaN values")
        if np.isinf(X_arr).any():
            raise ValueError("Input feature matrix contains Inf values")

        unique_labels = np.unique(y_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"Labels must be binary 0 or 1, found {unique_labels}")

        return X_arr, y_arr

    def train(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        time_based_split: Optional[bool] = None,
    ) -> TrainingResult:
        """
        Train the cost-sensitive LightGBM model and calibrate posterior probabilities.

        Separates calibration data from final independent evaluation data to guarantee
        unbiased Brier score, log loss, and PR-AUC assessment.

        Args:
            X: Feature matrix of shape (N, 16).
            y: Binary target labels (0=legitimate, 1=fraud).
            X_val: Optional validation feature matrix.
            y_val: Optional validation target labels.
            time_based_split: Optional boolean override for temporal splitting.

        Returns:
            TrainingResult dataclass containing the fitted FraudClassifier and metrics.
        """
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm library is not installed in the environment")

        t0 = time.perf_counter()
        X_clean, y_clean = self._sanitize_inputs(X, y)
        use_time_split = self.config.time_based_split if time_based_split is None else time_based_split
        is_pure_temporal = False
        eval_protocol = "temporal_walk_forward" if use_time_split else "stratified_random"

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together (both None or both non-None)")

        # Train / validation split
        if X_val is None and y_val is None:
            if use_time_split:
                # Temporal Walk-Forward split: train on earlier transactions, validate on future
                # Strictly eliminates look-ahead temporal data leakage in non-stationary fraud streams
                split_idx = int(len(X_clean) * (1.0 - self.config.val_split))
                X_tr, X_va = X_clean[:split_idx], X_clean[split_idx:]
                y_tr, y_va = y_clean[:split_idx], y_clean[split_idx:]
                if np.sum(y_va == 1) == 0:
                    warnings.warn(
                        f"Temporal validation slice (last {int(self.config.val_split*100)}%) contains 0 fraud instances. "
                        "Falling back to stratified random split to ensure positive fraud instances exist in evaluation set. "
                        "Evaluation marked as is_pure_temporal=False.",
                        UserWarning,
                    )
                    eval_protocol = "stratified_fallback_due_to_zero_temporal_fraud"
                    is_pure_temporal = False
                    n_pos = int(np.sum(y_clean == 1))
                    stratify = y_clean if n_pos >= 4 else None
                    X_tr, X_va, y_tr, y_va = train_test_split(
                        X_clean,
                        y_clean,
                        test_size=self.config.val_split,
                        random_state=self.config.random_state,
                        stratify=stratify,
                    )
                else:
                    is_pure_temporal = True
            else:
                is_pure_temporal = False
                n_pos = int(np.sum(y_clean == 1))
                stratify = y_clean if n_pos >= 4 else None
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_clean,
                    y_clean,
                    test_size=self.config.val_split,
                    random_state=self.config.random_state,
                    stratify=stratify,
                )
        else:
            X_tr, y_tr = X_clean, y_clean
            X_va, y_va = self._sanitize_inputs(X_val, y_val)
            eval_protocol = "external_validation_set"
            is_pure_temporal = bool(use_time_split)

        # Separate calibration set from independent evaluation set (P1 Evaluation Validity)
        # Calibrator is fitted strictly on (X_cal, y_cal), while Brier score, log loss, and PR-AUC
        # are measured on the unseen (X_eval, y_eval) test partition.
        n_pos_va = int(np.sum(y_va == 1))
        cal_eval_independent = True
        if n_pos_va >= 2 and len(y_va) >= 6:
            if is_pure_temporal:
                cal_idx = int(len(X_va) * 0.5)
                if np.sum(y_va[:cal_idx] == 1) > 0 and np.sum(y_va[cal_idx:] == 1) > 0:
                    X_cal, X_eval = X_va[:cal_idx], X_va[cal_idx:]
                    y_cal, y_eval = y_va[:cal_idx], y_va[cal_idx:]
                else:
                    # Shuffled partition within holdout
                    is_pure_temporal = False
                    warnings.warn(
                        "Temporal validation window lacked positive fraud instances in both sub-partitions. "
                        "Shuffled stratified split applied to validation set; evaluation marked is_pure_temporal=False.",
                        UserWarning,
                    )
                    X_cal, X_eval, y_cal, y_eval = train_test_split(
                        X_va, y_va, test_size=0.5, random_state=self.config.random_state, stratify=y_va
                    )
            else:
                X_cal, X_eval, y_cal, y_eval = train_test_split(
                    X_va, y_va, test_size=0.5, random_state=self.config.random_state, stratify=y_va
                )
        else:
            # Extreme imbalance / single fraud instance fallback:
            # Explicitly mark that calibration and evaluation are not independent.
            cal_eval_independent = False
            warnings.warn(
                f"Validation set has insufficient fraud instances ({n_pos_va} < 2) to cleanly partition into "
                "disjoint calibration and evaluation sets. Metrics are computed on the calibration set and marked non-independent.",
                UserWarning,
            )
            X_cal, X_eval = X_va, X_va
            y_cal, y_eval = y_va, y_va

        # Enforce contiguous array layout on split datasets for Windows LightGBM C API safety
        X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
        y_tr = np.ascontiguousarray(y_tr, dtype=np.int32)
        X_cal = np.ascontiguousarray(X_cal, dtype=np.float32)
        y_cal = np.ascontiguousarray(y_cal, dtype=np.int32)
        X_eval = np.ascontiguousarray(X_eval, dtype=np.float32)
        y_eval = np.ascontiguousarray(y_eval, dtype=np.int32)

        # Build custom asymmetric cost objective
        loss_obj = CostSensitiveAsymmetricLoss(
            w_pos=self.config.w_pos,
            gamma=self.config.gamma,
        )

        # Configure LightGBM classifier with strict Windows single-thread safety
        clf = lgb.LGBMClassifier(
            objective=loss_obj.sklearn_objective,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            n_jobs=1,  # STRICT Windows n_jobs=1
            verbose=-1,
            random_state=self.config.random_state,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            clf.fit(X_tr, y_tr)
            # CRITICAL ML INVARIANT: Use raw_score=True to get continuous decision margins (logits).
            # Avoids clf.predict() which collapses continuous predictions into binary [0, 1] class labels,
            # destroying ranking resolution and distorting Platt probability calibration.
            raw_tr = clf.predict(X_tr, raw_score=True)
            raw_cal = clf.predict(X_cal, raw_score=True)
            raw_eval = clf.predict(X_eval, raw_score=True)

        raw_tr = np.asarray(raw_tr, dtype=np.float64).ravel()
        raw_cal = np.asarray(raw_cal, dtype=np.float64).ravel()
        raw_eval = np.asarray(raw_eval, dtype=np.float64).ravel()

        # Fit probability calibrator on calibration partition using continuous decision margins
        calibrator = ProbabilityCalibrator(
            method=self.config.calibration_method,
            w_pos=self.config.w_pos,
        )
        calibrator.fit(raw_cal, y_cal)

        # Compute calibrated probabilities on independent evaluation partition
        eval_probs = np.asarray(calibrator.calibrate(raw_eval), dtype=np.float64)

        # Calculate evaluation metrics on independent evaluation partition
        # Train PR-AUC
        prec_tr, rec_tr, _ = precision_recall_curve(y_tr, raw_tr)
        train_pr_auc = float(auc(rec_tr, prec_tr))

        # Independent Val PR-AUC & ROC-AUC
        if len(np.unique(y_eval)) > 1:
            prec_va, rec_va, _ = precision_recall_curve(y_eval, raw_eval)
            val_pr_auc = float(auc(rec_va, prec_va))
            val_roc_auc = float(roc_auc_score(y_eval, raw_eval))
        else:
            val_pr_auc = float(np.mean(y_eval))
            val_roc_auc = 0.5

        # Brier Score & Log Loss on independent evaluation partition
        val_brier = float(brier_score_loss(y_eval, eval_probs))
        val_probs_clipped = np.clip(eval_probs, 1e-12, 1.0 - 1e-12)
        val_ll = float(log_loss(y_eval, val_probs_clipped, labels=[0, 1]))

        # Cost-optimal operational threshold selection
        threshold_candidates = np.linspace(0.01, 0.99, 99)
        best_cost = float("inf")
        best_tau = 0.5
        for tau in threshold_candidates:
            p_bin = (eval_probs >= tau).astype(np.int32)
            fp_c = int(np.sum((p_bin == 1) & (y_eval == 0)))
            fn_c = int(np.sum((p_bin == 0) & (y_eval == 1)))
            c_tot = float(5.0 * fp_c + (self.config.w_pos * 5.0) * fn_c)
            if c_tot < best_cost:
                best_cost = c_tot
                best_tau = float(tau)

        # Metrics at cost-optimal threshold
        opt_bin = (eval_probs >= best_tau).astype(np.int32)
        tp_count = int(np.sum((opt_bin == 1) & (y_eval == 1)))
        fp_opt = int(np.sum((opt_bin == 1) & (y_eval == 0)))
        fn_opt = int(np.sum((opt_bin == 0) & (y_eval == 1)))
        neg_count = int(np.sum(y_eval == 0))

        val_fpr = float(fp_opt / neg_count) if neg_count > 0 else 0.0
        val_precision = float(tp_count / (tp_count + fp_opt)) if (tp_count + fp_opt) > 0 else 0.0
        val_recall = float(tp_count / (tp_count + fn_opt)) if (tp_count + fn_opt) > 0 else 0.0
        val_f1 = float(2 * val_precision * val_recall / (val_precision + val_recall)) if (val_precision + val_recall) > 0 else 0.0
        review_queue_rate = float(np.mean(opt_bin))
        val_cost = best_cost

        t1 = time.perf_counter()

        classifier = FraudClassifier(
            booster=clf,
            calibrator=calibrator,
            feature_names=self.config.feature_names,
        )

        return TrainingResult(
            classifier=classifier,
            train_pr_auc=train_pr_auc,
            val_pr_auc=val_pr_auc,
            val_roc_auc=val_roc_auc,
            val_brier_score=val_brier,
            val_log_loss=val_ll,
            training_duration_sec=t1 - t0,
            n_train_samples=len(X_tr),
            n_val_samples=len(X_eval),
            fraud_rate_train=float(np.mean(y_tr)),
            fraud_rate_val=float(np.mean(y_eval)),
            evaluation_protocol=eval_protocol,
            calibration_evaluation_independent=cal_eval_independent,
            is_pure_temporal=is_pure_temporal,
            val_fpr=val_fpr,
            val_cost_loss=val_cost,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            optimal_threshold=best_tau,
            review_queue_rate=review_queue_rate,
        )

    def train_from_stream(
        self,
        events: Sequence[Any],
        feature_store: Optional[Any] = None,
        time_based_split: bool = True,
    ) -> TrainingResult:
        """
        Train directly from a chronological list of stream transaction events.

        Defaults to time_based_split=True to evaluate strictly on future transactions
        without look-ahead temporal leakage.
        """
        from app.features.feature_store import StreamingFeatureStore

        fs = feature_store or StreamingFeatureStore()
        vectors: List[np.ndarray] = []
        labels: List[int] = []

        for ev in events:
            # Extract features before committing
            extracted = fs.extract_features(ev, commit=True)
            vectors.append(extracted.feature_vector)
            labels.append(int(getattr(ev, "true_label", 0)))

        X = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
        y = np.ascontiguousarray(labels, dtype=np.int32)
        return self.train(X, y, time_based_split=time_based_split)

