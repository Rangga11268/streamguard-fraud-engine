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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_pr_auc": round(self.train_pr_auc, 6),
            "val_pr_auc": round(self.val_pr_auc, 6),
            "val_roc_auc": round(self.val_roc_auc, 6),
            "val_brier_score": round(self.val_brier_score, 6),
            "val_log_loss": round(self.val_log_loss, 6),
            "training_duration_sec": round(self.training_duration_sec, 4),
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "fraud_rate_train": round(self.fraud_rate_train, 6),
            "fraud_rate_val": round(self.fraud_rate_val, 6),
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
    ) -> TrainingResult:
        """
        Train the cost-sensitive LightGBM model and calibrate posterior probabilities.

        Args:
            X: Feature matrix of shape (N, 16).
            y: Binary target labels (0=legitimate, 1=fraud).
            X_val: Optional validation feature matrix.
            y_val: Optional validation target labels.

        Returns:
            TrainingResult dataclass containing the fitted FraudClassifier and metrics.
        """
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm library is not installed in the environment")

        t0 = time.perf_counter()
        X_clean, y_clean = self._sanitize_inputs(X, y)

        # Train / validation split
        if X_val is None or y_val is None:
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

        # Enforce contiguous array layout on split datasets for Windows LightGBM C API safety
        X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
        y_tr = np.ascontiguousarray(y_tr, dtype=np.int32)
        X_va = np.ascontiguousarray(X_va, dtype=np.float32)
        y_va = np.ascontiguousarray(y_va, dtype=np.int32)

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
            raw_tr = clf.predict(X_tr)
            raw_va = clf.predict(X_va)

        raw_tr = np.asarray(raw_tr, dtype=np.float64).ravel()
        raw_va = np.asarray(raw_va, dtype=np.float64).ravel()

        # Fit probability calibrator on validation split
        calibrator = ProbabilityCalibrator(
            method=self.config.calibration_method,
            w_pos=self.config.w_pos,
        )
        calibrator.fit(raw_va, y_va)

        # Compute calibrated probabilities
        val_probs = np.asarray(calibrator.calibrate(raw_va), dtype=np.float64)

        # Calculate evaluation metrics
        # Train PR-AUC
        prec_tr, rec_tr, _ = precision_recall_curve(y_tr, raw_tr)
        train_pr_auc = float(auc(rec_tr, prec_tr))

        # Val PR-AUC
        prec_va, rec_va, _ = precision_recall_curve(y_va, raw_va)
        val_pr_auc = float(auc(rec_va, prec_va))

        # Val ROC-AUC
        if len(np.unique(y_va)) > 1:
            val_roc_auc = float(roc_auc_score(y_va, raw_va))
        else:
            val_roc_auc = 0.5

        # Brier Score & Log Loss on calibrated probabilities
        val_brier = float(brier_score_loss(y_va, val_probs))
        val_probs_clipped = np.clip(val_probs, 1e-12, 1.0 - 1e-12)
        val_ll = float(log_loss(y_va, val_probs_clipped))

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
            n_val_samples=len(X_va),
            fraud_rate_train=float(np.mean(y_tr)),
            fraud_rate_val=float(np.mean(y_va)),
        )

    def train_from_stream(
        self,
        events: Sequence[Any],
        feature_store: Optional[Any] = None,
    ) -> TrainingResult:
        """
        Train directly from a chronological list of stream transaction events.

        Extracts 16D feature vectors chronologically without temporal leakage.
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
        return self.train(X, y)
