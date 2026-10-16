"""
Probability Calibration Engine for Cost-Sensitive Fraud Classifiers.

Maps raw tree margins (logits) or cost-weighted risk scores into true calibrated
posterior probabilities P(y=1|x) in [0.0, 1.0].

Supports two complementary approaches:
1. Platt Scaling (Sigmoid Logistic Regression with Laplace regularization):
   Solves for parameters A, B:
       P(y=1 | s) = σ(A * s + B) = 1 / (1 + exp(-(A * s + B)))
   Using Platt's Laplace regularized target labels:
       t_+ = (N_+ + 1) / (N_+ + 2)
       t_- = 1 / (N_- + 2)
   to prevent overfitting under extreme class imbalance (<0.1% fraud).

2. Bayes Odds Adjustment:
   Inverts the known positive weight penalty w_pos = C_FN / C_FP:
       Odds_true = (1 / w_pos) * Odds_model
       P(y=1 | s) = σ(s - ln(w_pos))
   enabling instant, exact O(1) calibration without iterative optimization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize


class PlattCalibrator:
    """
    Sigmoid / Platt scaling calibrator with Laplace smoothed target labels.

    Learns parameters A (slope) and B (intercept) such that:
        P(y=1 | s) = 1 / (1 + exp(-(A * s + B)))
    """

    def __init__(self, l2_reg: float = 1e-4) -> None:
        self.l2_reg = float(l2_reg)
        self.A: float = 1.0
        self.B: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> "PlattCalibrator":
        """
        Fit Platt sigmoid parameters A and B using L-BFGS-B with Laplace smoothed targets.

        Args:
            raw_scores: 1D array of raw margins or logit predictions.
            y_true: 1D binary labels (0 or 1).
        """
        s = np.asarray(raw_scores, dtype=np.float64).ravel()
        y = np.asarray(y_true, dtype=np.int32).ravel()

        if len(s) != len(y):
            raise ValueError(f"Length mismatch: raw_scores ({len(s)}) != y_true ({len(y)})")

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        if n_pos == 0:
            # Fallback if no positive examples present in calibration split
            self.A = 1.0
            self.B = -10.0
            self.is_fitted = True
            return self

        # Platt regularized targets (Laplace smoothing)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(y == 1, t_pos, t_neg)

        # Objective function and exact gradient
        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            A, B = params[0], params[1]
            z = np.clip(A * s + B, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)

            # Negative log-likelihood with L2 regularization
            loss = - float(np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))
            loss += self.l2_reg * (A * A + B * B)

            diff = p - targets
            grad_A = float(np.sum(diff * s)) + 2.0 * self.l2_reg * A
            grad_B = float(np.sum(diff)) + 2.0 * self.l2_reg * B
            return loss, np.array([grad_A, grad_B], dtype=np.float64)

        # Initial guess: positive slope, intercept from prior log-odds
        prior_odds = (n_pos + 1.0) / (n_neg + 1.0)
        init_B = float(math.log(max(prior_odds, 1e-6)))
        init_A = 1.0

        res = minimize(
            loss_and_grad,
            x0=np.array([init_A, init_B], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.01, 20.0), (-30.0, 30.0)],  # Enforce positive slope for monotonicity
            options={"maxiter": 100, "ftol": 1e-7},
        )

        self.A = float(res.x[0])
        self.B = float(res.x[1])
        self.is_fitted = True
        return self

    def calibrate(self, raw_scores: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map raw margin(s) to calibrated probability in [0.0, 1.0]."""
        is_scalar = isinstance(raw_scores, (int, float, np.floating))
        s = np.asarray(raw_scores, dtype=np.float64)

        z = np.clip(self.A * s + self.B, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = np.clip(prob, 0.0, 1.0)

        # Handle NaNs or Infs if any slipped through
        prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

        if is_scalar:
            return float(prob.item())
        return prob.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "platt",
            "A": self.A,
            "B": self.B,
            "l2_reg": self.l2_reg,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlattCalibrator":
        cal = cls(l2_reg=data.get("l2_reg", 1e-4))
        cal.A = float(data.get("A", 1.0))
        cal.B = float(data.get("B", 0.0))
        cal.is_fitted = bool(data.get("is_fitted", True))
        return cal


class BayesOddsCalibrator:
    """
    Analytic probability calibrator via Bayes Odds Adjustment.

    When a model is trained with positive weight penalty w_pos = C_FN / C_FP,
    the raw tree margins are shifted by approximately ln(w_pos). This calibrator
    mathematically inverts that shift:
        P(y=1 | s) = σ(s - ln(w_pos) + offset)
    """

    def __init__(self, w_pos: float = 300.0, offset: float = 0.0) -> None:
        if w_pos <= 0.0:
            raise ValueError(f"w_pos must be positive, got {w_pos}")
        self.w_pos = float(w_pos)
        self.offset = float(offset)
        self.log_w_pos: float = math.log(self.w_pos)
        self.is_fitted: bool = True

    def fit(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> "BayesOddsCalibrator":
        """Optionally tune offset on calibration data to match empirical base rate."""
        s = np.asarray(raw_scores, dtype=np.float64).ravel()
        y = np.asarray(y_true, dtype=np.int32).ravel()
        if len(y) > 0 and np.sum(y == 1) > 0:
            target_rate = float(np.mean(y))
            # Binary search or 1D optimization for offset
            def rate_diff(off: float) -> float:
                z = np.clip(s - self.log_w_pos + off, -30.0, 30.0)
                p = 1.0 / (1.0 + np.exp(-z))
                return float((np.mean(p) - target_rate) ** 2)

            res = minimize(rate_diff, x0=[0.0], method="Nelder-Mead")
            self.offset = float(res.x[0])
        return self

    def calibrate(self, raw_scores: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map raw margin(s) to calibrated probability in [0.0, 1.0]."""
        is_scalar = isinstance(raw_scores, (int, float, np.floating))
        s = np.asarray(raw_scores, dtype=np.float64)

        z = np.clip(s - self.log_w_pos + self.offset, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = np.clip(prob, 0.0, 1.0)
        prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

        if is_scalar:
            return float(prob.item())
        return prob.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "bayes_odds",
            "w_pos": self.w_pos,
            "offset": self.offset,
            "log_w_pos": self.log_w_pos,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BayesOddsCalibrator":
        cal = cls(
            w_pos=data.get("w_pos", 300.0),
            offset=data.get("offset", 0.0),
        )
        return cal


class ProbabilityCalibrator:
    """
    Unified probability calibration interface for StreamGuard models.

    Supports methods:
        - "platt": Platt scaling with Laplace smoothed targets (recommended default)
        - "bayes_odds": Bayes odds cost-weight inversion
    """

    def __init__(
        self,
        method: str = "platt",
        w_pos: float = 300.0,
        l2_reg: float = 1e-4,
    ) -> None:
        self.method = method.lower()
        if self.method == "platt":
            self.underlying: Union[PlattCalibrator, BayesOddsCalibrator] = PlattCalibrator(l2_reg=l2_reg)
        elif self.method in ("bayes_odds", "bayes"):
            self.underlying = BayesOddsCalibrator(w_pos=w_pos)
        else:
            raise ValueError(f"Unknown calibration method: '{method}'. Choose 'platt' or 'bayes_odds'.")

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibrator on held-out scores and labels."""
        self.underlying.fit(raw_scores, y_true)
        return self

    def calibrate(self, raw_scores: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calibrate input margin(s) to strictly bounded posterior probability in [0.0, 1.0]."""
        return self.underlying.calibrate(raw_scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "calibrator_data": self.underlying.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilityCalibrator":
        method = data.get("method", "platt")
        cal = cls(method=method)
        cdata = data.get("calibrator_data", {})
        if method == "platt":
            cal.underlying = PlattCalibrator.from_dict(cdata)
        else:
            cal.underlying = BayesOddsCalibrator.from_dict(cdata)
        return cal
