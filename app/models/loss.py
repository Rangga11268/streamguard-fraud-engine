"""
Custom Cost-Sensitive and Asymmetric Focal Loss Objectives for LightGBM.

Formulated for extreme class imbalance (<0.1% fraud rate) where false negatives (FN)
carry financial chargeback losses orders of magnitude larger than false positive (FP) friction:
    w_pos = C_FN / C_FP ≈ 200.0 - 500.0 (default: 300.0)

Derives exact first (gradient) and second (hessian) derivatives with respect to
raw tree margins (logits), with numeric stabilization against overflow/underflow
and positive semi-definite regularization to ensure robust Newton split finding.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np


class CostSensitiveAsymmetricLoss:
    """
    Cost-sensitive weighted asymmetric loss objective for LightGBM.

    Loss Formulation:
        L(y, ŷ) = - [ w_pos * y * ln(σ(ŷ)) + (1 - y) * ln(1 - σ(ŷ)) ]
    where:
        ŷ is the raw tree margin (logit)
        p = σ(ŷ) = 1 / (1 + exp(-ŷ))
        w_i = w_pos * y_i + (1 - y_i)

    Derivatives:
        First derivative (gradient):
            g_i = ∂L / ∂ŷ_i = w_i * (p_i - y_i)
        Second derivative (hessian):
            h_i = ∂²L / ∂ŷ_i² = w_i * p_i * (1 - p_i)

    Attributes:
        w_pos: Positive class weight penalty (C_FN / C_FP). Default: 300.0.
        gamma: Focusing parameter for focal loss. When gamma=0.0, reduces to weighted cross-entropy.
        eps_clip: Margin clipping threshold to prevent exp overflow.
        min_hess: Minimum hessian threshold for numeric stability in Newton boosting steps.
    """

    def __init__(
        self,
        w_pos: float = 300.0,
        gamma: float = 0.0,
        eps_clip: float = 30.0,
        min_hess: float = 1e-6,
    ) -> None:
        if w_pos <= 0.0:
            raise ValueError(f"w_pos must be positive, got {w_pos}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.w_pos = float(w_pos)
        self.gamma = float(gamma)
        self.eps_clip = float(eps_clip)
        self.min_hess = float(min_hess)

    def compute_loss(
        self,
        y_true: np.ndarray,
        raw_margins: np.ndarray,
    ) -> float:
        """Compute the scalar mean loss for given true labels and raw margins."""
        y = np.asarray(y_true, dtype=np.float64)
        m = np.clip(np.asarray(raw_margins, dtype=np.float64), -self.eps_clip, self.eps_clip)
        p = 1.0 / (1.0 + np.exp(-m))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)

        if self.gamma == 0.0:
            # Weighted cross-entropy
            loss_i = - (self.w_pos * y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        else:
            # Asymmetric focal loss
            p_t = np.where(y == 1, p, 1.0 - p)
            alpha_t = np.where(y == 1, self.w_pos, 1.0)
            loss_i = - alpha_t * ((1.0 - p_t) ** self.gamma) * np.log(p_t)

        return float(np.mean(loss_i))

    def compute_derivatives(
        self,
        y_true: np.ndarray,
        raw_margins: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate analytical first derivative g (gradient) and second derivative h (hessian).

        Returns:
            grad: 1D np.ndarray of shape (N,) with float32 or float64 gradients.
            hess: 1D np.ndarray of shape (N,) with strictly positive hessians.
        """
        y = np.asarray(y_true, dtype=np.float64)
        m = np.clip(np.asarray(raw_margins, dtype=np.float64), -self.eps_clip, self.eps_clip)
        p = 1.0 / (1.0 + np.exp(-m))
        w = np.where(y == 1.0, self.w_pos, 1.0)

        if self.gamma == 0.0:
            # Standard cost-sensitive asymmetric weighted cross entropy
            grad = w * (p - y)
            hess = np.maximum(w * p * (1.0 - p), self.min_hess)
        else:
            # Asymmetric focal loss derivatives
            p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
            log_p = np.log(p_safe)
            log_1mp = np.log(1.0 - p_safe)

            # y = 1 term: w_pos * (1 - p)^gamma * [gamma * p * ln(p) + p - 1]
            grad_pos = self.w_pos * ((1.0 - p) ** self.gamma) * (
                self.gamma * p * log_p + p - 1.0
            )
            # y = 0 term: p^gamma * [p - gamma * (1 - p) * ln(1 - p)]
            grad_neg = (p ** self.gamma) * (
                p - self.gamma * (1.0 - p) * log_1mp
            )

            grad = np.where(y == 1.0, grad_pos, grad_neg)
            # Fisher information / diagonal positive semi-definite approximation for stable Newton steps
            hess = np.maximum(w * p * (1.0 - p), self.min_hess)

        return grad.astype(np.float32), hess.astype(np.float32)

    def sklearn_objective(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Callable conforming to scikit-learn LightGBM objective signature (y_true, y_pred)."""
        return self.compute_derivatives(y_true, y_pred)

    def native_objective(
        self,
        preds: np.ndarray,
        train_data: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Callable conforming to native LightGBM train objective signature (preds, train_data)."""
        y_true = train_data.get_label()
        return self.compute_derivatives(y_true, preds)

    def __call__(
        self,
        arg1: Any,
        arg2: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auto-dispatching callable supporting both scikit-learn and native LightGBM formats.

        If arg2 has `.get_label()`, treats (arg1=preds, arg2=train_data).
        Otherwise treats (arg1=y_true, arg2=y_pred).
        """
        if hasattr(arg2, "get_label"):
            return self.native_objective(arg1, arg2)
        return self.sklearn_objective(arg1, arg2)


class CostSensitiveFocalLoss(CostSensitiveAsymmetricLoss):
    """Convenience subclass for cost-sensitive focal loss with default gamma=2.0."""

    def __init__(
        self,
        w_pos: float = 300.0,
        gamma: float = 2.0,
        eps_clip: float = 30.0,
        min_hess: float = 1e-6,
    ) -> None:
        super().__init__(w_pos=w_pos, gamma=gamma, eps_clip=eps_clip, min_hess=min_hess)


def make_cost_sensitive_objective(
    w_pos: float = 300.0,
    gamma: float = 0.0,
    api: str = "sklearn",
) -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """
    Factory function returning a configured LightGBM custom objective callable.

    Args:
        w_pos: Penalty weight for false negatives (C_FN / C_FP, typically 200 - 500).
        gamma: Focal loss focusing parameter (0.0 for weighted CE, >0.0 for focal).
        api: "sklearn" (y_true, y_pred), "native" (preds, train_data), or "auto".

    Returns:
        Callable returning (grad, hess).
    """
    loss_obj = CostSensitiveAsymmetricLoss(w_pos=w_pos, gamma=gamma)
    if api == "sklearn":
        return loss_obj.sklearn_objective
    elif api == "native":
        return loss_obj.native_objective
    return loss_obj


def pr_auc_eval_sklearn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[str, float, bool]:
    """LightGBM custom evaluation metric computing Precision-Recall AUC for sklearn API."""
    from sklearn.metrics import precision_recall_curve, auc
    y = np.asarray(y_true, dtype=np.int32)
    # y_pred in custom objective is raw margin, convert to monotonic score
    scores = np.asarray(y_pred, dtype=np.float32)
    prec, rec, _ = precision_recall_curve(y, scores)
    pr_auc_val = float(auc(rec, prec))
    return "pr_auc", pr_auc_val, True


def pr_auc_eval_native(
    preds: np.ndarray,
    train_data: Any,
) -> Tuple[str, float, bool]:
    """LightGBM custom evaluation metric computing Precision-Recall AUC for native API."""
    y_true = train_data.get_label()
    return pr_auc_eval_sklearn(y_true, preds)
