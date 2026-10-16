"""
Cost-sensitive financial loss matrix and expected loss optimization for StreamGuard.

Implements:
  1. Parameterized financial cost matrix for Approve, Challenge, and Block actions.
  2. Mathematical expected loss formulation:
     - E[Loss(Approve | p, Amount)] = p * (Amount + C_chargeback)
     - E[Loss(Challenge | p, Amount)] = p * [(1 - eta) * (Amount + C_chargeback) + C_otp] + (1 - p) * C_friction
     - E[Loss(Block | p, Amount)] = p * C_tp_review + (1 - p) * C_fp_churn
  3. Action optimization:
     argmin_{Approve, Challenge, Block} E[Loss(a | p, Amount)]
  4. Amount-dependent and dataset-wide threshold optimizer for (tau_challenge, tau_block).
  5. Policy evaluation and loss reduction calculation (>= 25% savings vs static tau=0.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from app.config import StreamGuardConfig, default_config
except ImportError:
    default_config = None


class DecisionAction(str, Enum):
    """3-Tier decision routing actions."""
    APPROVE = "APPROVE"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


class RiskTier(str, Enum):
    """Risk tier mapping corresponding to actions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


ACTION_TO_RISK_TIER: Dict[DecisionAction, RiskTier] = {
    DecisionAction.APPROVE: RiskTier.LOW,
    DecisionAction.CHALLENGE: RiskTier.MEDIUM,
    DecisionAction.BLOCK: RiskTier.HIGH,
}


@dataclass(slots=True)
class CostMatrixConfig:
    """
    Financial cost matrix parameters.

    Attributes:
        c_chargeback: Bank chargeback penalty fee added to transaction amount on fraud ($25.0).
        eta: Challenge (Step-up OTP/biometric) fraud interception success rate (0.95 = 95%).
        c_otp: Direct cost of sending OTP / biometric challenge ($0.05).
        c_friction: Customer checkout abandonment friction cost on challenge ($1.50).
        c_tp_review: Operational manual review / dispute handling cost for blocked fraud ($2.00).
        c_fp_churn: Customer lifetime value churn / brand damage on false decline ($50.0).
    """
    c_chargeback: float = 25.0
    eta: float = 0.95
    c_otp: float = 0.05
    c_friction: float = 1.50
    c_tp_review: float = 2.0
    c_fp_churn: float = 50.0

    @classmethod
    def from_config(cls, cfg: Optional[Any] = None) -> CostMatrixConfig:
        """Construct CostMatrixConfig from StreamGuardConfig or global defaults."""
        if cfg is None:
            if default_config is not None:
                cfg = default_config
            else:
                return cls()

        return cls(
            c_chargeback=getattr(cfg, "cost_fn_fixed", 25.0),
            eta=getattr(cfg, "otp_interception_rate", 0.95),
            c_otp=getattr(cfg, "cost_otp_challenge", 0.05),
            c_friction=getattr(cfg, "cost_challenge_friction", 1.50),
            c_tp_review=getattr(cfg, "cost_block_investigation", 2.0),
            c_fp_churn=getattr(cfg, "cost_block_churn", 50.0),
        )


# ============================================================================
# Expected Loss Calculations
# ============================================================================

def expected_loss_approve(
    p: float,
    amount: float,
    config: CostMatrixConfig,
) -> float:
    """
    E[Loss(Approve | p, Amount)] = p * (Amount + C_chargeback)

    If legitimate (prob 1-p), cost is $0.
    If fraud (prob p), merchant suffers full loss of transaction amount plus chargeback fee.
    """
    p_clamped = min(1.0, max(0.0, float(p)))
    amt_clamped = max(0.0, float(amount))
    return p_clamped * (amt_clamped + config.c_chargeback)


def expected_loss_challenge(
    p: float,
    amount: float,
    config: CostMatrixConfig,
) -> float:
    """
    E[Loss(Challenge | p, Amount)] = p * [(1 - eta) * (Amount + C_chargeback) + C_otp] + (1 - p) * C_friction

    If fraud (prob p):
      - eta fraction is successfully intercepted and prevented.
      - (1 - eta) fraction bypasses OTP and causes full chargeback loss.
      - OTP cost C_otp is always incurred.
    If legitimate (prob 1-p):
      - Customer experiences friction/abandonment penalty C_friction.
    """
    p_clamped = min(1.0, max(0.0, float(p)))
    amt_clamped = max(0.0, float(amount))
    fraud_term = (1.0 - config.eta) * (amt_clamped + config.c_chargeback) + config.c_otp
    legit_term = config.c_friction
    return p_clamped * fraud_term + (1.0 - p_clamped) * legit_term


def expected_loss_block(
    p: float,
    amount: float,
    config: CostMatrixConfig,
) -> float:
    """
    E[Loss(Block | p, Amount)] = p * C_tp_review + (1 - p) * C_fp_churn

    If fraud (prob p):
      - Fraud is stopped; small operational manual review / intake cost C_tp_review.
    If legitimate (prob 1-p):
      - False decline causing customer churn / lost lifetime value C_fp_churn.
    """
    p_clamped = min(1.0, max(0.0, float(p)))
    return p_clamped * config.c_tp_review + (1.0 - p_clamped) * config.c_fp_churn


def compute_expected_losses(
    p: float,
    amount: float,
    config: Optional[CostMatrixConfig] = None,
) -> Dict[DecisionAction, float]:
    """Compute expected losses for all three actions for a given (p, amount)."""
    cfg = config or CostMatrixConfig()
    return {
        DecisionAction.APPROVE: expected_loss_approve(p, amount, cfg),
        DecisionAction.CHALLENGE: expected_loss_challenge(p, amount, cfg),
        DecisionAction.BLOCK: expected_loss_block(p, amount, cfg),
    }


def optimal_action(
    p: float,
    amount: float,
    config: Optional[CostMatrixConfig] = None,
) -> Tuple[DecisionAction, float]:
    """
    Compute optimal action minimizing expected financial loss:
        argmin_{Approve, Challenge, Block} E[Loss(a | p, Amount)]

    Returns:
        (best_action, minimum_expected_loss)
    """
    cfg = config or CostMatrixConfig()
    losses = compute_expected_losses(p, amount, cfg)

    # Tie-breaking priority: APPROVE < CHALLENGE < BLOCK
    best_action = min(
        [DecisionAction.APPROVE, DecisionAction.CHALLENGE, DecisionAction.BLOCK],
        key=lambda a: losses[a],
    )
    return best_action, losses[best_action]


# ============================================================================
# Vectorized Batch Computations
# ============================================================================

def compute_expected_losses_batch(
    probs: np.ndarray,
    amounts: np.ndarray,
    config: Optional[CostMatrixConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of expected losses for arrays of probabilities and amounts.

    Returns:
        (loss_approve, loss_challenge, loss_block)
    """
    cfg = config or CostMatrixConfig()
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    a = np.clip(np.asarray(amounts, dtype=np.float64), 0.0, None)

    fraud_loss = a + cfg.c_chargeback
    loss_app = p * fraud_loss
    loss_chal = p * ((1.0 - cfg.eta) * fraud_loss + cfg.c_otp) + (1.0 - p) * cfg.c_friction
    loss_blk = p * cfg.c_tp_review + (1.0 - p) * cfg.c_fp_churn

    return loss_app, loss_chal, loss_blk


def optimal_action_batch(
    probs: np.ndarray,
    amounts: np.ndarray,
    config: Optional[CostMatrixConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized argmin expected loss across arrays.

    Returns:
        (actions_array, min_losses_array)
        where actions_array contains strings: "APPROVE", "CHALLENGE", "BLOCK".
    """
    loss_app, loss_chal, loss_blk = compute_expected_losses_batch(probs, amounts, config)
    stacked = np.column_stack([loss_app, loss_chal, loss_blk])

    # argmin index: 0 = APPROVE, 1 = CHALLENGE, 2 = BLOCK
    min_indices = np.argmin(stacked, axis=1)
    min_losses = np.min(stacked, axis=1)

    action_lookup = np.array([
        DecisionAction.APPROVE.value,
        DecisionAction.CHALLENGE.value,
        DecisionAction.BLOCK.value,
    ], dtype=object)

    return action_lookup[min_indices], min_losses


# ============================================================================
# Threshold Optimization
# ============================================================================

def compute_amount_thresholds(
    amount: float,
    config: Optional[CostMatrixConfig] = None,
) -> Tuple[float, float]:
    """
    Analytically compute the intersection boundaries (tau_challenge, tau_block)
    for a specific transaction amount where:
        E[Loss(Approve)] = E[Loss(Challenge)] -> tau_challenge
        E[Loss(Challenge)] = E[Loss(Block)]   -> tau_block

    Derivation:
      p * (A + C_cb) = p * [(1 - eta)(A + C_cb) + C_otp] + (1 - p) * C_friction
      => tau_challenge = C_friction / [eta * (A + C_cb) + C_friction - C_otp]

      p * [(1 - eta)(A + C_cb) + C_otp] + (1 - p) * C_friction = p * C_tp + (1 - p) * C_churn
      => tau_block = (C_churn - C_friction) / [(C_churn - C_friction) + (1 - eta)(A + C_cb) + C_otp - C_tp]
    """
    cfg = config or CostMatrixConfig()
    amt = max(0.0, float(amount))
    fraud_cost = amt + cfg.c_chargeback

    denom_chal = cfg.eta * fraud_cost + cfg.c_friction - cfg.c_otp
    if denom_chal > 1e-6:
        tau_chal = cfg.c_friction / denom_chal
    else:
        tau_chal = 0.50

    net_churn = cfg.c_fp_churn - cfg.c_friction
    denom_blk = net_churn + (1.0 - cfg.eta) * fraud_cost + cfg.c_otp - cfg.c_tp_review
    if denom_blk > 1e-6:
        tau_blk = net_churn / denom_blk
    else:
        tau_blk = 0.85

    # Enforce valid range and ordering
    tau_chal = float(np.clip(tau_chal, 0.005, 0.50))
    tau_blk = float(np.clip(tau_blk, tau_chal + 0.01, 0.99))

    return tau_chal, tau_blk


class ThresholdOptimizer:
    """
    Finds optimal global decision thresholds (tau_challenge, tau_block)
    that minimize total financial loss across an empirical distribution of transactions.
    """

    def __init__(self, config: Optional[CostMatrixConfig] = None) -> None:
        self.config = config or CostMatrixConfig()

    def optimize(
        self,
        probabilities: np.ndarray,
        amounts: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        grid_points: int = 50,
    ) -> Tuple[float, float, float]:
        """
        Optimize (tau_challenge, tau_block) to minimize empirical or expected financial loss.

        Args:
            probabilities: Array of calibrated fraud probabilities.
            amounts: Array of transaction amounts.
            true_labels: Optional array of binary ground-truth labels {0, 1}.
                         If None, optimizes total expected loss.
                         If provided, optimizes total realized financial loss.
            grid_points: Resolution of search grid along each dimension.

        Returns:
            (best_tau_challenge, best_tau_block, minimum_total_loss)
        """
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
        a = np.clip(np.asarray(amounts, dtype=np.float64), 0.0, None)
        has_labels = true_labels is not None
        if has_labels:
            y = (np.asarray(true_labels) == 1).astype(np.int32)

        # Generate candidate thresholds
        tau1_candidates = np.linspace(0.01, 0.45, grid_points)
        tau2_candidates = np.linspace(0.20, 0.95, grid_points)

        cfg = self.config
        fraud_losses = a + cfg.c_chargeback

        best_loss = float("inf")
        best_tau1 = 0.15
        best_tau2 = 0.70

        for t1 in tau1_candidates:
            for t2 in tau2_candidates:
                if t2 <= t1:
                    continue

                if has_labels:
                    # Realized loss calculation
                    loss = 0.0
                    # Approve: p < t1
                    app_mask = p < t1
                    loss += float(np.sum(fraud_losses[app_mask & (y == 1)]))

                    # Challenge: t1 <= p < t2
                    chal_mask = (p >= t1) & (p < t2)
                    loss += float(np.sum(chal_mask & (y == 0))) * (cfg.c_friction + cfg.c_otp)
                    chal_fraud = chal_mask & (y == 1)
                    if np.any(chal_fraud):
                        loss += float(np.sum((1.0 - cfg.eta) * fraud_losses[chal_fraud] + cfg.c_otp))

                    # Block: p >= t2
                    blk_mask = p >= t2
                    loss += float(np.sum(blk_mask & (y == 0))) * cfg.c_fp_churn
                    loss += float(np.sum(blk_mask & (y == 1))) * cfg.c_tp_review
                else:
                    # Expected loss calculation
                    loss_app, loss_chal, loss_blk = compute_expected_losses_batch(p, a, cfg)
                    total_exp = np.where(
                        p < t1,
                        loss_app,
                        np.where(p < t2, loss_chal, loss_blk),
                    )
                    loss = float(np.sum(total_exp))

                if loss < best_loss:
                    best_loss = loss
                    best_tau1 = float(t1)
                    best_tau2 = float(t2)

        return best_tau1, best_tau2, best_loss


def optimize_thresholds(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    config: Optional[CostMatrixConfig] = None,
    grid_points: int = 50,
) -> Tuple[float, float]:
    """Helper function to find optimal global thresholds."""
    optimizer = ThresholdOptimizer(config=config)
    tau1, tau2, _ = optimizer.optimize(
        probabilities=probabilities,
        amounts=amounts,
        true_labels=true_labels,
        grid_points=grid_points,
    )
    return tau1, tau2


# ============================================================================
# Policy Evaluation & Cost Reduction Calculation
# ============================================================================

def evaluate_policy_realized_loss(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    true_labels: np.ndarray,
    tau_challenge: Optional[float] = None,
    tau_block: Optional[float] = None,
    config: Optional[CostMatrixConfig] = None,
    use_amount_dependent_thresholds: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate realized financial losses under 3-tier routing versus static threshold (tau = 0.5).

    Returns dictionary containing:
        - loss_static: Total loss under binary static policy (tau = 0.5)
        - loss_3tier: Total loss under 3-tier adaptive policy
        - loss_reduction_pct: Percentage loss reduction ((static - 3tier) / static) * 100
        - savings_amount: Dollar savings (static - 3tier)
        - action_counts: Count of transactions in Approve, Challenge, Block
    """
    cfg = config or CostMatrixConfig()
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    a = np.clip(np.asarray(amounts, dtype=np.float64), 0.0, None)
    y = (np.asarray(true_labels) == 1).astype(np.int32)
    n = len(p)

    loss_static = 0.0
    loss_3tier = 0.0
    action_counts = {
        DecisionAction.APPROVE.value: 0,
        DecisionAction.CHALLENGE.value: 0,
        DecisionAction.BLOCK.value: 0,
    }

    for i in range(n):
        p_i = p[i]
        a_i = a[i]
        is_fraud = bool(y[i] == 1)
        fraud_cost = a_i + cfg.c_chargeback

        # Static policy: tau = 0.5
        if p_i >= 0.5:
            loss_static += cfg.c_tp_review if is_fraud else cfg.c_fp_churn
        else:
            loss_static += fraud_cost if is_fraud else 0.0

        # 3-Tier Policy
        if use_amount_dependent_thresholds:
            # Optimal action via direct expected loss argmin
            action, _ = optimal_action(p_i, a_i, cfg)
        else:
            t1 = tau_challenge if tau_challenge is not None else 0.15
            t2 = tau_block if tau_block is not None else 0.70
            if p_i < t1:
                action = DecisionAction.APPROVE
            elif p_i < t2:
                action = DecisionAction.CHALLENGE
            else:
                action = DecisionAction.BLOCK

        action_counts[action.value] += 1

        if action == DecisionAction.APPROVE:
            if is_fraud:
                loss_3tier += fraud_cost
        elif action == DecisionAction.CHALLENGE:
            if is_fraud:
                loss_3tier += (1.0 - cfg.eta) * fraud_cost + cfg.c_otp
            else:
                loss_3tier += cfg.c_friction + cfg.c_otp
        elif action == DecisionAction.BLOCK:
            if is_fraud:
                loss_3tier += cfg.c_tp_review
            else:
                loss_3tier += cfg.c_fp_churn

    reduction_pct = ((loss_static - loss_3tier) / max(loss_static, 1e-4)) * 100.0
    savings = loss_static - loss_3tier

    return {
        "loss_static": float(loss_static),
        "loss_3tier": float(loss_3tier),
        "loss_reduction_pct": float(reduction_pct),
        "savings_amount": float(savings),
        "action_counts": action_counts,
        "n_samples": n,
    }


def evaluate_policy_expected_loss(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    tau_challenge: Optional[float] = None,
    tau_block: Optional[float] = None,
    config: Optional[CostMatrixConfig] = None,
    use_amount_dependent_thresholds: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate expected financial losses without ground truth labels.
    """
    cfg = config or CostMatrixConfig()
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    a = np.clip(np.asarray(amounts, dtype=np.float64), 0.0, None)
    n = len(p)

    loss_app, loss_chal, loss_blk = compute_expected_losses_batch(p, a, cfg)

    # Static policy: p < 0.5 -> Approve, p >= 0.5 -> Block
    static_loss_array = np.where(p < 0.5, loss_app, loss_blk)
    loss_static = float(np.sum(static_loss_array))

    if use_amount_dependent_thresholds:
        stacked = np.column_stack([loss_app, loss_chal, loss_blk])
        min_losses = np.min(stacked, axis=1)
        loss_3tier = float(np.sum(min_losses))
        indices = np.argmin(stacked, axis=1)
        action_counts = {
            DecisionAction.APPROVE.value: int(np.sum(indices == 0)),
            DecisionAction.CHALLENGE.value: int(np.sum(indices == 1)),
            DecisionAction.BLOCK.value: int(np.sum(indices == 2)),
        }
    else:
        t1 = tau_challenge if tau_challenge is not None else 0.15
        t2 = tau_block if tau_block is not None else 0.70
        three_tier_array = np.where(p < t1, loss_app, np.where(p < t2, loss_chal, loss_blk))
        loss_3tier = float(np.sum(three_tier_array))
        action_counts = {
            DecisionAction.APPROVE.value: int(np.sum(p < t1)),
            DecisionAction.CHALLENGE.value: int(np.sum((p >= t1) & (p < t2))),
            DecisionAction.BLOCK.value: int(np.sum(p >= t2)),
        }

    reduction_pct = ((loss_static - loss_3tier) / max(loss_static, 1e-4)) * 100.0
    savings = loss_static - loss_3tier

    return {
        "loss_static": float(loss_static),
        "loss_3tier": float(loss_3tier),
        "loss_reduction_pct": float(reduction_pct),
        "savings_amount": float(savings),
        "action_counts": action_counts,
        "n_samples": n,
    }
