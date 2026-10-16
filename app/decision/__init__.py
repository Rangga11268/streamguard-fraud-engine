"""
StreamGuard Decision Routing Package.

Provides cost-sensitive expected loss optimization, adaptive 3-tier routing
(Approve, Challenge, Block), and dynamic boundary recalibration under distribution drift.
"""

from app.decision.cost_matrix import (
    ACTION_TO_RISK_TIER,
    CostMatrixConfig,
    DecisionAction,
    RiskTier,
    compute_amount_thresholds,
    compute_expected_losses,
    compute_expected_losses_batch,
    evaluate_policy_expected_loss,
    evaluate_policy_realized_loss,
    expected_loss_approve,
    expected_loss_block,
    expected_loss_challenge,
    optimal_action,
    optimal_action_batch,
    optimize_thresholds,
    ThresholdOptimizer,
)
from app.decision.router import (
    AdaptiveDecisionRouter,
    RecalibrationEvent,
    ScoringResult,
)

__all__ = [
    "ACTION_TO_RISK_TIER",
    "AdaptiveDecisionRouter",
    "CostMatrixConfig",
    "DecisionAction",
    "RecalibrationEvent",
    "RiskTier",
    "ScoringResult",
    "ThresholdOptimizer",
    "compute_amount_thresholds",
    "compute_expected_losses",
    "compute_expected_losses_batch",
    "evaluate_policy_expected_loss",
    "evaluate_policy_realized_loss",
    "expected_loss_approve",
    "expected_loss_block",
    "expected_loss_challenge",
    "optimal_action",
    "optimal_action_batch",
    "optimize_thresholds",
]
