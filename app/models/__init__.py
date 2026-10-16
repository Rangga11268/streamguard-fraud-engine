"""
StreamGuard Models Package.

Provides cost-sensitive LightGBM models, asymmetric focal loss, probability calibration,
real-time sub-15ms inference wrappers, and Windows-safe training pipelines.
"""

from app.models.loss import (
    CostSensitiveAsymmetricLoss,
    CostSensitiveFocalLoss,
    make_cost_sensitive_objective,
    pr_auc_eval_sklearn,
    pr_auc_eval_native,
)
from app.models.calibration import (
    ProbabilityCalibrator,
    PlattCalibrator,
    BayesOddsCalibrator,
)
from app.models.classifier import (
    FraudClassifier,
    ModelPrediction,
)
from app.models.trainer import (
    FraudModelTrainer,
    ModelTrainingConfig,
    TrainingResult,
)

__all__ = [
    "CostSensitiveAsymmetricLoss",
    "CostSensitiveFocalLoss",
    "make_cost_sensitive_objective",
    "pr_auc_eval_sklearn",
    "pr_auc_eval_native",
    "ProbabilityCalibrator",
    "PlattCalibrator",
    "BayesOddsCalibrator",
    "FraudClassifier",
    "ModelPrediction",
    "FraudModelTrainer",
    "ModelTrainingConfig",
    "TrainingResult",
]
