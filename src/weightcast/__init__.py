"""Weightcast: meta-model early stopping from WeightWatcher features.

Public API:
    WeightcastClassifier / WeightcastRegressor — Lightning callbacks.
    WeightcastAutoConfig — inject Weightcast into NeuralForecast AutoModels.
    CatBoostAUCClassifier / CatBoostRegressionModel — meta-learners.
    WeightWatcherCallback — offline snapshot collection during training.
    NeuralWeightsFeatureEng — feature engineering over WeightWatcher details.
"""

from src.weightcast.auto import WeightcastAutoConfig, StepAccumulator
from src.weightcast.callbacks import (
    WeightcastCallback,
    WeightcastClassifier,
    WeightcastRegressor,
)
from src.weightcast.learner_classifier import CatBoostAUCClassifier
from src.weightcast.learner_regressor import CatBoostRegressionModel, ConformalPredictiveDistribution
from src.weightcast.watcher_callback import WeightWatcherCallback
from src.weightcast.watcher_summarizer import NeuralWeightsFeatureEng

__all__ = [
    'WeightcastAutoConfig',
    'StepAccumulator',
    'WeightcastCallback',
    'WeightcastClassifier',
    'WeightcastRegressor',
    'CatBoostAUCClassifier',
    'CatBoostRegressionModel',
    'ConformalPredictiveDistribution',
    'WeightWatcherCallback',
    'NeuralWeightsFeatureEng',
]
