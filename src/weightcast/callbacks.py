"""PyTorch Lightning callbacks for Weightcast early stopping.

During training, periodically extract WeightWatcher features, score them with
a pre-trained meta-model, and stop if P(underperform baseline) is high.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import weightwatcher as ww
from pytorch_lightning.callbacks import Callback

from src.weightcast.learner_classifier import CatBoostAUCClassifier
from src.weightcast.learner_regressor import CatBoostRegressionModel
from src.config import CB_N_STEPS
from src.weightcast.watcher_summarizer import NeuralWeightsFeatureEng


class WeightcastCallback(Callback, ABC):
    """Base class for Weightcast early stopping callbacks.

    Uses a pre-trained meta-model to predict P(MASE > MASE_baseline) from
    WeightWatcher features during training. Stops if probability exceeds threshold.

    Subclasses must implement `_predict_exceedance` to define how the
    meta-model produces exceedance probabilities.

    Attributes:
        predictions: Log of (step, prob_exceed, threshold) at each check.
        stopped_early: Whether training was stopped by this callback.
        stop_step: Global step at which stopping occurred (if any).
    """

    MIN_STEPS_BEFORE_STOPPING = 50
    _callback_name: str = 'WeightcastCallback'

    def __init__(
            self,
            feature_columns: list[str],
            config_data: Dict[str, Any],
            category_mappings: Optional[Dict[str, Dict[str, int]]] = None,
            stopping_threshold: float = 0.5,
            every_n_steps: int = CB_N_STEPS,
            min_steps: int = MIN_STEPS_BEFORE_STOPPING,
            verbose: bool = True,
    ):
        """
        Args:
            feature_columns: Feature names expected by the meta-model.
            config_data: Hyperparameters of the trial being trained.
            category_mappings: Encodings for categorical config fields.
            stopping_threshold: Stop when P(exceed) > this value.
            every_n_steps: Evaluate the meta-model every N optimizer steps.
            min_steps: Burn-in before early stopping is allowed.
            verbose: Print P(exceed) at each check.
        """
        super().__init__()
        self.name = self._callback_name
        self.feature_columns = feature_columns
        self.stopping_threshold = stopping_threshold
        self.every_n_steps = every_n_steps
        self.min_steps = min_steps
        self.verbose = verbose
        self.config_data = config_data
        self.category_mappings = category_mappings or {}

        self.predictions: list[dict] = []
        self.stopped_early: bool = False
        self.stop_step: Optional[int] = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step

        if step < self.min_steps:
            return

        if step % self.every_n_steps != 0:
            return

        features = self._extract_features(pl_module, step)
        if features is None:
            return

        prob_exceed = self._predict_exceedance(features)

        self.predictions.append({
            'step': step,
            'prob_exceed': prob_exceed,
            'threshold': self.stopping_threshold,
        })

        if self.verbose:
            print(f"  [Step {step}] P(exceed) = {prob_exceed:.3f}", end="")

        if prob_exceed > self.stopping_threshold:
            if self.verbose:
                print(f" > {self.stopping_threshold} -> STOPPING EARLY")
            trainer.should_stop = True
            self.stopped_early = True
            self.stop_step = step
        elif self.verbose:
            print()

    def _extract_features(self, pl_module, step: int) -> Optional[pd.DataFrame]:
        """Extract WeightWatcher features from the model."""
        try:
            watcher = ww.WeightWatcher(model=pl_module)
            details = watcher.analyze(plot=False)

            smr_stats = NeuralWeightsFeatureEng.snapshop_detail_stats(details, add_performance=False)
            smr_stats['step'] = step

            for k, v in self.config_data.items():
                if k == 'scaler_type' and v is None:
                    smr_stats[k] = 'none'
                smr_stats[k] = v

            smr_stats['learning_rate'] = NeuralWeightsFeatureEng.bin_learning_rate(smr_stats['learning_rate'])
            smr_stats['start_padding_enabled'] = int(smr_stats['start_padding_enabled'])

            features_dict = {col: smr_stats.get(col, np.nan) for col in self.feature_columns}

            for col, mapping in self.category_mappings.items():
                col_val = features_dict[col]
                is_str = isinstance(features_dict[col], str)
                if isinstance(col_val, list):
                    col_val = str(col_val)

                if col in features_dict and col_val in mapping:
                    features_dict[col] = mapping[col_val]
                elif col in features_dict and is_str:
                    features_dict[col] = -1

            return pd.DataFrame([features_dict])
        except Exception as e:
            if self.verbose:
                print(f"  [Step {step}] Feature extraction failed: {e}")
            return None

    @abstractmethod
    def _predict_exceedance(self, features: pd.DataFrame) -> float:
        """Predict probability of exceeding baseline performance.
        
        Subclasses must implement this to define how the meta-model
        produces exceedance probabilities.
        """
        pass

    @classmethod
    def get_cb(cls, nf) -> "WeightcastCallback":
        """Retrieve the actual callback instance from a fitted NeuralForecast model.

        NeuralForecast deep-copies callbacks, so the original instance won't have
        the updated state. Use this method to get the actual callback after fitting.

        Searches across all models in the NeuralForecast instance.
        """
        for model in nf.models:
            all_cbs = model.trainer_kwargs.get('callbacks', [])
            for cb in all_cbs:
                if getattr(cb, 'name', None) == cls._callback_name:
                    return cb
        raise ValueError(f"{cls._callback_name} not found in any model callbacks")


class WeightcastRegressor(WeightcastCallback):
    """Early stopping callback using regression meta-model with conformal prediction.

    Uses a pre-trained regression model to predict P(MASE_diff > threshold) from
    WeightWatcher features during training. Stops if probability exceeds threshold.
    """

    _callback_name = 'WeightcastRegressor'

    def __init__(
            self,
            meta_model: CatBoostRegressionModel,
            feature_columns: list[str],
            config_data: Dict[str, Any],
            category_mappings: Optional[Dict[str, Dict[str, int]]] = None,
            stopping_threshold: float = 0.5,
            exceedance_threshold: float = 0.0,
            every_n_steps: int = CB_N_STEPS,
            min_steps: int = WeightcastCallback.MIN_STEPS_BEFORE_STOPPING,
            verbose: bool = True,
    ):
        super().__init__(
            feature_columns=feature_columns,
            config_data=config_data,
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            every_n_steps=every_n_steps,
            min_steps=min_steps,
            verbose=verbose,
        )
        self.meta_model = meta_model
        self.exceedance_threshold = exceedance_threshold

    def _predict_exceedance(self, features: pd.DataFrame) -> float:
        """Predict probability of exceeding baseline using conformal regression.

        Uses raw conformal probabilities (no calibration) for speed.
        For early stopping, ranking accuracy matters more than calibration.
        """
        prob = self.meta_model.prob_exceeds(
            features[self.feature_columns],
            self.exceedance_threshold,
            calibration_method="none",
        )
        return float(prob[0])


class WeightcastClassifier(WeightcastCallback):
    """Early stopping callback using binary classifier.

    Uses a pre-trained binary classifier to predict P(MASE > MASE_baseline) from
    WeightWatcher features during training. Stops if probability exceeds threshold.

    This is simpler than the regression + conformal approach since the classifier
    directly outputs P(exceeds baseline).
    """

    _callback_name = 'WeightcastClassifier'

    def __init__(
            self,
            meta_classifier: CatBoostAUCClassifier,
            feature_columns: list[str],
            config_data: Dict[str, Any],
            category_mappings: Optional[Dict[str, Dict[str, int]]] = None,
            stopping_threshold: float = 0.5,
            every_n_steps: int = CB_N_STEPS,
            min_steps: int = WeightcastCallback.MIN_STEPS_BEFORE_STOPPING,
            verbose: bool = True,
    ):
        super().__init__(
            feature_columns=feature_columns,
            config_data=config_data,
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            every_n_steps=every_n_steps,
            min_steps=min_steps,
            verbose=verbose,
        )
        self.meta_classifier = meta_classifier

    def _predict_exceedance(self, features: pd.DataFrame) -> float:
        """Predict probability of exceeding baseline using binary classifier."""
        prob = self.meta_classifier.predict_proba_positive(
            features[self.feature_columns],
            calibrated=True,
        )
        return float(prob[0])
