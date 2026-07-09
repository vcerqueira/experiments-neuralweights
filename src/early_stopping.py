from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import weightwatcher as ww
from pytorch_lightning.callbacks import Callback

from src.algorithms import CatBoostRegressionModel, CatBoostAUCClassifier
from src.config import CB_N_STEPS
from src.weights.weight_summarizer import NeuralWeightsFeatureEng


class MetaModelEarlyStopCallback(Callback):
    """Early stopping callback based on meta-model exceedance probability predictions.

    Uses a pre-trained meta-model to predict P(MASE > MASE_baseline) from
    WeightWatcher features during training. Stops if probability exceeds threshold.
    """

    MIN_STEPS_BEFORE_STOPPING = 50

    def __init__(
            self,
            meta_model: CatBoostRegressionModel,
            feature_columns: list[str],
            config_data: Dict[str, Any],
            stopping_threshold: float = 0.5,
            exceedance_threshold: float = 0.0,
            every_n_steps: int = CB_N_STEPS,
            min_steps: int = MIN_STEPS_BEFORE_STOPPING,
            verbose: bool = True,
            category_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        super().__init__()
        self.name = 'meta_early_stop'
        self.meta_model = meta_model
        self.feature_columns = feature_columns
        self.stopping_threshold = stopping_threshold
        self.exceedance_threshold = exceedance_threshold
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
                if col in features_dict and features_dict[col] in mapping:
                    features_dict[col] = mapping[features_dict[col]]
                elif col in features_dict and isinstance(features_dict[col], str):
                    features_dict[col] = -1

            features_df = pd.DataFrame([features_dict])
            print(features_df)

            return features_df
        except Exception as e:
            if self.verbose:
                print(f"  [Step {step}] Feature extraction failed: {e}")
            return None

    def _extract_features_old(self, pl_module, step: int) -> Optional[pd.DataFrame]:
        """Extract WeightWatcher features from the model."""
        try:
            watcher = ww.WeightWatcher(model=pl_module)
            details = watcher.analyze(plot=False)
            summary = watcher.get_summary(details)
            summary['step'] = step

            features_dict = {col: summary.get(col, np.nan) for col in self.feature_columns}
            return pd.DataFrame([features_dict])
        except Exception as e:
            if self.verbose:
                print(f"  [Step {step}] Feature extraction failed: {e}")
            return None

    def _predict_exceedance(self, features: pd.DataFrame) -> float:
        """Predict probability of exceeding baseline performance.
        
        Uses raw conformal probabilities (no calibration) for speed.
        For early stopping, ranking accuracy matters more than calibration.
        """
        prob = self.meta_model.prob_exceeds(
            features[self.feature_columns],
            self.exceedance_threshold,
            calibration_method="none",
            # calibration_method="isotonic",
        )
        return float(prob[0])

    @staticmethod
    def get_cb(nf) -> "MetaModelEarlyStopCallback":
        """Retrieve the actual callback instance from a fitted NeuralForecast model.
        
        NeuralForecast deep-copies callbacks, so the original instance won't have
        the updated state. Use this method to get the actual callback after fitting.
        """
        all_cbs = nf.models[0].trainer_kwargs.get('callbacks', [])
        for cb in all_cbs:
            if getattr(cb, 'name', None) == 'meta_early_stop':
                return cb
        raise ValueError("MetaModelEarlyStopCallback not found in model callbacks")


class ClassifierEarlyStopCallback(Callback):
    """Early stopping callback based on a classifier predicting exceedance.

    Uses a pre-trained binary classifier to predict P(MASE > MASE_baseline) from
    WeightWatcher features during training. Stops if probability exceeds threshold.
    
    This is simpler than the regression + conformal approach since the classifier
    directly outputs P(exceeds baseline).
    """

    MIN_STEPS_BEFORE_STOPPING = 50

    def __init__(
            self,
            meta_classifier: CatBoostAUCClassifier,
            feature_columns: list[str],
            config_data: Dict,
            category_mappings,
            stopping_threshold: float = 0.5,
            every_n_steps: int = CB_N_STEPS,
            min_steps: int = MIN_STEPS_BEFORE_STOPPING,
            verbose: bool = True,
    ):
        super().__init__()
        self.name = 'classifier_early_stop'
        self.meta_classifier = meta_classifier
        self.feature_columns = feature_columns
        self.stopping_threshold = stopping_threshold
        self.every_n_steps = every_n_steps
        self.min_steps = min_steps
        self.verbose = verbose
        self.config_data = config_data
        self.category_mappings=category_mappings

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
                # if k == 'scaler_type' and v is None:
                #     smr_stats[k] = 'none'
                smr_stats[k] = v

            smr_stats['learning_rate'] = NeuralWeightsFeatureEng.bin_learning_rate(smr_stats['learning_rate'])
            smr_stats['start_padding_enabled'] = int(smr_stats['start_padding_enabled'])

            features_dict = {col: smr_stats.get(col, np.nan) for col in self.feature_columns}

            for col, mapping in self.category_mappings.items():
                if col in features_dict and features_dict[col] in mapping:
                    features_dict[col] = mapping[features_dict[col]]
                elif col in features_dict and isinstance(features_dict[col], str):
                    features_dict[col] = -1

            features_df = pd.DataFrame([features_dict])
            print(features_df)

            return features_df
        except Exception as e:
            if self.verbose:
                print(f"  [Step {step}] Feature extraction failed: {e}")
            return None

    def _predict_exceedance(self, features: pd.DataFrame) -> float:
        """Predict probability of exceeding baseline performance."""
        prob = self.meta_classifier.predict_proba_positive(
            features[self.feature_columns],
            calibrated=True,
        )
        return float(prob[0])

    @staticmethod
    def get_cb(nf) -> "ClassifierEarlyStopCallback":
        """Retrieve the actual callback instance from a fitted NeuralForecast model."""
        all_cbs = nf.models[0].trainer_kwargs.get('callbacks', [])
        for cb in all_cbs:
            if getattr(cb, 'name', None) == 'classifier_early_stop':
                return cb
        raise ValueError("ClassifierEarlyStopCallback not found in model callbacks")
