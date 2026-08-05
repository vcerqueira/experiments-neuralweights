"""Weightcast integration with NeuralForecast AutoModels.

Provides config wrappers that inject Weightcast callbacks into Optuna-based
hyperparameter search, enabling meta-model early stopping during AutoModel fitting.
"""
from typing import Callable, Optional, Union

import optuna
from pytorch_lightning.callbacks import Callback

from src.weightcast.callbacks import WeightcastClassifier, WeightcastRegressor
from src.weightcast.learner_classifier import CatBoostAUCClassifier
from src.weightcast.learner_regressor import CatBoostRegressionModel


class StepAccumulator:
    """Shared accumulator for step counts across trials.

    Uses a class-level registry to survive callback deep-copying by PyTorch Lightning.
    Each accumulator has a unique ID that callbacks use to find the registry entry.

    Example:
        >>> accumulator = StepAccumulator()
        >>> config_fn = WeightcastAutoConfig(..., step_accumulator=accumulator)
        >>> # After all trials:
        >>> print(f"Total steps: {accumulator.total_steps}")
    """

    _registry: dict[str, "StepAccumulator"] = {}
    _counter = 0

    def __init__(self):
        StepAccumulator._counter += 1
        self.id = f"acc_{StepAccumulator._counter}"
        self.total_steps = 0
        self.trial_steps: list[int] = []
        StepAccumulator._registry[self.id] = self

    def add_trial(self, steps: int):
        self.total_steps += steps
        self.trial_steps.append(steps)

    def reset(self):
        self.total_steps = 0
        self.trial_steps = []

    @classmethod
    def get(cls, acc_id: str) -> "StepAccumulator":
        """Retrieve accumulator by ID from class-level registry."""
        return cls._registry[acc_id]


class StepCounterCallback(Callback):
    """Lightweight callback that counts training steps for a single trial.

    Uses accumulator ID to look up the shared StepAccumulator from class registry,
    surviving deep-copy by PyTorch Lightning.
    """

    def __init__(self, accumulator_id: str):
        super().__init__()
        self.name = 'step_counter'
        self.accumulator_id = accumulator_id
        self._current_trial_steps = 0
        self._recorded = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._current_trial_steps += 1

    def _record_steps(self):
        """Record steps to accumulator (called once per trial)."""
        if not self._recorded and self._current_trial_steps > 0:
            accumulator = StepAccumulator.get(self.accumulator_id)
            accumulator.add_trial(self._current_trial_steps)
            self._recorded = True

    def on_train_end(self, trainer, pl_module):
        self._record_steps()

    def on_exception(self, trainer, pl_module, exception):
        self._record_steps()


class WeightcastAutoConfig:
    """Config factory for AutoModels with Weightcast early stopping.

    Wraps a config sampler and injects the appropriate Weightcast callback
    (classifier or regressor) into each sampled config. Automatically detects
    the meta-model type and creates the corresponding callback.

    Args:
        config_sampler: Callable that takes an Optuna trial and returns a config dict.
        model_name: Name of the model (e.g., 'MLP', 'NHITS', 'PatchTST').
        meta_model: Trained meta-model (CatBoostAUCClassifier or CatBoostRegressionModel).
        feature_columns: Feature columns expected by the meta-model.
        category_mappings: Category mappings for encoding categorical features.
        stopping_threshold: Probability threshold for early stopping (default: 0.5).
        exceedance_threshold: For regressor only - value threshold for P(Y > threshold).
        cb_n_steps: Check callback every N training steps (default: 100).
        min_steps: Minimum steps before callback activates (default: 50).
        verbose: Whether to print callback predictions (default: True).
        step_accumulator: Optional StepAccumulator to track total steps across trials.

    Example:
        >>> accumulator = StepAccumulator()
        >>> config_fn = WeightcastAutoConfig(
        ...     config_sampler=CONFIG_SAMPLERS['MLP'](input_size=24),
        ...     model_name='MLP',
        ...     meta_model=classifier,  # or regressor
        ...     feature_columns=features,
        ...     category_mappings=mappings,
        ...     step_accumulator=accumulator,
        ... )
        >>> auto_model = AutoMLP(h=12, config=config_fn, backend='optuna', ...)
        >>> nf = NeuralForecast(models=[auto_model], freq='D')
        >>> nf.fit(df=train)
        >>> print(f"Total steps: {accumulator.total_steps}")
    """

    def __init__(
            self,
            config_sampler: Callable[[optuna.Trial], dict],
            model_name: str,
            meta_model: Union[CatBoostAUCClassifier, CatBoostRegressionModel],
            feature_columns: list[str],
            category_mappings: dict,
            stopping_threshold: float = 0.5,
            exceedance_threshold: float = 0.0,
            cb_n_steps: int = 100,
            min_steps: int = 50,
            verbose: bool = True,
            step_accumulator: Optional[StepAccumulator] = None,
    ):
        self.config_sampler = config_sampler
        self.model_name = model_name
        self.meta_model = meta_model
        self.feature_columns = feature_columns
        self.category_mappings = category_mappings
        self.stopping_threshold = stopping_threshold
        self.exceedance_threshold = exceedance_threshold
        self.cb_n_steps = cb_n_steps
        self.min_steps = min_steps
        self.verbose = verbose
        self.accumulator_id = step_accumulator.id if step_accumulator is not None else None

        self._is_classifier = isinstance(meta_model, CatBoostAUCClassifier)

    def __call__(self, trial: optuna.Trial) -> dict:
        """Sample config and inject Weightcast callback."""
        config = self.config_sampler(trial)
        config_data = self._prepare_config_data(config)

        if self._is_classifier:
            callback = WeightcastClassifier(
                meta_classifier=self.meta_model,
                feature_columns=self.feature_columns,
                config_data=config_data,
                category_mappings=self.category_mappings,
                stopping_threshold=self.stopping_threshold,
                every_n_steps=self.cb_n_steps,
                min_steps=self.min_steps,
                verbose=self.verbose,
            )
        else:
            callback = WeightcastRegressor(
                meta_model=self.meta_model,
                feature_columns=self.feature_columns,
                config_data=config_data,
                category_mappings=self.category_mappings,
                stopping_threshold=self.stopping_threshold,
                exceedance_threshold=self.exceedance_threshold,
                every_n_steps=self.cb_n_steps,
                min_steps=self.min_steps,
                verbose=self.verbose,
            )

        callbacks = [callback]
        if self.accumulator_id is not None:
            callbacks.append(StepCounterCallback(self.accumulator_id))

        config["callbacks"] = callbacks
        return config

    def _prepare_config_data(self, config: dict) -> dict:
        """Prepare config_data dict for the callback with expected field names."""
        config_data = config.copy()
        config_data.pop('callbacks', None)

        input_size = config_data.pop('input_size', None)
        if input_size is not None:
            config_data['input_size_multiplier'] = 1

        config_data['model'] = self.model_name
        return config_data

