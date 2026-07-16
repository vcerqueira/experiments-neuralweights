import warnings
from typing import Callable, Optional

import optuna
from pytorch_lightning.callbacks import Callback

from src.early_stopping import ClassifierEarlyStopCallback

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StepAccumulator:
    """Shared accumulator for step counts across trials.
    
    Uses a class-level registry to survive callback deep-copying by PyTorch Lightning.
    Each accumulator has a unique ID that callbacks use to find the registry entry.
    
    Example:
        >>> accumulator = StepAccumulator()
        >>> # Pass to config function, which creates StepCounterCallback(accumulator.id)
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
    
    Example:
        >>> accumulator = StepAccumulator()
        >>> callback = StepCounterCallback(accumulator.id)
        >>> # After all trials:
        >>> print(f"Total steps: {accumulator.total_steps}")
    """

    def __init__(self, accumulator_id: str):
        super().__init__()
        self.name = 'step_counter'
        self.accumulator_id = accumulator_id
        self._current_trial_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._current_trial_steps += 1

    def on_train_end(self, trainer, pl_module):
        accumulator = StepAccumulator.get(self.accumulator_id)
        accumulator.add_trial(self._current_trial_steps)
        self._current_trial_steps = 0


class ConfigWithStepCounter:
    """Wrapper that adds a step counter callback to any config sampler.
    
    Creates a NEW StepCounterCallback for each trial, all sharing the same
    StepAccumulator via ID lookup (survives deep-copy).
    
    Args:
        config_sampler: Callable that takes an Optuna trial and returns a config dict.
        accumulator: StepAccumulator to track steps across all trials.
    
    Example:
        >>> accumulator = StepAccumulator()
        >>> config_fn = ConfigWithStepCounter(config_sampler, accumulator)
        >>> auto_model = AutoMLP(config=config_fn, ...)
        >>> # After fit:
        >>> print(f"Total steps: {accumulator.total_steps}")
    """

    def __init__(
            self,
            config_sampler: Callable[[optuna.Trial], dict],
            accumulator: StepAccumulator,
    ):
        self.config_sampler = config_sampler
        self.accumulator_id = accumulator.id

    def __call__(self, trial: optuna.Trial) -> dict:
        config = self.config_sampler(trial)
        step_counter = StepCounterCallback(self.accumulator_id)
        existing_callbacks = config.get("callbacks", [])
        config["callbacks"] = existing_callbacks + [step_counter]
        return config


class AutoConfigWithCallback:
    """Callable config factory for AutoModels with meta-model callback injection.

    Wraps a config sampler function and automatically injects a
    ClassifierEarlyStopCallback into each sampled config.

    Args:
        config_sampler: Callable that takes an Optuna trial and returns a config dict.
        model_name: Name of the model (used for preparing config_data).
        meta_classifier: Trained meta-classifier for early stopping.
        feature_columns: Feature columns expected by the meta-classifier.
        category_mappings: Category mappings for encoding.
        stopping_threshold: Threshold for early stopping decision.
        cb_n_steps: Check callback every N steps.
        min_steps: Minimum steps before callback activates.
        verbose: Whether to print callback predictions.
        step_accumulator: Optional StepAccumulator to track steps across trials.

    Example:
        >>> accumulator = StepAccumulator()
        >>> sampler = mlp_config_sampler(input_size=24)
        >>> config_fn = AutoConfigWithCallback(
        ...     config_sampler=sampler,
        ...     model_name='MLP',
        ...     meta_classifier=clf,
        ...     feature_columns=features,
        ...     category_mappings=mappings,
        ...     step_accumulator=accumulator,
        ... )
        >>> auto_model = AutoMLP(h=12, config=config_fn, ...)
    """

    def __init__(
            self,
            config_sampler: Callable[[optuna.Trial], dict],
            model_name: str,
            meta_classifier,
            feature_columns: list[str],
            category_mappings: dict,
            stopping_threshold: float = 0.70,
            cb_n_steps: int = 100,
            min_steps: int = 50,
            verbose: bool = True,
            step_accumulator: Optional[StepAccumulator] = None,
    ):
        self.config_sampler = config_sampler
        self.model_name = model_name
        self.meta_classifier = meta_classifier
        self.feature_columns = feature_columns
        self.category_mappings = category_mappings
        self.stopping_threshold = stopping_threshold
        self.cb_n_steps = cb_n_steps
        self.min_steps = min_steps
        self.verbose = verbose
        self.accumulator_id = step_accumulator.id if step_accumulator is not None else None

    def __call__(self, trial: optuna.Trial) -> dict:
        """Sample config and inject early stopping callback."""
        config = self.config_sampler(trial)

        callback = ClassifierEarlyStopCallback(
            meta_classifier=self.meta_classifier,
            feature_columns=self.feature_columns,
            config_data=self._prepare_config_data(config),
            category_mappings=self.category_mappings,
            stopping_threshold=self.stopping_threshold,
            every_n_steps=self.cb_n_steps,
            min_steps=self.min_steps,
            verbose=self.verbose,
        )

        callbacks = [callback]
        if self.accumulator_id is not None:
            step_counter = StepCounterCallback(self.accumulator_id)
            callbacks.append(step_counter)

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


def mlp_config_sampler(input_size: int) -> Callable[[optuna.Trial], dict]:
    """Create config sampler for MLP model."""

    def sampler(trial: optuna.Trial) -> dict:
        return {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 1024]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "standard"]),
            "max_steps": trial.suggest_categorical("max_steps", [500, 1000, 2000, 5000]),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

    return sampler


def nhits_config_sampler(input_size: int) -> Callable[[optuna.Trial], dict]:
    """Create config sampler for NHITS model."""

    def sampler(trial: optuna.Trial) -> dict:
        return {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2]),
            "n_pool_kernel_size": trial.suggest_categorical("n_pool_kernel_size", [
                [2, 2, 1], [3, 2, 1], [6, 2, 1], [8, 4, 1],
                [1, 1, 1], [2, 2, 2], [4, 4, 4], [24, 8, 2], [16, 8, 1]
            ]),
            "n_freq_downsample": trial.suggest_categorical("n_freq_downsample", [
                [168, 24, 1], [24, 12, 1], [60, 8, 1], [40, 20, 1],
                [6, 2, 1], [24, 8, 2], [1, 1, 1],
            ]),
            "mlp_units": trial.suggest_categorical("mlp_units", [
                3 * [[64, 64]], 3 * [[128, 128]], 3 * [[256, 256]], 3 * [[512, 512]],
            ]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "revin", "standard"]),
            "max_steps": trial.suggest_int("max_steps", 500, 2000, step=100),
            "pooling_mode": trial.suggest_categorical("pooling_mode", ['MaxPool1d', 'AvgPool1d']),
            "interpolation_mode": trial.suggest_categorical("interpolation_mode", ['linear', 'nearest', 'cubic']),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "dropout_prob_theta": trial.suggest_categorical("dropout_prob_theta", [0.0, 0.1, 0.2]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

    return sampler


def patchtst_config_sampler(input_size: int) -> Callable[[optuna.Trial], dict]:
    """Create config sampler for PatchTST model."""

    def sampler(trial: optuna.Trial) -> dict:
        return {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2, input_size * 3]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 128, 256]),
            "linear_hidden_size": trial.suggest_categorical("linear_hidden_size", [64, 128, 256]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8, 16]),
            "encoder_layers": trial.suggest_categorical("encoder_layers", [1, 2, 3]),
            "patch_len": trial.suggest_categorical("patch_len", [16, 24]),
            "stride": trial.suggest_categorical("stride", [2, 4, 8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "standard"]),
            "revin": trial.suggest_categorical("revin", [False, True]),
            "max_steps": trial.suggest_categorical("max_steps", [500, 1000, 2000, 5000]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            "res_attention": trial.suggest_categorical("res_attention", [True, False]),
            "batch_normalization": trial.suggest_categorical("batch_normalization", [True, False]),
            "learn_pos_embed": trial.suggest_categorical("learn_pos_embed", [True, False]),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

    return sampler


CONFIG_SAMPLERS = {
    'MLP': mlp_config_sampler,
    'NHITS': nhits_config_sampler,
    'PatchTST': patchtst_config_sampler,
}
