"""Config wrappers for Optuna-based HPO with step counting and pruning.

Provides utilities for integrating Optuna pruners (MedianPruner, SuccessiveHalvingPruner,
HyperbandPruner) with NeuralForecast AutoModels, and tracking training steps across trials.
"""
from typing import Callable

import optuna
from pytorch_lightning.callbacks import Callback


class TrialRegistry:
    """Class-level registry for Optuna trials to survive callback deep-copying.

    PyTorch Lightning deep-copies callbacks, breaking references to trial objects.
    This registry stores trials by ID so callbacks can look them up at runtime.
    """
    _trials: dict[str, optuna.Trial] = {}
    _counter = 0

    @classmethod
    def register(cls, trial: optuna.Trial) -> str:
        """Register a trial and return its ID."""
        cls._counter += 1
        trial_id = f"trial_{cls._counter}"
        cls._trials[trial_id] = trial
        return trial_id

    @classmethod
    def get(cls, trial_id: str) -> optuna.Trial:
        """Retrieve trial by ID."""
        return cls._trials[trial_id]

    @classmethod
    def remove(cls, trial_id: str):
        """Remove trial from registry (cleanup)."""
        cls._trials.pop(trial_id, None)


class StepAccumulator:
    """Shared accumulator for step counts across trials.

    Uses a class-level registry to survive callback deep-copying by PyTorch Lightning.
    Each accumulator has a unique ID that callbacks use to find the registry entry.

    Example:
        >>> accumulator = StepAccumulator()
        >>> config_fn = ConfigWithStepCounter(config_sampler, accumulator)
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


class OptunaPruningCallback(Callback):
    """PyTorch Lightning callback for Optuna pruning.

    Reports intermediate validation metrics to Optuna and handles pruning decisions.
    Uses TrialRegistry to look up the trial at runtime, surviving deep-copy.

    The callback stops training gracefully via trainer.should_stop and raises
    TrialPruned in on_fit_end, ensuring proper cleanup before the exception.

    Args:
        trial_id: ID of the trial in TrialRegistry.
        monitor: Metric name to monitor for pruning (e.g., 'valid_loss').
    """

    def __init__(self, trial_id: str, monitor: str = 'valid_loss'):
        super().__init__()
        self.name = 'optuna_pruning'
        self.trial_id = trial_id
        self.monitor = monitor
        self._epoch = 0
        self._pruned = False
        self._pruned_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is None:
            return

        try:
            trial = TrialRegistry.get(self.trial_id)
            trial.report(float(current_value), self._epoch)
            self._epoch += 1

            if trial.should_prune():
                self._pruned = True
                self._pruned_epoch = self._epoch
                trainer.should_stop = True
        except (KeyError, RuntimeError, optuna.exceptions.UpdateFinishedTrialError):
            pass

    def on_fit_end(self, trainer, pl_module):
        if self._pruned:
            try:
                trial = TrialRegistry.get(self.trial_id)
                if trial.state == optuna.trial.TrialState.RUNNING:
                    raise optuna.TrialPruned(f"Trial was pruned at epoch {self._pruned_epoch}.")
            except (KeyError, AttributeError):
                pass


class ConfigWithStepCounter:
    """Wrapper that adds step counting to any config sampler.

    Creates a StepCounterCallback for each trial, all sharing the same
    StepAccumulator via ID lookup (survives deep-copy).

    Example:
        >>> accumulator = StepAccumulator()
        >>> config_fn = ConfigWithStepCounter(config_sampler, accumulator)
        >>> auto_model = AutoMLP(config=config_fn, ...)
        >>> nf.fit(...)
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


class ConfigWithPruningCallback:
    """Wrapper that adds Optuna pruning and step counting to a config sampler.

    For Optuna pruners (MedianPruner, SuccessiveHalvingPruner, HyperbandPruner) to work,
    intermediate values must be reported during training via trial.report(). This wrapper
    injects OptunaPruningCallback which handles this automatically.

    Example:
        >>> accumulator = StepAccumulator()
        >>> config_fn = ConfigWithPruningCallback(config_sampler, accumulator)
        >>> auto_model = AutoMLP(
        ...     config=config_fn,
        ...     optuna_options=OptunaOptions(create_study_kwargs={"pruner": MedianPruner()}),
        ... )
    """

    def __init__(
            self,
            config_sampler: Callable[[optuna.Trial], dict],
            accumulator: StepAccumulator,
            monitor: str = 'valid_loss',
    ):
        self.config_sampler = config_sampler
        self.accumulator_id = accumulator.id
        self.monitor = monitor

    def __call__(self, trial: optuna.Trial) -> dict:
        config = self.config_sampler(trial)

        trial_id = TrialRegistry.register(trial)

        step_counter = StepCounterCallback(self.accumulator_id)
        pruning_callback = OptunaPruningCallback(trial_id, monitor=self.monitor)

        existing_callbacks = config.get("callbacks", [])
        config["callbacks"] = existing_callbacks + [step_counter, pruning_callback]
        return config

