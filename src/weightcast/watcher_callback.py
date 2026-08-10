"""Collect WeightWatcher layer statistics during NeuralForecast training.

Used offline for metadata collection (pre_metadata_collection scripts).
For online early stopping, see ``src.weightcast.callbacks``.
"""
import weightwatcher as ww
from pytorch_lightning.callbacks import Callback


class WeightWatcherCallback(Callback):
    """Snapshot WeightWatcher summary/details every ``every_n_steps`` steps.

    Also records snapshots at train start and train end.

    Attributes:
        summaries: List of (step-tagged) WeightWatcher summary dicts/Series.
        details: List of (step-tagged) per-layer detail DataFrames.

    Example:
        >>> cb = WeightWatcherCallback(every_n_steps=100)
        >>> model = MLP(..., callbacks=[cb])
        >>> nf.fit(df)
        >>> actual = WeightWatcherCallback.get_cb(nf)
        >>> len(actual.details)
    """

    def __init__(self, every_n_steps: int = 10):
        super().__init__()
        self.name = 'weightwatcher'
        self.every_n_steps = every_n_steps
        self.summaries: list[tuple[int, object]] = []
        self.details: list[tuple[int, object]] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step

        if step == 0 or step % self.every_n_steps != 0:
            return

        w_summary, w_details = self._get_ww_data(pl_module, step)

        self.summaries.append(w_summary)
        self.details.append(w_details)

    def on_train_start(self, trainer, pl_module):
        w_summary, w_details = self._get_ww_data(pl_module, 0)

        self.summaries.append(w_summary)
        self.details.append(w_details)

    def on_train_end(self, trainer, pl_module):
        w_summary, w_details = self._get_ww_data(pl_module, -1)

        self.summaries.append(w_summary)
        self.details.append(w_details)

    @staticmethod
    def _get_ww_data(pl_module, step: int):
        """Run WeightWatcher.analyze and tag outputs with ``step``."""
        watcher = ww.WeightWatcher(model=pl_module)
        details = watcher.analyze(plot=False)
        summary = watcher.get_summary(details)
        summary['step'] = step
        details['step'] = step

        return summary, details

    @staticmethod
    def get_cb(nf) -> "WeightWatcherCallback":
        """Retrieve the callback from the first model in a fitted NeuralForecast.

        NeuralForecast deep-copies callbacks, so use this after ``fit``.

        # todo this is getting the cb from the first model only
        """
        all_cbs = nf.models[0].trainer_kwargs['callbacks']
        ww_cb = next(cb for cb in all_cbs if cb.name == "weightwatcher")

        return ww_cb

