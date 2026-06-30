import weightwatcher as ww
from pytorch_lightning.callbacks import Callback


class WeightWatcherCallback(Callback):
    """Run WeightWatcher every `every_n_steps` optimizer steps."""


    """
        
    import weightwatcher as ww
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    watcher = ww.WeightWatcher(model=nf.models[0])
    details = watcher.analyze(plot=False)
    summary = watcher.get_summary(details)
    print(pd.Series(summary))
    print(details.T)
    
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
        watcher = ww.WeightWatcher(model=pl_module)
        details = watcher.analyze(plot=False)
        summary = watcher.get_summary(details)
        summary['step'] = step
        details['step'] = step

        return summary, details

    @staticmethod
    def get_cb(nf):
        # todo getting from the first model only
        all_cbs = nf.models[0].trainer_kwargs['callbacks']
        ww_cb = next(cb for cb in all_cbs if cb.name == "weightwatcher")

        return ww_cb

