import warnings

import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase
from functools import partial

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from src.loaders import ChronosDataset
from src.weights.watcher_callback import WeightWatcherCallback

warnings.filterwarnings('ignore')

# ---- data loading and partitioning
target = 'monash_m1_monthly'

# _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
_, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target, min_n_instances=2 * (n_lags + horizon))
mase_func = partial(mase, seasonality=seas_len)

train, test = ChronosDataset.time_wise_split(df, horizon)



ww_callback = WeightWatcherCallback(every_n_steps=10)

nf = NeuralForecast(
    models=[
        NHITS(
            h=horizon,
            input_size=2 * horizon,
            max_steps=100,
            callbacks=[ww_callback],
        )
    ],
    freq=freq,
)
nf.fit(df=train)

fcst = nf.predict()
fcst['ds'] = test['ds']

holdout_set = test.merge(fcst, how='left', on=['unique_id', 'ds'])

radar = ModelRadar(
    cv_df=holdout_set,
    metrics=[mase_func],
    model_names=['MLP'],
    train_df=train,
    hardness_reference='MLP',
    ratios_reference='MLP',
)

err_inner = radar.evaluate(keep_uids=False)



cb = WeightWatcherCallback.get_cb(nf)


cbs_df = pd.DataFrame(cb.summaries)
cbd_df = pd.concat(cb.details).reset_index(drop=True)
# todo adicionar nome do modelo
# todo adicionar config id
# todo horizon, n_lags, freq, seas_len

