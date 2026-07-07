from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase
from functools import partial

from src.algorithms import CatBoostRegressionModel
from src.config import N_SAMPLES, SEED, TRY_MPS, MAX_SAMPLES
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.nf_arch import ModelsConfig
from src.neural.param_samples import ConfigSampler
from src.utils import load_dataset_splits, read_all_metadata
from src.early_stopping import MetaModelEarlyStopCallback

warnings.filterwarnings('ignore')

STOPPING_THRESHOLD = 0.80
N_TRIALS = 10
CB_N_STEPS = 50

data_dir = Path('./assets/results')
model_name = 'MLP'

metadata = read_all_metadata(data_dir, model_name, detailed=False)
df_after_train = metadata.sample(100000).reset_index(drop=True)

target_dataset = 'monash_tourism_monthly'

meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)


def train_meta_model(
        meta_train,
        conformal_cal_size: float = 0.3,
        y_clip: tuple[float, float] = (-2.5, 2.5),
) -> tuple[CatBoostRegressionModel, list[str]]:
    """Train meta-model on all datasets except the target (LOO).
    
    Args:
        data_dir: Directory containing metadata CSV files.
        model_name: Name of the neural model (e.g., 'MLP').
        target_dataset: Dataset to exclude (will be used for testing).
        conformal_cal_size: Fraction of data for conformal calibration.
        y_clip: Min/max clipping for target variable.
    
    Returns:
        Trained CatBoostRegressionModel and list of feature column names.
    """

    y_reg = meta_train['mase_sn'] - meta_train['mase']
    y_reg = np.clip(y_reg, a_min=y_clip[0], a_max=y_clip[1])

    feature_cols = [
        col for col in meta_train.columns
        if col not in ['mase', 'mase_sn',
                       'model', 'config_id',
                       'step',
                       'dataset']
    ]
    X = meta_train[feature_cols]

    reg = CatBoostRegressionModel(
        conformal=True,
        conformal_cal_size=conformal_cal_size,
        calibration_method="isotonic",
    )
    reg.fit(X, y_reg)

    print(f"Meta-model trained with {len(feature_cols)} features")
    return reg, feature_cols


train, valid, test, horizon, n_lags, freq, seas_len = load_dataset_splits(target_dataset, get_valid=True)
mase_func = partial(mase, seasonality=seas_len)

meta_model, feature_columns = train_meta_model(meta_train)

config_pool = NEURAL_CONFIG_POOL[model_name]
config_list = ConfigSampler.generate_samples(
    config_pool=config_pool,
    num_samples=N_SAMPLES,
    random_state=SEED,
)

sf = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq)
sf.fit(train)
fcst_sf = sf.predict(h=horizon)
fcst_sf['ds'] = valid['ds']
holdout = valid.merge(fcst_sf, how='left', on=['unique_id', 'ds'])
mase_sn = mase_func(holdout, models=['SeasonalNaive'], train_df=train).mean(numeric_only=True)['SeasonalNaive']

search_results = []
configs_tried = 0
for config_sample in config_list:
    if configs_tried >= N_TRIALS:
        break

    cfg_id = config_sample.pop('config_id')

    configs_tried += 1

    early_stop_cb = MetaModelEarlyStopCallback(
        meta_model=meta_model,
        feature_columns=feature_columns,
        stopping_threshold=STOPPING_THRESHOLD,
        exceedance_threshold=0.0,
        every_n_steps=CB_N_STEPS,
        min_steps=30,
        verbose=True,
    )

    model = ModelsConfig.create_model_instance(
        model_class=model_name,
        model_config=config_sample.copy(),
        horizon=horizon,
        input_size=n_lags,
        try_mps=TRY_MPS,
        callbacks=[early_stop_cb],
    )

    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train)

    actual_cb = MetaModelEarlyStopCallback.get_cb(nf)

    fcst = nf.predict()
    fcst['ds'] = valid['ds']

    holdout = valid.merge(fcst, how='left', on=['unique_id', 'ds'])

    mase_model = mase_func(holdout, models=[model_name], train_df=train)

    result = {
        'config_id': cfg_id,
        'dataset': target_dataset,
        'model': model_name,
        'mase': float(mase_model[model_name].mean()),
        'stopped_early': actual_cb.stopped_early,
        'stop_step': actual_cb.stop_step,
        'n_predictions': len(actual_cb.predictions),
    }

    if actual_cb.predictions:
        result['final_prob_exceed'] = actual_cb.predictions[-1]['prob_exceed']
    else:
        result['final_prob_exceed'] = np.nan

    search_results.append(result)
    # del early_stop_cb

    exceeds_baseline = result['mase'] > mase_sn
    status = "WORSE" if exceeds_baseline else "BETTER"
    result['status'] = status

results_df = pd.DataFrame(search_results)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

results_df
