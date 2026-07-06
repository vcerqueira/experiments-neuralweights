"""
Random search with meta-model based early stopping.

This script implements uncertainty-aware early stopping for HPO:
1. Train a meta-model on historical data (excluding target dataset) using LOO
2. During training, predict P(MASE > MASE_baseline) using WeightWatcher features
3. If P > threshold, stop training early and move to next configuration

This enables efficient HPO by avoiding full training of likely-poor configurations.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase
from functools import partial

from src.algorithms import CatBoostRegressionModel
from src.config import N_SAMPLES, SEED, TRY_MPS, MAX_SAMPLES, CB_N_STEPS
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.nf_arch import ModelsConfig
from src.neural.param_samples import ConfigSampler
from src.utils import load_dataset_splits, read_all_metadata
from src.early_stopping import MetaModelEarlyStopCallback

warnings.filterwarnings('ignore')

EXCEEDANCE_THRESHOLD = 0.75
N_TRIALS = 10

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


train, test, horizon, n_lags, freq, seas_len = load_dataset_splits(target_dataset)
mase_func = partial(mase, seasonality=seas_len)

meta_model, feature_columns = train_meta_model(meta_train)

config_pool = NEURAL_CONFIG_POOL[model_name]
config_list = ConfigSampler.generate_samples(
    config_pool=config_pool,
    num_samples=N_SAMPLES,
    random_state=SEED,
)

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
        stopping_threshold=0.6,
        exceedance_threshold=0.0,
        every_n_steps=CB_N_STEPS,
        min_steps=50,
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

    sf = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq)

    nf = NeuralForecast(models=[model], freq=freq)
    sf.fit(train)

    try:
        nf.fit(df=train)
    except Exception as e:
        print(f"  Training failed: {e}")
        continue

    fcst_sf = sf.predict(h=horizon)
    fcst = nf.predict()
    fcst['ds'] = test['ds']
    fcst_sf['ds'] = test['ds']

    holdout = test.merge(fcst, how='left', on=['unique_id', 'ds'])
    holdout = holdout.merge(fcst_sf, how='left', on=['unique_id', 'ds'])

    mase_model = mase_func(holdout, models=[model_name])
    mase_sn = mase_func(holdout, models=['SeasonalNaive'])

    result = {
        'config_id': cfg_id,
        'dataset': target_dataset,
        'model': model_name,
        'mase': float(mase_model[model_name].mean()),
        'mase_sn': float(mase_sn['SeasonalNaive'].mean()),
        'stopped_early': early_stop_cb.stopped_early,
        'stop_step': early_stop_cb.stop_step,
        'n_predictions': len(early_stop_cb.predictions),
    }

    if early_stop_cb.predictions:
        result['final_prob_exceed'] = early_stop_cb.predictions[-1]['prob_exceed']
    else:
        result['final_prob_exceed'] = np.nan

    search_results.append(result)

    exceeds_baseline = result['mase'] > result['mase_sn']
    status = "WORSE" if exceeds_baseline else "BETTER"
    early_str = f" (stopped @ step {result['stop_step']})" if result['stopped_early'] else ""

    pd.DataFrame([result]).to_csv(result_fp, index=False)

results_df = pd.DataFrame(search_results)
