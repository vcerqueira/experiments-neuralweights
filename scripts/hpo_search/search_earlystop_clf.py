from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase

from src.algorithms import CatBoostAUCClassifier
from src.config import N_SAMPLES, SEED, TRY_MPS
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.nf_arch import ModelsConfig
from src.neural.param_samples import ConfigSampler
from src.utils import (
    read_all_metadata,
    build_meta_xy,
    load_dataset_splits,
)
from src.early_stopping import ClassifierEarlyStopCallback

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

STOPPING_THRESHOLD = 0.70
N_TRIALS = 10
CB_N_STEPS = 100

model_name = 'MLP'

metadata, category_mappings = read_all_metadata(
    './assets',
    model_name,
    processed_file=f'./assets/metadata_{model_name}.csv',
    sample_n=150000,
)


def train_meta_classifier(
        df: pd.DataFrame,
        calibrate: bool = True,
        cal_size: float = 0.2,
) -> tuple[CatBoostAUCClassifier, list[str]]:
    """Train a binary classifier to predict exceedance (MASE > MASE_baseline).
    
    Args:
        df: Training metadata (excluding target dataset).
        calibrate: Whether to calibrate probabilities.
        cal_size: Fraction of data for calibration set.
    
    Returns:
        Trained CatBoostAUCClassifier and list of feature column names.
    """
    data = build_meta_xy(df, task="classification", use_step_as_feature=True)

    clf = CatBoostAUCClassifier(
        calibrate=calibrate,
        calibration_method="isotonic",
        cal_size=cal_size,
    )
    clf.fit(data.X, data.y)

    print(f"Meta-classifier trained with {len(data.feature_columns)} features")
    print(f"  Class distribution: {data.y.mean():.1%} beats baseline")
    return clf, data.feature_columns


target_dataset = 'monash_m3_monthly'
meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)

train_full, train, valid, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
    target_dataset, get_valid=True
)
mase_func = partial(mase, seasonality=seas_len)

meta_classifier, feature_columns = train_meta_classifier(meta_train, calibrate=True)

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
print(f"Baseline validation MASE (SeasonalNaive): {mase_sn:.4f}")

search_results = []
configs_tried = 0
for config_sample in config_list:
    if configs_tried >= N_TRIALS:
        break

    cfg_id = config_sample.pop('config_id')
    configs_tried += 1

    print(f"\n[Config {configs_tried}/{N_TRIALS}] {cfg_id}")

    early_stop_cb = ClassifierEarlyStopCallback(
        meta_classifier=meta_classifier,
        feature_columns=feature_columns,
        stopping_threshold=STOPPING_THRESHOLD,
        every_n_steps=CB_N_STEPS,
        min_steps=30,
        verbose=True,
        config_data=config_sample,
        category_mappings=category_mappings,
    )

    nf_model_cb = ModelsConfig.create_model_instance(
        model_class=model_name,
        model_config=config_sample.copy(),
        horizon=horizon,
        input_size=n_lags,
        try_mps=TRY_MPS,
        callbacks=[early_stop_cb],
    )

    nf_model_nocb = ModelsConfig.create_model_instance(
        model_class=model_name,
        model_config=config_sample.copy(),
        horizon=horizon,
        input_size=n_lags,
        try_mps=TRY_MPS,
        callbacks=[],
        alias=f'{model_name}-NoCB'
    )

    nf = NeuralForecast(models=[nf_model_cb, nf_model_nocb], freq=freq)
    nf.fit(df=train)

    actual_cb = ClassifierEarlyStopCallback.get_cb(nf)

    fcst = nf.predict()
    fcst['ds'] = valid['ds']

    holdout = valid.merge(fcst, how='left', on=['unique_id', 'ds'])

    mase_model = mase_func(holdout, models=[model_name, f'{model_name}-NoCB'], train_df=train)

    result = {
        'config_id': cfg_id,
        'dataset': target_dataset,
        'model': model_name,
        'valid_mase_cb': float(mase_model[model_name].mean()),
        'valid_mase_nocb': float(mase_model[f'{model_name}-NoCB'].mean()),
        'valid_mase_sn': mase_sn,
        'stopped_early': actual_cb.stopped_early,
        'stop_step': actual_cb.stop_step,
        'n_predictions': len(actual_cb.predictions),
    }

    if actual_cb.predictions:
        result['final_prob_exceed'] = actual_cb.predictions[-1]['prob_exceed']
    else:
        result['final_prob_exceed'] = np.nan

    exceeds_baseline = result['valid_mase_cb'] > mase_sn
    status = "WORSE" if exceeds_baseline else "BETTER"
    result['exceeds_baseline'] = exceeds_baseline
    result['status'] = status

    early_str = f" (stopped @ step {result['stop_step']})" if result['stopped_early'] else ""
    print(f"  MASE(cb)={result['valid_mase_cb']:.4f}, MASE(nocb)={result['valid_mase_nocb']:.4f} "
          f"vs baseline={mase_sn:.4f} -> {status}{early_str}")

    search_results.append(result)

results_df = pd.DataFrame(search_results)

results_df['exceeds_baseline'] = (results_df['valid_mase_cb'] > results_df['valid_mase_sn']).astype(int)

print("\n" + "=" * 60)
print("SEARCH SUMMARY")
print("=" * 60)

auc = roc_auc_score(results_df['exceeds_baseline'], results_df['final_prob_exceed'])
ll = log_loss(results_df['exceeds_baseline'], results_df['final_prob_exceed'])
print(f"Meta-classifier AUC: {auc:.3f}, Log Loss: {ll:.3f}")

print(f"Configs tried: {len(results_df)}")
print(f"Early stopped: {results_df['stopped_early'].sum()} ({100 * results_df['stopped_early'].mean():.1f}%)")
print(f"Exceeded baseline: {results_df['exceeds_baseline'].sum()} ({100 * results_df['exceeds_baseline'].mean():.1f}%)")

print("\n",
      results_df[['config_id', 'valid_mase_cb', 'valid_mase_nocb', 'stopped_early', 'final_prob_exceed', 'status']])




