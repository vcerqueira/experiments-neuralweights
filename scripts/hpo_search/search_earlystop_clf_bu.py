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
    print(f"  Class distribution: {data.y.mean():.1%} exceeds baseline (y=1)")
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
config_registry = {}

for config_sample in config_list:
    if configs_tried >= N_TRIALS:
        break

    cfg_id = config_sample.pop('config_id')
    config_registry[cfg_id] = config_sample.copy()
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

# =============================================================================
# FINAL EVALUATION: Train best configs on train_full, evaluate on test
# =============================================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

# Best config WITH callback (ignore early-stopped configs)
completed_runs = results_df[~results_df['stopped_early']]
if len(completed_runs) > 0:
    best_cb_row = completed_runs.loc[completed_runs['valid_mase_cb'].idxmin()]
    best_cb_config_id = best_cb_row['config_id']
    best_cb_config = config_registry[best_cb_config_id]
    print(f"\nBest config (with callback, completed): {best_cb_config_id}")
    print(f"  Validation MASE: {best_cb_row['valid_mase_cb']:.4f}")
else:
    best_cb_config_id = None
    best_cb_config = None
    print("\nNo completed runs with callback (all stopped early)")

# Best config WITHOUT callback (all configs)
best_nocb_row = results_df.loc[results_df['valid_mase_nocb'].idxmin()]
best_nocb_config_id = best_nocb_row['config_id']
best_nocb_config = config_registry[best_nocb_config_id]
print(f"\nBest config (without callback): {best_nocb_config_id}")
print(f"  Validation MASE: {best_nocb_row['valid_mase_nocb']:.4f}")

# Train on train_full, evaluate on test
print("\nTraining best configs on full training data...")

final_models = []
final_model_names = []

# Best from callback approach (if exists)
if best_cb_config is not None:
    nf_best_cb = ModelsConfig.create_model_instance(
        model_class=model_name,
        model_config=best_cb_config.copy(),
        horizon=horizon,
        input_size=n_lags,
        try_mps=TRY_MPS,
        callbacks=[],
        alias=f'{model_name}-BestCB',
    )
    final_models.append(nf_best_cb)
    final_model_names.append(f'{model_name}-BestCB')

# Best from no-callback approach
nf_best_nocb = ModelsConfig.create_model_instance(
    model_class=model_name,
    model_config=best_nocb_config.copy(),
    horizon=horizon,
    input_size=n_lags,
    try_mps=TRY_MPS,
    callbacks=[],
    alias=f'{model_name}-BestNoCB',
)
final_models.append(nf_best_nocb)
final_model_names.append(f'{model_name}-BestNoCB')

# Train neural models
nf_final = NeuralForecast(models=final_models, freq=freq)
nf_final.fit(df=train_full)
fcst_final = nf_final.predict()
fcst_final['ds'] = test['ds']

# Train Seasonal Naive on train_full
sf_final = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq)
sf_final.fit(train_full)
fcst_sf_final = sf_final.predict(h=horizon)
fcst_sf_final['ds'] = test['ds']

# Merge all forecasts
test_holdout = test.merge(fcst_final, how='left', on=['unique_id', 'ds'])
test_holdout = test_holdout.merge(fcst_sf_final, how='left', on=['unique_id', 'ds'])

# Compute test MASE for all models
all_test_models = final_model_names + ['SeasonalNaive']
mase_test = mase_func(test_holdout, models=all_test_models, train_df=train_full)

print("\n" + "-" * 40)
print("TEST SET RESULTS (MASE)")
print("-" * 40)

test_results = {}
for m in all_test_models:
    test_mase = float(mase_test[m].mean())
    test_results[m] = test_mase
    print(f"  {m}: {test_mase:.4f}")

# Summary comparison
print("\n" + "-" * 40)
print("SUMMARY")
print("-" * 40)
sn_test_mase = test_results['SeasonalNaive']
print(f"Baseline (SeasonalNaive): {sn_test_mase:.4f}")

if best_cb_config is not None:
    cb_test_mase = test_results[f'{model_name}-BestCB']
    cb_vs_sn = "BETTER" if cb_test_mase < sn_test_mase else "WORSE"
    cb_pct = 100 * (sn_test_mase - cb_test_mase) / sn_test_mase
    print(f"Best (with callback):     {cb_test_mase:.4f} ({cb_vs_sn}, {cb_pct:+.1f}% vs SN)")

nocb_test_mase = test_results[f'{model_name}-BestNoCB']
nocb_vs_sn = "BETTER" if nocb_test_mase < sn_test_mase else "WORSE"
nocb_pct = 100 * (sn_test_mase - nocb_test_mase) / sn_test_mase
print(f"Best (without callback):  {nocb_test_mase:.4f} ({nocb_vs_sn}, {nocb_pct:+.1f}% vs SN)")

# Compare the two approaches
if best_cb_config is not None:
    if cb_test_mase < nocb_test_mase:
        winner = "Callback approach"
        diff_pct = 100 * (nocb_test_mase - cb_test_mase) / nocb_test_mase
    else:
        winner = "No-callback approach"
        diff_pct = 100 * (cb_test_mase - nocb_test_mase) / cb_test_mase
    print(f"\nWinner: {winner} ({diff_pct:.1f}% better)")


