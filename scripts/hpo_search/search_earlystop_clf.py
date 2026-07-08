from __future__ import annotations

import warnings
from pathlib import Path
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
from src.utils import load_dataset_splits, read_all_metadata
from src.early_stopping import ClassifierEarlyStopCallback

warnings.filterwarnings('ignore')

STOPPING_THRESHOLD = 0.70
N_TRIALS = 10
CB_N_STEPS = 100

data_dir = Path('./assets/results')
model_name = 'MLP'

# metadata = read_all_metadata(data_dir, model_name, detailed=False)
# metadata = pd.read_csv('./assets/metadata.csv')
# object_cols = metadata.select_dtypes(include=['object']).columns.tolist()
# for col in object_cols:
#     metadata[col] = metadata[col].astype('category').cat.codes

metadata = pd.read_csv('./assets/metadata.csv').drop(columns=['log_norm.1','scaler_type'])
# metadata['scaler_type'] = metadata['scaler_type'].fillna("None")

object_cols = metadata.select_dtypes(include=['object']).columns.tolist()
category_mappings: dict[str, dict[str, int]] = {}
for col in object_cols:
    cat_type = metadata[col].astype('category')
    category_mappings[col] = {v: i for i, v in enumerate(cat_type.cat.categories)}
    metadata[col] = cat_type.cat.codes


target_dataset = 'monash_m1_monthly'

meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)


def train_meta_classifier(
        meta_train: pd.DataFrame,
        calibrate: bool = True,
        cal_size: float = 0.2,
) -> tuple[CatBoostAUCClassifier, list[str]]:
    """Train a binary classifier to predict exceedance (MASE > MASE_baseline).
    
    Args:
        meta_train: Training metadata (excluding target dataset).
        calibrate: Whether to calibrate probabilities.
        cal_size: Fraction of data for calibration set.
    
    Returns:
        Trained CatBoostAUCClassifier and list of feature column names.
    """
    y_binary = (meta_train['mase'] > meta_train['mase_sn']).astype(int)

    feature_cols = [
        col for col in meta_train.columns
        if col not in ['mase', 'mase_sn',
                       'model', 'config_id',
                       'dataset']
    ]
    X = meta_train[feature_cols]

    clf = CatBoostAUCClassifier(
        calibrate=calibrate,
        calibration_method="isotonic",
        cal_size=cal_size,
    )
    clf.fit(X, y_binary)

    print(f"Meta-classifier trained with {len(feature_cols)} features")
    print(f"  Class distribution: {y_binary.mean():.1%} exceeds baseline")
    return clf, feature_cols


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
print(f"Baseline MASE (SeasonalNaive): {mase_sn:.4f}")

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

    actual_cb = ClassifierEarlyStopCallback.get_cb(nf)

    fcst = nf.predict()
    fcst['ds'] = valid['ds']

    holdout = valid.merge(fcst, how='left', on=['unique_id', 'ds'])

    mase_model = mase_func(holdout, models=[model_name], train_df=train)

    result = {
        'config_id': cfg_id,
        'dataset': target_dataset,
        'model': model_name,
        'mase': float(mase_model[model_name].mean()),
        'mase_sn': mase_sn,
        'stopped_early': actual_cb.stopped_early,
        'stop_step': actual_cb.stop_step,
        'n_predictions': len(actual_cb.predictions),
    }

    if actual_cb.predictions:
        result['final_prob_exceed'] = actual_cb.predictions[-1]['prob_exceed']
    else:
        result['final_prob_exceed'] = np.nan

    exceeds_baseline = result['mase'] > mase_sn
    status = "WORSE" if exceeds_baseline else "BETTER"
    result['exceeds_baseline'] = exceeds_baseline
    result['status'] = status

    early_str = f" (stopped @ step {result['stop_step']})" if result['stopped_early'] else ""
    print(f"  MASE={result['mase']:.4f} vs baseline={mase_sn:.4f} -> {status}{early_str}")

    search_results.append(result)

results_df = pd.DataFrame(search_results)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(roc_auc_score(results_df['status'], results_df['final_prob_exceed']))
print(log_loss(results_df['status'], results_df['final_prob_exceed']))

print(results_df)
