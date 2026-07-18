"""Sensitivity analysis for WASP early stopping hyperparameters.

Varies:
- STOPPING_THRESHOLD: 0.70 to 0.95 (step 0.05)
- MIN_CB_N_STEPS: 1 to 1001 (step 200)

Only uses TPE+WASP to isolate the effect of these parameters.
"""
import warnings
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoPatchTST
from utilsforecast.losses import mase

from src.search import train_meta_classifier
from src.config_callbacks import (
    CONFIG_SAMPLERS,
    AutoConfigWithCallback,
    StepAccumulator,
)
from src.utils import read_all_metadata, load_dataset_splits

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Sensitivity grid
STOPPING_THRESHOLDS = np.arange(0.70, 0.96, 0.05).round(2).tolist()
MIN_CB_N_STEPS_LIST = list(range(1, 1002, 200))

N_TRIALS = 50
CB_N_STEPS = 100
MODEL_NAME = 'MLP'
OUTPUT_DIR = Path('./assets/results_sens')
PARTIAL_OUTPUT_DIR = OUTPUT_DIR / 'partial'

AUTO_MODEL_CLASSES = {
    'MLP': AutoMLP,
    'NHITS': AutoNHITS,
    'PatchTST': AutoPatchTST,
}

print(f"Sensitivity Analysis Grid:")
print(f"  STOPPING_THRESHOLDS: {STOPPING_THRESHOLDS}")
print(f"  MIN_CB_N_STEPS_LIST: {MIN_CB_N_STEPS_LIST}")
print(f"  Total combinations per dataset: {len(STOPPING_THRESHOLDS) * len(MIN_CB_N_STEPS_LIST)}")

metadata, category_mappings = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
)

all_datasets = sorted(metadata['dataset'].unique().tolist())

all_results = []
for i, target_dataset in enumerate(all_datasets):
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
    print("=" * 70)

    train, _, _, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
        target_dataset, get_valid=True
    )

    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True).copy()
    meta_classifier, clf_feature_columns = train_meta_classifier(meta_train, calibrate=True)

    mase_func = partial(mase, seasonality=seas_len)
    config_sampler = CONFIG_SAMPLERS[MODEL_NAME](input_size=n_lags)
    AutoModelClass = AUTO_MODEL_CLASSES[MODEL_NAME]

    dataset_results = []

    for stopping_threshold, min_cb_n_steps in product(STOPPING_THRESHOLDS, MIN_CB_N_STEPS_LIST):
        print(f"\n  [threshold={stopping_threshold}, min_steps={min_cb_n_steps}]", end=" ")

        step_accumulator = StepAccumulator()

        config_fn = AutoConfigWithCallback(
            config_sampler=config_sampler,
            model_name=MODEL_NAME,
            meta_classifier=meta_classifier,
            feature_columns=clf_feature_columns,
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            cb_n_steps=CB_N_STEPS,
            min_steps=min_cb_n_steps,
            verbose=False,
            step_accumulator=step_accumulator,
        )

        auto_base_args = {
            'h': horizon,
            'backend': "optuna",
            'num_samples': N_TRIALS,
            'refit_with_val': True,
        }

        tpe_wasp = AutoModelClass(
            config=config_fn,
            search_alg=optuna.samplers.TPESampler(seed=42),
            **auto_base_args,
            alias='TPE+WASP'
        )

        nf = NeuralForecast(models=[tpe_wasp], freq=freq)
        nf.fit(df=train, val_size=horizon)

        fcst = nf.predict()
        fcst['ds'] = test['ds']

        holdout = test.merge(fcst, how='left', on=['unique_id', 'ds'])
        test_mase_value = float(mase_func(holdout, models=['TPE+WASP'], train_df=train)['TPE+WASP'].mean())

        result = {
            'dataset': target_dataset,
            'stopping_threshold': stopping_threshold,
            'min_cb_n_steps': min_cb_n_steps,
            'test_mase': test_mase_value,
            'total_steps': step_accumulator.total_steps,
            'n_trials': len(step_accumulator.trial_steps),
        }

        dataset_results.append(result)
        all_results.append(result)

        print(f"MASE={test_mase_value:.4f}, steps={step_accumulator.total_steps:,}")

    PARTIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_df = pd.DataFrame(dataset_results)
    partial_path = PARTIAL_OUTPUT_DIR / f"sensitivity_{MODEL_NAME}_{target_dataset}.csv"
    partial_df.to_csv(partial_path, index=False)
    print(f"\nDataset results saved to {partial_path}")

all_results_df = pd.DataFrame(all_results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
final_path = OUTPUT_DIR / f"sensitivity_{MODEL_NAME}.csv"
all_results_df.to_csv(final_path, index=False)
print(f"\n{'=' * 70}")
print(f"Final aggregated results saved to {final_path}")
print(f"Total experiments: {len(all_results_df)}")
