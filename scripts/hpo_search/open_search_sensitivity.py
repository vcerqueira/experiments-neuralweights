import warnings
from functools import partial
from itertools import product
from pathlib import Path

import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoPatchTST
from utilsforecast.losses import mase

from src.neural.config_pool import CONFIG_SAMPLERS
from src.workflows.search_utils import train_meta_classifier, train_meta_regressor
from src.workflows.metadata_utils import read_all_metadata, load_dataset_splits
from src.weightcast.auto import WeightcastAutoConfig, StepAccumulator

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

USE_REGRESSOR = False
STOPPING_THRESHOLDS = [0.5, 0.75]
MIN_CB_N_STEPS_LIST = [0, 100, 500]
N_TRIALS_LIST = [5, 10, 20]
EXCEEDANCE_THRESHOLD = 0.0

CB_N_STEPS = 100
MODEL_NAME = 'MLP'
SEARCH_SEED = 42

META_TYPE = 'reg' if USE_REGRESSOR else 'clf'
OUTPUT_DIR = Path(f'./assets/results_sens_{META_TYPE}')

AUTO_MODEL_CLASSES = {
    'MLP': AutoMLP,
    'NHITS': AutoNHITS,
    'PatchTST': AutoPatchTST,
}

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

    train, _, _, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
        target_dataset, get_valid=True
    )

    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True).copy()
    if USE_REGRESSOR:
        meta_model, feature_columns = train_meta_regressor(meta_train)
    else:
        meta_model, feature_columns = train_meta_classifier(meta_train, calibrate=True)

    mase_func = partial(mase, seasonality=seas_len)
    config_sampler = CONFIG_SAMPLERS[MODEL_NAME](input_size=n_lags)
    AutoModelClass = AUTO_MODEL_CLASSES[MODEL_NAME]

    dataset_results = []

    for stopping_threshold, min_cb_n_steps, n_trials in product(
            STOPPING_THRESHOLDS, MIN_CB_N_STEPS_LIST, N_TRIALS_LIST
    ):
        print(
            f"\n  [threshold={stopping_threshold}, min_steps={min_cb_n_steps}, "
            f"n_trials={n_trials}]",
            end=" ",
        )

        step_accumulator = StepAccumulator()

        config_fn = WeightcastAutoConfig(
            config_sampler=config_sampler,
            model_name=MODEL_NAME,
            meta_model=meta_model,
            feature_columns=feature_columns,
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            exceedance_threshold=EXCEEDANCE_THRESHOLD,
            cb_n_steps=CB_N_STEPS,
            min_steps=min_cb_n_steps,
            verbose=False,
            step_accumulator=step_accumulator,
        )

        auto_base_args = {
            'h': horizon,
            'backend': "optuna",
            'num_samples': n_trials,
            'refit_with_val': True,
        }

        rs_wc = AutoModelClass(
            config=config_fn,
            search_alg=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            **auto_base_args,
            alias='RS+WC'
        )

        nf = NeuralForecast(models=[rs_wc], freq=freq)
        nf.fit(df=train, val_size=horizon)

        fcst = nf.predict()
        fcst['ds'] = test['ds']

        holdout = test.merge(fcst, how='left', on=['unique_id', 'ds'])
        test_mase_value = float(
            mase_func(holdout, models=['RS+WC'], train_df=train)['RS+WC'].mean()
        )

        result = {
            'dataset': target_dataset,
            'meta_type': META_TYPE,
            'stopping_threshold': stopping_threshold,
            'min_cb_n_steps': min_cb_n_steps,
            'n_trials': n_trials,
            'test_mase': test_mase_value,
            'total_steps': step_accumulator.total_steps,
            'n_trials_completed': len(step_accumulator.trial_steps),
        }

        dataset_results.append(result)
        all_results.append(result)

        print(f"MASE={test_mase_value:.4f}, steps={step_accumulator.total_steps:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_df = pd.DataFrame(dataset_results)
    final_path = OUTPUT_DIR / f"sensitivity_{MODEL_NAME}_{META_TYPE}.csv"
    partial_df.to_csv(final_path, index=False)

all_results_df = pd.DataFrame(all_results)

final_path = OUTPUT_DIR / f"sensitivity_{MODEL_NAME}_{META_TYPE}.csv"
all_results_df.to_csv(final_path, index=False)
