import warnings
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.common._base_auto import OptunaOptions
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoPatchTST
from utilsforecast.losses import mase

from src.search import train_meta_classifier
from src.config_callbacks import (
    CONFIG_SAMPLERS,
    AutoConfigWithCallback,
    ConfigWithStepCounter,
    StepAccumulator,
)
from src.utils import read_all_metadata, load_dataset_splits

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

STOPPING_THRESHOLD = 0.80  # here
N_TRIALS = 50
CB_N_STEPS = 100
MIN_CB_N_STEPS = 301  # here
MODEL_NAME = 'PatchTST'
OUTPUT_DIR = Path('./assets/results_search')
PARTIAL_OUTPUT_DIR = Path('./assets/results_search_partial')

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
# all_datasets = [all_datasets[2]]

all_test_results = []
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

    step_accumulators = {
        'RS': StepAccumulator(),
        'RS+WASP': StepAccumulator(),
        'TPE+WASP': StepAccumulator(),
        'TPE': StepAccumulator(),
        'TPE+Med': StepAccumulator(),
        'TPE+SH': StepAccumulator(),
        'TPE+HB': StepAccumulator(),
        'TPE+Med+WASP': StepAccumulator(),
    }

    config_with_counter = {
        alias: ConfigWithStepCounter(config_sampler, acc)
        for alias, acc in step_accumulators.items()
        if alias not in ['RS+WASP', 'TPE+WASP', 'TPE+Med+WASP']
    }

    config_wasp = {
        alias: AutoConfigWithCallback(
            config_sampler=config_sampler,
            model_name=MODEL_NAME,
            meta_classifier=meta_classifier,
            feature_columns=clf_feature_columns,
            category_mappings=category_mappings,
            stopping_threshold=STOPPING_THRESHOLD,
            cb_n_steps=CB_N_STEPS,
            min_steps=MIN_CB_N_STEPS,
            verbose=True,
            step_accumulator=step_accumulators[alias],
        )
        for alias in ['RS+WASP', 'TPE+WASP', 'TPE+Med+WASP']
    }

    auto_base_args = {
        'h': horizon,
        'backend': "optuna",
        'num_samples': N_TRIALS,
        'refit_with_val': True,
    }

    AutoModelClass = AUTO_MODEL_CLASSES[MODEL_NAME]

    # random search
    randoms = AutoModelClass(
        config=config_with_counter['RS'],
        search_alg=optuna.samplers.RandomSampler(seed=42),
        **auto_base_args,
        alias='RS'
    )

    # random search + wasp
    randoms_wasp = AutoModelClass(
        config=config_wasp['RS+WASP'],
        search_alg=optuna.samplers.RandomSampler(seed=42),
        **auto_base_args,
        alias='RS+WASP'
    )

    # TPE+WASP
    tpe_wasp = AutoModelClass(
        config=config_wasp['TPE+WASP'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        **auto_base_args,
        alias='TPE+WASP'
    )

    # TPE
    tpe = AutoModelClass(
        config=config_with_counter['TPE'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        **auto_base_args,
        alias='TPE'
    )

    # TPE+median
    tpe_med = AutoModelClass(
        config=config_with_counter['TPE+Med'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.MedianPruner()}
        ),
        **auto_base_args,
        alias='TPE+Med'
    )

    # TPE+sh
    tpe_sh = AutoModelClass(
        config=config_with_counter['TPE+SH'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.SuccessiveHalvingPruner()}
        ),
        **auto_base_args,
        alias='TPE+SH'
    )

    # TPE+hyperband
    tpe_hb = AutoModelClass(
        config=config_with_counter['TPE+HB'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.HyperbandPruner()}
        ),
        **auto_base_args,
        alias='TPE+HB'
    )

    # TPE+WASP+MedianPruner
    tpe_wasp_med = AutoModelClass(
        config=config_wasp['TPE+Med+WASP'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.MedianPruner()}
        ),
        **auto_base_args,
        alias='TPE+Med+WASP'
    )

    models = [
        # randoms,
        # randoms_wasp,
        tpe_wasp,
        tpe,
        # tpe_med,
        # tpe_sh,
        # tpe_hb,
        # tpe_wasp_med
    ]

    nf = NeuralForecast(models=models,
                        freq=freq)
    nf.fit(df=train, val_size=horizon)

    # get cb info
    # nf.models[0].trainer

    fcst = nf.predict()
    fcst['ds'] = test['ds']

    aliases = [m.alias for m in models]

    holdout = test.merge(fcst, how='left', on=['unique_id', 'ds'])
    test_mase = mase_func(holdout, models=aliases, train_df=train)
    test_mase_value = test_mase[aliases].mean()

    step_counts = {f'{alias}_steps': acc.total_steps for alias, acc in step_accumulators.items()}

    test_results = {
        'dataset': target_dataset,
        **test_mase_value.to_dict(),
        **step_counts,
    }

    all_test_results.append(test_results)

    print(f"\nTest MASE:\n{test_mase_value}")
    print(f"\nTotal training steps per approach:")
    for alias, acc in step_accumulators.items():
        print(f"  {alias}: {acc.total_steps:,} steps ({len(acc.trial_steps)} trials)")

    PARTIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = PARTIAL_OUTPUT_DIR / f"open_test_{MODEL_NAME}_{target_dataset}.csv"
    pd.DataFrame([test_results]).to_csv(partial_path, index=False)

all_test_df = pd.DataFrame(all_test_results)

test_path = OUTPUT_DIR / f"open_test_{MODEL_NAME}.csv"
all_test_df.to_csv(test_path, index=False)
print(f"\nFinal aggregated results saved to {test_path}")
