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
    ConfigWithPruningCallback,
    StepAccumulator,
)
from src.utils import read_all_metadata, load_dataset_splits

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

STOPPING_THRESHOLD = 0.5  # here
N_TRIALS = 30
CB_N_STEPS = 100
MIN_CB_N_STEPS = 101  # here
MODEL_NAME = 'NHITS'
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
all_datasets = [all_datasets[6]]
# all_datasets = all_datasets[2:]

all_test_results = []
for i, target_dataset in enumerate(all_datasets):
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
    print("=" * 70)

    train, _, _, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
        target_dataset, get_valid=True
    )

    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True).copy()
    meta_classifier, clf_feature_columns = train_meta_classifier(meta_train,
                                                                 calibrate=False,
                                                                 cal_size=0.4)

    mase_func = partial(mase, seasonality=seas_len)

    config_sampler = CONFIG_SAMPLERS[MODEL_NAME](input_size=n_lags)

    step_accumulators = {
        'RS': StepAccumulator(),
        'RS+WASP': StepAccumulator(),
        'RS+Med': StepAccumulator(),
        'RS+SH': StepAccumulator(),
        'RS+HB': StepAccumulator(),
        'TPE': StepAccumulator(),
        'TPE+WASP': StepAccumulator(),
        'TPE+Med': StepAccumulator(),
        'TPE+SH': StepAccumulator(),
        'TPE+HB': StepAccumulator(),
    }

    # Variants without pruner: only step counter
    no_pruner_aliases = ['RS', 'TPE']
    config_no_pruner = {
        alias: ConfigWithStepCounter(config_sampler, step_accumulators[alias])
        for alias in no_pruner_aliases
    }

    # Variants with Optuna pruner: need PyTorchLightningPruningCallback for pruner to work
    pruner_aliases = ['RS+Med', 'RS+SH', 'RS+HB', 'TPE+Med', 'TPE+SH', 'TPE+HB']
    config_with_pruner = {
        alias: ConfigWithPruningCallback(config_sampler, step_accumulators[alias], monitor='valid_loss')
        for alias in pruner_aliases
    }

    # Variants with WASP callback
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
        for alias in ['RS+WASP', 'TPE+WASP']
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
        config=config_no_pruner['RS'],
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

    # rs+median (with PyTorchLightningPruningCallback for pruner to work)
    randoms_med = AutoModelClass(
        config=config_with_pruner['RS+Med'],
        search_alg=optuna.samplers.RandomSampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.MedianPruner()}
        ),
        **auto_base_args,
        alias='RS+Med'
    )

    # rs+sh
    randoms_sh = AutoModelClass(
        config=config_with_pruner['RS+SH'],
        search_alg=optuna.samplers.RandomSampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.SuccessiveHalvingPruner()}
        ),
        **auto_base_args,
        alias='RS+SH'
    )

    # rs+hyperband
    randoms_hb = AutoModelClass(
        config=config_with_pruner['RS+HB'],
        search_alg=optuna.samplers.RandomSampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.HyperbandPruner()}
        ),
        **auto_base_args,
        alias='RS+HB'
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
        config=config_no_pruner['TPE'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        **auto_base_args,
        alias='TPE'
    )

    # TPE+median (with PyTorchLightningPruningCallback for pruner to work)
    tpe_med = AutoModelClass(
        config=config_with_pruner['TPE+Med'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.MedianPruner()}
        ),
        **auto_base_args,
        alias='TPE+Med'
    )

    # TPE+sh
    tpe_sh = AutoModelClass(
        config=config_with_pruner['TPE+SH'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.SuccessiveHalvingPruner()}
        ),
        **auto_base_args,
        alias='TPE+SH'
    )

    # TPE+hyperband
    tpe_hb = AutoModelClass(
        config=config_with_pruner['TPE+HB'],
        search_alg=optuna.samplers.TPESampler(seed=42),
        optuna_options=OptunaOptions(
            create_study_kwargs={"pruner": optuna.pruners.HyperbandPruner()}
        ),
        **auto_base_args,
        alias='TPE+HB'
    )


    models = [
        randoms,
        randoms_wasp,
        randoms_med,
        randoms_sh,
        randoms_hb,
        # tpe,
        # tpe_wasp,
        # tpe_med,
        # tpe_sh,
        # tpe_hb,
    ]

    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train, val_size=horizon)

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
