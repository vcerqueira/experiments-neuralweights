import warnings
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.common._base_auto import OptunaOptions
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoPatchTST
from utilsforecast.losses import mase

from src.neural.config_pool import CONFIG_SAMPLERS
from src.workflows.search_utils import train_meta_classifier, train_meta_regressor
from src.workflows.extra_callbacks import (
    ConfigWithStepCounter,
    ConfigWithPruningCallback,
    StepAccumulator,
)
from src.workflows.metadata_utils import read_all_metadata, load_dataset_splits
from src.weightcast.auto import WeightcastAutoConfig

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

USE_REGRESSOR = False  # True = conformal regressor, False = binary classifier
STOPPING_THRESHOLD = 0.5
EXCEEDANCE_THRESHOLD = 0.0  # For regressor: P(MASE_diff > this)
N_TRIALS = 30
CB_N_STEPS = 100
MIN_CB_N_STEPS = 1
MODEL_NAME = 'MLP'
SEARCH_SEED = 123

META_TYPE = 'reg' if USE_REGRESSOR else 'clf'
OUTPUT_DIR = Path(f'./assets/results_search_open_{META_TYPE}')

AUTO_MODEL_CLASSES = {
    'MLP': AutoMLP,
    'NHITS': AutoNHITS,
    'PatchTST': AutoPatchTST,
}

if __name__ == "__main__":
    metadata, category_mappings = read_all_metadata(
        './assets',
        MODEL_NAME,
        processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
    )

    all_datasets = sorted(metadata['dataset'].unique().tolist())
    # all_datasets = [all_datasets[2]]
    # all_datasets = all_datasets[3:]

    all_test_results = []
    for i, target_dataset in enumerate(all_datasets):
        print("\n" + "=" * 70)
        print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
        print("=" * 70)

        _, train, _, _, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
            target_dataset, get_valid=True
        )

        # Train meta-model (LOO)
        meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True).copy()
        if USE_REGRESSOR:
            meta_model, feature_columns = train_meta_regressor(meta_train)
        else:
            meta_model, feature_columns = train_meta_classifier(meta_train, calibrate=False)

        mase_func = partial(mase, seasonality=seas_len)

        config_sampler = CONFIG_SAMPLERS[MODEL_NAME](input_size=n_lags)

        step_accumulators = {
            'RS': StepAccumulator(),
            'RS+WC': StepAccumulator(),
            'RS+Med': StepAccumulator(),
            'RS+SH': StepAccumulator(),
            'RS+HB': StepAccumulator(),
        }

        # without pruner: only step counter
        config_no_pruner = {
            'RS': ConfigWithStepCounter(config_sampler, step_accumulators['RS'])
        }

        # with Optuna pruner
        pruner_aliases = ['RS+Med', 'RS+SH', 'RS+HB']
        config_with_pruner = {
            alias: ConfigWithPruningCallback(config_sampler, step_accumulators[alias], monitor='valid_loss')
            for alias in pruner_aliases
        }

        # Weightcast callback
        config_wc = {
            'RS+WC': WeightcastAutoConfig(
                config_sampler=config_sampler,
                model_name=MODEL_NAME,
                meta_model=meta_model,
                feature_columns=feature_columns,
                category_mappings=category_mappings,
                stopping_threshold=STOPPING_THRESHOLD,
                exceedance_threshold=EXCEEDANCE_THRESHOLD,
                cb_n_steps=CB_N_STEPS,
                min_steps=MIN_CB_N_STEPS,
                verbose=True,
                step_accumulator=step_accumulators['RS+WC'],
            )
        }

        auto_base_args = {
            'h': horizon,
            'backend': "optuna",
            'num_samples': N_TRIALS,
            'refit_with_val': True,
        }

        AutoModelClass = AUTO_MODEL_CLASSES[MODEL_NAME]

        # Random search (baseline)
        randoms = AutoModelClass(
            config=config_no_pruner['RS'],
            search_alg=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            **auto_base_args,
            alias='RS'
        )

        # Random search + Weightcast
        randoms_wc = AutoModelClass(
            config=config_wc['RS+WC'],
            search_alg=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            **auto_base_args,
            alias='RS+WC'
        )

        # RS + MedianPruner
        randoms_med = AutoModelClass(
            config=config_with_pruner['RS+Med'],
            search_alg=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            optuna_options=OptunaOptions(
                create_study_kwargs={"pruner": optuna.pruners.MedianPruner()}
            ),
            **auto_base_args,
            alias='RS+Med'
        )

        # RS + SuccessiveHalvingPruner
        randoms_sh = AutoModelClass(
            config=config_with_pruner['RS+SH'],
            search_alg=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            optuna_options=OptunaOptions(
                create_study_kwargs={"pruner": optuna.pruners.SuccessiveHalvingPruner()}
            ),
            **auto_base_args,
            alias='RS+SH'
        )

        # RS + HyperbandPruner
        randoms_hb = AutoModelClass(
            config=config_with_pruner['RS+HB'],
            search_alg=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            optuna_options=OptunaOptions(
                create_study_kwargs={"pruner": optuna.pruners.HyperbandPruner()}
            ),
            **auto_base_args,
            alias='RS+HB'
        )

        models = [
            randoms,
            randoms_wc,
            randoms_med,
            randoms_sh,
            randoms_hb,
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

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        partial_path = OUTPUT_DIR / f"open_test_{MODEL_NAME}_{N_TRIALS}_{target_dataset}.csv"
        pd.DataFrame([test_results]).to_csv(partial_path, index=False)

    all_test_df = pd.DataFrame(all_test_results)
    final_path = OUTPUT_DIR / f"open_test_{MODEL_NAME}_{N_TRIALS}.csv"
    all_test_df.to_csv(final_path, index=False)
