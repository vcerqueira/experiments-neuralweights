from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from src.config import N_SAMPLES, SEED
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler
from src.workflows.search_utils import (
    run_hpo_search,
    evaluate_best_configs,
    train_meta_classifier,
    train_meta_regressor,
)
from src.workflows.metadata_utils import read_all_metadata, load_dataset_splits

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

DO_TRANSFER = True
USE_REGRESSOR = True
STOPPING_THRESHOLD = 0.50
EXCEEDANCE_THRESHOLD = 0.0  # regressor: P(MASE_diff > this)
N_TRIALS = 30
CB_N_STEPS = 100
MIN_CB_N_STEPS = 400
MODEL_NAME = 'NHITS'
OUTPUT_DIR = Path('./assets/results_search')

MODE = 'transfer' if DO_TRANSFER else 'ind'
META_TYPE = 'reg' if USE_REGRESSOR else 'clf'

metadata, category_mappings = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
)

all_datasets = sorted(metadata['dataset'].unique().tolist())

config_pool = NEURAL_CONFIG_POOL[MODEL_NAME]
config_list_master = ConfigSampler.generate_samples(
    config_pool=config_pool,
    num_samples=N_SAMPLES,
    random_state=SEED,
)

if DO_TRANSFER:
    config_list_master = config_list_master[1000:]

all_search_results = []
all_test_results = []
for i, target_dataset in enumerate(all_datasets):
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
    print("=" * 70)

    _, train_full, train, valid, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
        target_dataset, get_valid=True
    )

    # Train meta-model on all datasets except target (LOO)
    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)
    if USE_REGRESSOR:
        meta_model, feature_columns = train_meta_regressor(meta_train,
                                                           model_name=MODEL_NAME,
                                                           dataset_name=target_dataset)
    else:
        meta_model, feature_columns = train_meta_classifier(meta_train,
                                                            calibrate=False,
                                                            model_name=MODEL_NAME,
                                                            dataset_name=target_dataset)

    config_list = [cfg.copy() for cfg in config_list_master]

    results_df, config_registry = run_hpo_search(
        target_dataset=target_dataset,
        metadata=metadata,
        category_mappings=category_mappings,
        config_list=config_list,
        model_name=MODEL_NAME,
        train=train,
        valid=valid,
        horizon=horizon,
        n_lags=n_lags,
        freq=freq,
        seas_len=seas_len,
        meta_model=meta_model,
        feature_columns=feature_columns,
        n_trials=N_TRIALS,
        stopping_threshold=STOPPING_THRESHOLD,
        exceedance_threshold=EXCEEDANCE_THRESHOLD,
        cb_n_steps=CB_N_STEPS,
        min_steps=MIN_CB_N_STEPS,
        verbose=True,
    )

    wc_auc, wc_ll = None, None
    if results_df['wc_exceeds_baseline'].nunique() > 1 and results_df['wc_prob_exceed'].notna().sum() > 1:
        valid_wc = results_df[results_df['wc_prob_exceed'].notna()]
        if valid_wc['wc_exceeds_baseline'].nunique() > 1:
            wc_auc = roc_auc_score(valid_wc['wc_exceeds_baseline'].astype(int), valid_wc['wc_prob_exceed'])
            wc_ll = log_loss(valid_wc['wc_exceeds_baseline'].astype(int), valid_wc['wc_prob_exceed'])
            print(f"Weightcast - AUC: {wc_auc:.3f}, LogLoss: {wc_ll:.3f}")

    print("\n", results_df[[
        'config_id',
        'valid_mase_wc', 'wc_stopped_early', 'wc_prob_exceed',
        'valid_mase_nocb',
    ]].to_string())

    test_results = evaluate_best_configs(
        results_df=results_df,
        config_registry=config_registry,
        model_name=MODEL_NAME,
        train_full=train_full,
        test=test,
        horizon=horizon,
        n_lags=n_lags,
        freq=freq,
        seas_len=seas_len,
        verbose=True,
    )

    test_results['wc_search_auc'] = wc_auc
    test_results['wc_search_ll'] = wc_ll
    test_results['n_wc_early_stopped'] = int(results_df['wc_stopped_early'].sum())
    test_results['n_trials'] = len(results_df)

    results_df['wc_search_auc'] = wc_auc
    results_df['wc_search_ll'] = wc_ll
    all_search_results.append(results_df)

    test_results['dataset'] = target_dataset
    all_test_results.append(test_results)

all_search_df = pd.concat(all_search_results, ignore_index=True)
all_test_df = pd.DataFrame(all_test_results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
search_path = OUTPUT_DIR / f"controlled_search_{MODEL_NAME}_{MODE}_{META_TYPE}.csv"
test_path = OUTPUT_DIR / f"controlled_test_{MODEL_NAME}_{MODE}_{META_TYPE}.csv"
all_search_df.to_csv(search_path, index=False)
all_test_df.to_csv(test_path, index=False)
print(f"\nSaved: {search_path}, {test_path}")
