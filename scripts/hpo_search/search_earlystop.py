import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from src.config import N_SAMPLES, SEED
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler
from src.search import run_hpo_search, evaluate_best_configs
from src.utils import read_all_metadata, load_dataset_splits

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# =============================================================================
# Configuration
# =============================================================================
STOPPING_THRESHOLD = 0.70
N_TRIALS = 30
CB_N_STEPS = 100
MIN_CB_N_STEPS = 301
MODEL_NAME = 'MLP'
OUTPUT_DIR = Path('./assets/results_search')

# =============================================================================
# Load metadata and generate config samples
# =============================================================================
print("Loading metadata...")
metadata, category_mappings = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
    # sample_n=200000,
)

all_datasets = sorted(metadata['dataset'].unique().tolist())

# all_datasets = [all_datasets[2]]

print(f"Found {len(all_datasets)} datasets: {all_datasets}")

config_pool = NEURAL_CONFIG_POOL[MODEL_NAME]
config_list_master = ConfigSampler.generate_samples(
    config_pool=config_pool,
    num_samples=N_SAMPLES,
    random_state=SEED,
)

# LOO
all_search_results = []
all_test_results = []
for i, target_dataset in enumerate(all_datasets):
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
    print("=" * 70)

    train_full, train, valid, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
        target_dataset, get_valid=True
    )

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
        n_trials=N_TRIALS,
        stopping_threshold=STOPPING_THRESHOLD,
        cb_n_steps=CB_N_STEPS,
        min_steps=MIN_CB_N_STEPS,
        verbose=True,
    )

    clf_auc, clf_ll = None, None
    if results_df['clf_exceeds_baseline'].nunique() > 1 and results_df['clf_prob_exceed'].notna().sum() > 1:
        valid_clf = results_df[results_df['clf_prob_exceed'].notna()]
        if valid_clf['clf_exceeds_baseline'].nunique() > 1:
            clf_auc = roc_auc_score(valid_clf['clf_exceeds_baseline'].astype(int), valid_clf['clf_prob_exceed'])
            clf_ll = log_loss(valid_clf['clf_exceeds_baseline'].astype(int), valid_clf['clf_prob_exceed'])
            print(f"Classifier - AUC: {clf_auc:.3f}, LogLoss: {clf_ll:.3f}")

    reg_auc, reg_ll = None, None
    if results_df['reg_exceeds_baseline'].nunique() > 1 and results_df['reg_prob_exceed'].notna().sum() > 1:
        valid_reg = results_df[results_df['reg_prob_exceed'].notna()]
        if valid_reg['reg_exceeds_baseline'].nunique() > 1:
            reg_auc = roc_auc_score(valid_reg['reg_exceeds_baseline'].astype(int), valid_reg['reg_prob_exceed'])
            reg_ll = log_loss(valid_reg['reg_exceeds_baseline'].astype(int), valid_reg['reg_prob_exceed'])
            print(f"Regressor  - AUC: {reg_auc:.3f}, LogLoss: {reg_ll:.3f}")

    print("\n", results_df[[
        'config_id',
        'valid_mase_clf', 'clf_stopped_early', 'clf_prob_exceed',
        'valid_mase_reg', 'reg_stopped_early', 'reg_prob_exceed',
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

    test_results['clf_search_auc'] = clf_auc
    test_results['clf_search_ll'] = clf_ll
    test_results['reg_search_auc'] = reg_auc
    test_results['reg_search_ll'] = reg_ll
    test_results['n_clf_early_stopped'] = int(results_df['clf_stopped_early'].sum())
    test_results['n_reg_early_stopped'] = int(results_df['reg_stopped_early'].sum())
    test_results['n_trials'] = len(results_df)

    results_df['clf_search_auc'] = clf_auc
    results_df['clf_search_ll'] = clf_ll
    results_df['reg_search_auc'] = reg_auc
    results_df['reg_search_ll'] = reg_ll
    all_search_results.append(results_df)

    test_results['dataset'] = target_dataset
    all_test_results.append(test_results)

all_search_df = pd.concat(all_search_results, ignore_index=True)
all_test_df = pd.DataFrame(all_test_results)

search_path = OUTPUT_DIR / f"search_{MODEL_NAME}.csv"
test_path = OUTPUT_DIR / f"test_{MODEL_NAME}.csv"
all_search_df.to_csv(search_path, index=False)
all_test_df.to_csv(test_path, index=False)
