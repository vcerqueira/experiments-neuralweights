"""HPO search with classifier-based early stopping (leave-one-out evaluation)."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from src.config import N_SAMPLES, SEED
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler
from src.search import (
    run_hpo_search,
    evaluate_best_configs,
    save_search_results,
)
from src.utils import read_all_metadata, load_dataset_splits

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# =============================================================================
# Configuration
# =============================================================================
STOPPING_THRESHOLD = 0.70
N_TRIALS = 25
CB_N_STEPS = 100
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
    sample_n=200000,
)

# Get all unique datasets
all_datasets = sorted(metadata['dataset'].unique().tolist())
print(f"Found {len(all_datasets)} datasets: {all_datasets}")

# Generate config samples (same for all datasets)
config_pool = NEURAL_CONFIG_POOL[MODEL_NAME]
config_list_master = ConfigSampler.generate_samples(
    config_pool=config_pool,
    num_samples=N_SAMPLES,
    random_state=SEED,
)

# =============================================================================
# Leave-one-out evaluation
# =============================================================================
all_search_results = []
all_test_results = []

for i, target_dataset in enumerate(all_datasets):
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
    print("=" * 70)

    # Load dataset splits
    try:
        train_full, train, valid, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
            target_dataset, get_valid=True
        )
    except Exception as e:
        print(f"  Skipping {target_dataset}: {e}")
        continue

    # Fresh copy of config list for each dataset
    config_list = [cfg.copy() for cfg in config_list_master]

    # Run HPO search
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
        verbose=True,
    )

    # Print search summary
    print("\n" + "-" * 40)
    print("SEARCH SUMMARY")
    print("-" * 40)

    if len(results_df) > 1 and results_df['exceeds_baseline'].nunique() > 1:
        auc = roc_auc_score(results_df['exceeds_baseline'], results_df['final_prob_exceed'])
        ll = log_loss(results_df['exceeds_baseline'], results_df['final_prob_exceed'])
        print(f"Meta-classifier AUC: {auc:.3f}, Log Loss: {ll:.3f}")
    else:
        auc, ll = None, None
        print("Not enough class variation for AUC/LogLoss")

    print(f"Configs tried: {len(results_df)}")
    print(f"Early stopped: {results_df['stopped_early'].sum()} ({100 * results_df['stopped_early'].mean():.1f}%)")
    print(
        f"Exceeded baseline: {results_df['exceeds_baseline'].sum()} ({100 * results_df['exceeds_baseline'].mean():.1f}%)")

    print("\n",
          results_df[['config_id', 'valid_mase_cb', 'valid_mase_nocb', 'stopped_early', 'final_prob_exceed', 'status']])

    # Evaluate best configs on test set
    print("\n" + "-" * 40)
    print("FINAL EVALUATION ON TEST SET")
    print("-" * 40)

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

    # Add search metrics to test results
    test_results['search_auc'] = auc
    test_results['search_ll'] = ll
    test_results['n_early_stopped'] = int(results_df['stopped_early'].sum())
    test_results['n_trials'] = len(results_df)

    # Save results for this dataset
    search_path, test_path = save_search_results(
        results_df=results_df,
        test_results=test_results,
        target_dataset=target_dataset,
        output_dir=OUTPUT_DIR,
    )
    print(f"\nSaved: {search_path}")
    print(f"Saved: {test_path}")

    # Collect for aggregation
    results_df['search_auc'] = auc
    results_df['search_ll'] = ll
    all_search_results.append(results_df)

    test_results['dataset'] = target_dataset
    all_test_results.append(test_results)

# =============================================================================
# Aggregate results across all datasets
# =============================================================================
print("\n" + "=" * 70)
print("AGGREGATE RESULTS (ALL DATASETS)")
print("=" * 70)

if all_search_results:
    all_search_df = pd.concat(all_search_results, ignore_index=True)
    all_test_df = pd.DataFrame(all_test_results)

    # Save aggregated results
    all_search_df.to_csv(OUTPUT_DIR / "search_all.csv", index=False)
    all_test_df.to_csv(OUTPUT_DIR / "test_all.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'search_all.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'test_all.csv'}")

    # Summary statistics
    print("\n" + "-" * 40)
    print("TEST SET SUMMARY (mean across datasets)")
    print("-" * 40)

    sn_col = 'SeasonalNaive'
    cb_col = f'{MODEL_NAME}-BestCB'
    nocb_col = f'{MODEL_NAME}-BestNoCB'

    if sn_col in all_test_df.columns:
        print(f"SeasonalNaive:        {all_test_df[sn_col].mean():.4f} ± {all_test_df[sn_col].std():.4f}")

    if cb_col in all_test_df.columns:
        valid_cb = all_test_df[all_test_df[cb_col].notna()]
        if len(valid_cb) > 0:
            print(
                f"Best (with callback): {valid_cb[cb_col].mean():.4f} ± {valid_cb[cb_col].std():.4f} ({len(valid_cb)} datasets)")

    if nocb_col in all_test_df.columns:
        print(f"Best (no callback):   {all_test_df[nocb_col].mean():.4f} ± {all_test_df[nocb_col].std():.4f}")

    # Early stopping stats
    print("\n" + "-" * 40)
    print("EARLY STOPPING SUMMARY")
    print("-" * 40)
    total_trials = all_search_df['config_id'].count()
    total_early_stopped = all_search_df['stopped_early'].sum()
    print(f"Total configs tried: {total_trials}")
    print(f"Total early stopped: {total_early_stopped} ({100 * total_early_stopped / total_trials:.1f}%)")

    # Per-dataset early stopping rate
    es_by_dataset = all_search_df.groupby('dataset')['stopped_early'].agg(['sum', 'count'])
    es_by_dataset['rate'] = es_by_dataset['sum'] / es_by_dataset['count']
    print("\nEarly stopping rate by dataset:")
    print(es_by_dataset.sort_values('rate', ascending=False))

    # Winner counts
    if cb_col in all_test_df.columns and nocb_col in all_test_df.columns:
        valid_comparison = all_test_df[all_test_df[cb_col].notna()]
        cb_wins = (valid_comparison[cb_col] < valid_comparison[nocb_col]).sum()
        nocb_wins = (valid_comparison[nocb_col] < valid_comparison[cb_col]).sum()
        ties = len(valid_comparison) - cb_wins - nocb_wins

        print("\n" + "-" * 40)
        print("WINNER COUNT")
        print("-" * 40)
        print(f"Callback approach wins:    {cb_wins}")
        print(f"No-callback approach wins: {nocb_wins}")
        print(f"Ties:                      {ties}")
else:
    print("No results collected.")
