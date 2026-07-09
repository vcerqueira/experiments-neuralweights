"""HPO search with classifier and regressor early stopping (leave-one-out evaluation).

Compares three approaches:
1. Classifier-based early stopping callback
2. Regressor-based early stopping callback (with conformal prediction)
3. No callback (baseline)
"""
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
N_TRIALS = 2
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
    sample_n=20000,
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

    # Run HPO search with both callbacks
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

    # Classifier callback stats
    clf_auc, clf_ll = None, None
    if results_df['clf_exceeds_baseline'].nunique() > 1 and results_df['clf_prob_exceed'].notna().sum() > 1:
        valid_clf = results_df[results_df['clf_prob_exceed'].notna()]
        if valid_clf['clf_exceeds_baseline'].nunique() > 1:
            clf_auc = roc_auc_score(valid_clf['clf_exceeds_baseline'].astype(int), valid_clf['clf_prob_exceed'])
            clf_ll = log_loss(valid_clf['clf_exceeds_baseline'].astype(int), valid_clf['clf_prob_exceed'])
            print(f"Classifier - AUC: {clf_auc:.3f}, LogLoss: {clf_ll:.3f}")
    
    # Regressor callback stats
    reg_auc, reg_ll = None, None
    if results_df['reg_exceeds_baseline'].nunique() > 1 and results_df['reg_prob_exceed'].notna().sum() > 1:
        valid_reg = results_df[results_df['reg_prob_exceed'].notna()]
        if valid_reg['reg_exceeds_baseline'].nunique() > 1:
            reg_auc = roc_auc_score(valid_reg['reg_exceeds_baseline'].astype(int), valid_reg['reg_prob_exceed'])
            reg_ll = log_loss(valid_reg['reg_exceeds_baseline'].astype(int), valid_reg['reg_prob_exceed'])
            print(f"Regressor  - AUC: {reg_auc:.3f}, LogLoss: {reg_ll:.3f}")

    print(f"\nConfigs tried: {len(results_df)}")
    print(f"Classifier early stopped: {results_df['clf_stopped_early'].sum()} ({100 * results_df['clf_stopped_early'].mean():.1f}%)")
    print(f"Regressor early stopped:  {results_df['reg_stopped_early'].sum()} ({100 * results_df['reg_stopped_early'].mean():.1f}%)")

    # Show results table
    print("\n", results_df[[
        'config_id', 
        'valid_mase_clf', 'clf_stopped_early', 'clf_prob_exceed',
        'valid_mase_reg', 'reg_stopped_early', 'reg_prob_exceed',
        'valid_mase_nocb',
    ]].to_string())

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
    test_results['clf_search_auc'] = clf_auc
    test_results['clf_search_ll'] = clf_ll
    test_results['reg_search_auc'] = reg_auc
    test_results['reg_search_ll'] = reg_ll
    test_results['n_clf_early_stopped'] = int(results_df['clf_stopped_early'].sum())
    test_results['n_reg_early_stopped'] = int(results_df['reg_stopped_early'].sum())
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
    results_df['clf_search_auc'] = clf_auc
    results_df['clf_search_ll'] = clf_ll
    results_df['reg_search_auc'] = reg_auc
    results_df['reg_search_ll'] = reg_ll
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
    print("TEST SET SUMMARY (mean ± std across datasets)")
    print("-" * 40)

    sn_col = 'SeasonalNaive'
    clf_col = f'{MODEL_NAME}-BestCLF'
    reg_col = f'{MODEL_NAME}-BestREG'
    nocb_col = f'{MODEL_NAME}-BestNoCB'

    if sn_col in all_test_df.columns:
        print(f"SeasonalNaive:    {all_test_df[sn_col].mean():.4f} ± {all_test_df[sn_col].std():.4f}")

    if clf_col in all_test_df.columns:
        valid_clf = all_test_df[all_test_df[clf_col].notna()]
        if len(valid_clf) > 0:
            print(f"Best (CLF):       {valid_clf[clf_col].mean():.4f} ± {valid_clf[clf_col].std():.4f} ({len(valid_clf)} datasets)")

    if reg_col in all_test_df.columns:
        valid_reg = all_test_df[all_test_df[reg_col].notna()]
        if len(valid_reg) > 0:
            print(f"Best (REG):       {valid_reg[reg_col].mean():.4f} ± {valid_reg[reg_col].std():.4f} ({len(valid_reg)} datasets)")

    if nocb_col in all_test_df.columns:
        print(f"Best (NoCB):      {all_test_df[nocb_col].mean():.4f} ± {all_test_df[nocb_col].std():.4f}")

    # Early stopping stats
    print("\n" + "-" * 40)
    print("EARLY STOPPING SUMMARY")
    print("-" * 40)
    total_trials = len(all_search_df)
    clf_early_stopped = all_search_df['clf_stopped_early'].sum()
    reg_early_stopped = all_search_df['reg_stopped_early'].sum()
    print(f"Total configs tried: {total_trials}")
    print(f"Classifier early stopped: {clf_early_stopped} ({100 * clf_early_stopped / total_trials:.1f}%)")
    print(f"Regressor early stopped:  {reg_early_stopped} ({100 * reg_early_stopped / total_trials:.1f}%)")

    # Per-dataset early stopping rate
    print("\nClassifier early stopping rate by dataset:")
    es_clf = all_search_df.groupby('dataset')['clf_stopped_early'].agg(['sum', 'count'])
    es_clf['rate'] = es_clf['sum'] / es_clf['count']
    print(es_clf.sort_values('rate', ascending=False))

    print("\nRegressor early stopping rate by dataset:")
    es_reg = all_search_df.groupby('dataset')['reg_stopped_early'].agg(['sum', 'count'])
    es_reg['rate'] = es_reg['sum'] / es_reg['count']
    print(es_reg.sort_values('rate', ascending=False))

    # Winner counts (pairwise comparisons)
    print("\n" + "-" * 40)
    print("WINNER COUNT (lower MASE is better)")
    print("-" * 40)

    def count_wins(col1, col2, df):
        """Count wins for col1 vs col2."""
        valid = df[df[col1].notna() & df[col2].notna()]
        wins = (valid[col1] < valid[col2]).sum()
        losses = (valid[col1] > valid[col2]).sum()
        ties = len(valid) - wins - losses
        return wins, losses, ties

    comparisons = [
        (clf_col, nocb_col, "CLF vs NoCB"),
        (reg_col, nocb_col, "REG vs NoCB"),
        (clf_col, reg_col, "CLF vs REG"),
        (clf_col, sn_col, "CLF vs SN"),
        (reg_col, sn_col, "REG vs SN"),
        (nocb_col, sn_col, "NoCB vs SN"),
    ]

    for col1, col2, label in comparisons:
        if col1 in all_test_df.columns and col2 in all_test_df.columns:
            wins, losses, ties = count_wins(col1, col2, all_test_df)
            print(f"{label:15s}: {wins:2d} wins, {losses:2d} losses, {ties:2d} ties")

else:
    print("No results collected.")
