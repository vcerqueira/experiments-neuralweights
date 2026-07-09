"""HPO search utilities with meta-model based early stopping."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase

from src.algorithms import CatBoostAUCClassifier
from src.config import TRY_MPS
from src.early_stopping import ClassifierEarlyStopCallback
from src.neural.nf_arch import ModelsConfig
from src.utils import build_meta_xy


def train_meta_classifier(
        df: pd.DataFrame,
        calibrate: bool = True,
        cal_size: float = 0.2,
        verbose: bool = True,
) -> tuple[CatBoostAUCClassifier, list[str]]:
    """Train a binary classifier to predict exceedance (MASE > MASE_baseline).
    
    Args:
        df: Training metadata (excluding target dataset).
        calibrate: Whether to calibrate probabilities.
        cal_size: Fraction of data for calibration set.
        verbose: Whether to print training info.
    
    Returns:
        Trained CatBoostAUCClassifier and list of feature column names.
    """
    data = build_meta_xy(df, task="classification", use_step_as_feature=True)

    clf = CatBoostAUCClassifier(
        calibrate=calibrate,
        calibration_method="isotonic",
        cal_size=cal_size,
    )
    clf.fit(data.X, data.y)

    if verbose:
        print(f"Meta-classifier trained with {len(data.feature_columns)} features")
        print(f"  Class distribution: {data.y.mean():.1%} exceeds baseline (y=1)")
    
    return clf, data.feature_columns


def run_hpo_search(
        target_dataset: str,
        metadata: pd.DataFrame,
        category_mappings: dict,
        config_list: list[dict],
        model_name: str,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        horizon: int,
        n_lags: int,
        freq: str,
        seas_len: int,
        *,
        n_trials: int = 10,
        stopping_threshold: float = 0.70,
        cb_n_steps: int = 100,
        min_steps: int = 30,
        verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run HPO search with meta-model early stopping for a single target dataset.
    
    Args:
        target_dataset: Name of the held-out dataset.
        metadata: Full metadata DataFrame.
        category_mappings: Category encoding mappings.
        config_list: List of config dictionaries to try.
        model_name: Neural network model name (e.g., 'MLP').
        train: Training DataFrame.
        valid: Validation DataFrame.
        horizon: Forecast horizon.
        n_lags: Number of input lags.
        freq: Time series frequency.
        seas_len: Seasonality length.
        n_trials: Number of configurations to try.
        stopping_threshold: P(exceed) threshold for early stopping.
        cb_n_steps: Callback frequency (steps).
        min_steps: Minimum steps before early stopping can trigger.
        verbose: Whether to print progress.
    
    Returns:
        Tuple of (results DataFrame, config registry dict).
    """
    mase_func = partial(mase, seasonality=seas_len)
    
    # Train meta-classifier on all datasets except target
    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)
    meta_classifier, feature_columns = train_meta_classifier(
        meta_train, calibrate=True, verbose=verbose
    )
    
    # Compute baseline MASE on validation
    sf = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq)
    sf.fit(train)
    fcst_sf = sf.predict(h=horizon)
    fcst_sf['ds'] = valid['ds']
    holdout_sn = valid.merge(fcst_sf, how='left', on=['unique_id', 'ds'])
    mase_sn = mase_func(holdout_sn, models=['SeasonalNaive'], train_df=train).mean(numeric_only=True)['SeasonalNaive']
    
    if verbose:
        print(f"Baseline validation MASE (SeasonalNaive): {mase_sn:.4f}")
    
    search_results = []
    config_registry = {}
    configs_tried = 0
    
    for config_sample in config_list:
        if configs_tried >= n_trials:
            break
        
        cfg_id = config_sample.pop('config_id')
        config_registry[cfg_id] = config_sample.copy()
        configs_tried += 1
        
        if verbose:
            print(f"\n[Config {configs_tried}/{n_trials}] {cfg_id}")
        
        early_stop_cb = ClassifierEarlyStopCallback(
            meta_classifier=meta_classifier,
            feature_columns=feature_columns,
            stopping_threshold=stopping_threshold,
            every_n_steps=cb_n_steps,
            min_steps=min_steps,
            verbose=verbose,
            config_data=config_sample,
            category_mappings=category_mappings,
        )
        
        nf_model_cb = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[early_stop_cb],
        )
        
        nf_model_nocb = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-NoCB'
        )
        
        nf = NeuralForecast(models=[nf_model_cb, nf_model_nocb], freq=freq)
        nf.fit(df=train)
        
        actual_cb = ClassifierEarlyStopCallback.get_cb(nf)
        
        fcst = nf.predict()
        fcst['ds'] = valid['ds']
        
        holdout = valid.merge(fcst, how='left', on=['unique_id', 'ds'])
        mase_model = mase_func(holdout, models=[model_name, f'{model_name}-NoCB'], train_df=train)
        
        result = {
            'config_id': cfg_id,
            'dataset': target_dataset,
            'model': model_name,
            'valid_mase_cb': float(mase_model[model_name].mean()),
            'valid_mase_nocb': float(mase_model[f'{model_name}-NoCB'].mean()),
            'valid_mase_sn': mase_sn,
            'stopped_early': actual_cb.stopped_early,
            'stop_step': actual_cb.stop_step,
            'n_predictions': len(actual_cb.predictions),
        }
        
        if actual_cb.predictions:
            result['final_prob_exceed'] = actual_cb.predictions[-1]['prob_exceed']
        else:
            result['final_prob_exceed'] = np.nan
        
        exceeds_baseline = result['valid_mase_cb'] > mase_sn
        result['exceeds_baseline'] = exceeds_baseline
        result['status'] = "WORSE" if exceeds_baseline else "BETTER"
        
        if verbose:
            early_str = f" (stopped @ step {result['stop_step']})" if result['stopped_early'] else ""
            print(f"  MASE(cb)={result['valid_mase_cb']:.4f}, MASE(nocb)={result['valid_mase_nocb']:.4f} "
                  f"vs baseline={mase_sn:.4f} -> {result['status']}{early_str}")
        
        search_results.append(result)
    
    results_df = pd.DataFrame(search_results)
    results_df['exceeds_baseline'] = (results_df['valid_mase_cb'] > results_df['valid_mase_sn']).astype(int)
    
    return results_df, config_registry


def evaluate_best_configs(
        results_df: pd.DataFrame,
        config_registry: dict[str, dict],
        model_name: str,
        train_full: pd.DataFrame,
        test: pd.DataFrame,
        horizon: int,
        n_lags: int,
        freq: str,
        seas_len: int,
        verbose: bool = True,
) -> dict[str, float]:
    """Evaluate best configs from search on test set.
    
    Args:
        results_df: Search results DataFrame.
        config_registry: Dict mapping config_id to config dict.
        model_name: Neural network model name.
        train_full: Full training data (train + valid).
        test: Test DataFrame.
        horizon: Forecast horizon.
        n_lags: Number of input lags.
        freq: Time series frequency.
        seas_len: Seasonality length.
        verbose: Whether to print results.
    
    Returns:
        Dict of test MASE scores for each model.
    """
    mase_func = partial(mase, seasonality=seas_len)
    
    # Best config WITH callback (ignore early-stopped configs)
    completed_runs = results_df[~results_df['stopped_early']]
    if len(completed_runs) > 0:
        best_cb_row = completed_runs.loc[completed_runs['valid_mase_cb'].idxmin()]
        best_cb_config_id = best_cb_row['config_id']
        best_cb_config = config_registry[best_cb_config_id]
        if verbose:
            print(f"\nBest config (with callback, completed): {best_cb_config_id}")
            print(f"  Validation MASE: {best_cb_row['valid_mase_cb']:.4f}")
    else:
        best_cb_config_id = None
        best_cb_config = None
        if verbose:
            print("\nNo completed runs with callback (all stopped early)")
    
    # Best config WITHOUT callback (all configs)
    best_nocb_row = results_df.loc[results_df['valid_mase_nocb'].idxmin()]
    best_nocb_config_id = best_nocb_row['config_id']
    best_nocb_config = config_registry[best_nocb_config_id]
    if verbose:
        print(f"\nBest config (without callback): {best_nocb_config_id}")
        print(f"  Validation MASE: {best_nocb_row['valid_mase_nocb']:.4f}")
    
    # Build final models
    final_models = []
    final_model_names = []
    
    if best_cb_config is not None:
        nf_best_cb = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=best_cb_config.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-BestCB',
        )
        final_models.append(nf_best_cb)
        final_model_names.append(f'{model_name}-BestCB')
    
    nf_best_nocb = ModelsConfig.create_model_instance(
        model_class=model_name,
        model_config=best_nocb_config.copy(),
        horizon=horizon,
        input_size=n_lags,
        try_mps=TRY_MPS,
        callbacks=[],
        alias=f'{model_name}-BestNoCB',
    )
    final_models.append(nf_best_nocb)
    final_model_names.append(f'{model_name}-BestNoCB')
    
    # Train neural models on train_full
    if verbose:
        print("\nTraining best configs on full training data...")
    
    nf_final = NeuralForecast(models=final_models, freq=freq)
    nf_final.fit(df=train_full)
    fcst_final = nf_final.predict()
    fcst_final['ds'] = test['ds']
    
    # Train Seasonal Naive on train_full
    sf_final = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq)
    sf_final.fit(train_full)
    fcst_sf_final = sf_final.predict(h=horizon)
    fcst_sf_final['ds'] = test['ds']
    
    # Merge all forecasts
    test_holdout = test.merge(fcst_final, how='left', on=['unique_id', 'ds'])
    test_holdout = test_holdout.merge(fcst_sf_final, how='left', on=['unique_id', 'ds'])
    
    # Compute test MASE
    all_test_models = final_model_names + ['SeasonalNaive']
    mase_test = mase_func(test_holdout, models=all_test_models, train_df=train_full)
    
    test_results = {}
    for m in all_test_models:
        test_results[m] = float(mase_test[m].mean())
    
    if verbose:
        print("\n" + "-" * 40)
        print("TEST SET RESULTS (MASE)")
        print("-" * 40)
        for m, score in test_results.items():
            print(f"  {m}: {score:.4f}")
        
        # Summary
        sn_mase = test_results['SeasonalNaive']
        print(f"\nBaseline (SeasonalNaive): {sn_mase:.4f}")
        
        if best_cb_config is not None:
            cb_mase = test_results[f'{model_name}-BestCB']
            cb_vs_sn = "BETTER" if cb_mase < sn_mase else "WORSE"
            cb_pct = 100 * (sn_mase - cb_mase) / sn_mase
            print(f"Best (with callback):     {cb_mase:.4f} ({cb_vs_sn}, {cb_pct:+.1f}% vs SN)")
        
        nocb_mase = test_results[f'{model_name}-BestNoCB']
        nocb_vs_sn = "BETTER" if nocb_mase < sn_mase else "WORSE"
        nocb_pct = 100 * (sn_mase - nocb_mase) / sn_mase
        print(f"Best (without callback):  {nocb_mase:.4f} ({nocb_vs_sn}, {nocb_pct:+.1f}% vs SN)")
    
    # Add metadata to results
    test_results['best_cb_config_id'] = best_cb_config_id
    test_results['best_nocb_config_id'] = best_nocb_config_id
    
    return test_results


def save_search_results(
        results_df: pd.DataFrame,
        test_results: dict,
        target_dataset: str,
        output_dir: Path,
) -> tuple[Path, Path]:
    """Save search results and test evaluation to CSV files.
    
    Args:
        results_df: Search results DataFrame.
        test_results: Test evaluation results dict.
        target_dataset: Name of target dataset.
        output_dir: Output directory.
    
    Returns:
        Tuple of (search_results_path, test_results_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save search results
    search_path = output_dir / f"search_{target_dataset}.csv"
    results_df.to_csv(search_path, index=False)
    
    # Save test results
    test_path = output_dir / f"test_{target_dataset}.csv"
    test_df = pd.DataFrame([test_results])
    test_df['dataset'] = target_dataset
    test_df.to_csv(test_path, index=False)
    
    return search_path, test_path
