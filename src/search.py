"""HPO search utilities with meta-model based early stopping."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase

from src.algorithms import CatBoostAUCClassifier, CatBoostRegressionModel
from src.config import TRY_MPS
from src.early_stopping import ClassifierEarlyStopCallback, MetaModelEarlyStopCallback
from src.neural.nf_arch import ModelsConfig
from src.utils import build_meta_xy


def train_meta_classifier(
        df: pd.DataFrame,
        calibrate: bool = True,
        cal_size: float = 0.2,
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
        calibration_method="platt",
        cal_size=cal_size,
    )
    clf.fit(data.X, data.y)

    return clf, data.feature_columns


def train_meta_regressor(
        df: pd.DataFrame,
        conformal_cal_size: float = 0.025,
        y_clip: tuple[float, float] = (-2.5, 2.5),
) -> tuple[CatBoostRegressionModel, list[str]]:
    """Train a regression model with conformal prediction for exceedance probability.
    
    Args:
        df: Training metadata (excluding target dataset).
        conformal_cal_size: Fraction of data for conformal calibration.
        y_clip: Min/max clipping for target variable.
        verbose: Whether to print training info.
    
    Returns:
        Trained CatBoostRegressionModel and list of feature column names.
    """
    data = build_meta_xy(
        df, 
        task="regression", 
        use_step_as_feature=True,
        performance_diff=True,
        y_clip=y_clip,
    )

    reg = CatBoostRegressionModel(
        conformal=True,
        conformal_cal_size=conformal_cal_size,
        calibration_method="isotonic",
    )
    reg.fit(data.X, data.y, calibrate_threshold=None)

    return reg, data.feature_columns


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
    """Run HPO search with both classifier and regressor early stopping callbacks.
    
    Trains 3 models per config:
    - Model with classifier-based early stopping callback
    - Model with regressor-based early stopping callback  
    - Model without any callback (baseline)
    
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
    
    # Train meta-models on all datasets except target
    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)
    
    meta_classifier, clf_feature_columns = train_meta_classifier(
        meta_train, calibrate=True
    )
    meta_regressor, reg_feature_columns = train_meta_regressor(
        meta_train
    )
    
    # baseline MASE on validation
    sf = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq)
    sf.fit(train)
    fcst_sf = sf.predict(h=horizon)
    fcst_sf['ds'] = valid['ds']
    holdout_sn = valid.merge(fcst_sf, how='left', on=['unique_id', 'ds'])
    mase_sn = mase_func(holdout_sn, models=['SeasonalNaive'], train_df=train).mean(numeric_only=True)['SeasonalNaive']
    
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
        
        # Classifier-based early stopping callback
        clf_cb = ClassifierEarlyStopCallback(
            meta_classifier=meta_classifier,
            feature_columns=clf_feature_columns,
            stopping_threshold=stopping_threshold,
            every_n_steps=cb_n_steps,
            min_steps=min_steps,
            verbose=verbose,
            config_data=config_sample,
            category_mappings=category_mappings,
        )
        
        # Regressor-based early stopping callback
        reg_cb = MetaModelEarlyStopCallback(
            meta_model=meta_regressor,
            feature_columns=reg_feature_columns,
            stopping_threshold=stopping_threshold,
            exceedance_threshold=0.0,
            every_n_steps=cb_n_steps,
            min_steps=min_steps,
            verbose=verbose,
            config_data=config_sample,
            category_mappings=category_mappings,
        )
        
        # Model with classifier callback
        nf_model_clf = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[clf_cb],
            alias=f'{model_name}-CLF',
        )
        
        # Model with regressor callback
        nf_model_reg = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[reg_cb],
            alias=f'{model_name}-REG',
        )
        
        # Model without callback
        nf_model_nocb = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-NoCB',
        )
        
        nf = NeuralForecast(models=[nf_model_clf, nf_model_reg, nf_model_nocb], freq=freq)
        nf.fit(df=train)
        
        # Retrieve actual callbacks after training
        actual_clf_cb = ClassifierEarlyStopCallback.get_cb(nf)
        actual_reg_cb = MetaModelEarlyStopCallback.get_cb(nf)
        
        fcst = nf.predict()
        fcst['ds'] = valid['ds']
        
        holdout = valid.merge(fcst, how='left', on=['unique_id', 'ds'])
        model_aliases = [f'{model_name}-CLF', f'{model_name}-REG', f'{model_name}-NoCB']
        mase_model = mase_func(holdout, models=model_aliases, train_df=train)
        
        result = {
            'config_id': cfg_id,
            'dataset': target_dataset,
            'model': model_name,
            'valid_mase_clf': float(mase_model[f'{model_name}-CLF'].mean()),
            'valid_mase_reg': float(mase_model[f'{model_name}-REG'].mean()),
            'valid_mase_nocb': float(mase_model[f'{model_name}-NoCB'].mean()),
            'valid_mase_sn': mase_sn,
            # Classifier callback stats
            'clf_stopped_early': actual_clf_cb.stopped_early,
            'clf_stop_step': actual_clf_cb.stop_step,
            'clf_n_predictions': len(actual_clf_cb.predictions),
            'clf_prob_exceed': actual_clf_cb.predictions[-1]['prob_exceed'] if actual_clf_cb.predictions else np.nan,
            # Regressor callback stats
            'reg_stopped_early': actual_reg_cb.stopped_early,
            'reg_stop_step': actual_reg_cb.stop_step,
            'reg_n_predictions': len(actual_reg_cb.predictions),
            'reg_prob_exceed': actual_reg_cb.predictions[-1]['prob_exceed'] if actual_reg_cb.predictions else np.nan,
        }
        
        # Determine status for each approach
        result['clf_exceeds_baseline'] = result['valid_mase_clf'] > mase_sn
        result['reg_exceeds_baseline'] = result['valid_mase_reg'] > mase_sn
        result['nocb_exceeds_baseline'] = result['valid_mase_nocb'] > mase_sn
        
        if verbose:
            clf_str = f" (stopped@{result['clf_stop_step']})" if result['clf_stopped_early'] else ""
            reg_str = f" (stopped@{result['reg_stop_step']})" if result['reg_stopped_early'] else ""
            print(f"  CLF={result['valid_mase_clf']:.4f}{clf_str}, "
                  f"REG={result['valid_mase_reg']:.4f}{reg_str}, "
                  f"NoCB={result['valid_mase_nocb']:.4f} "
                  f"(SN={mase_sn:.4f})")
        
        search_results.append(result)
    
    results_df = pd.DataFrame(search_results)
    
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
    
    Selects best config from each approach (classifier, regressor, no callback)
    based on validation MASE, trains on full training data, and evaluates on test.
    
    For classifier and regressor approaches, only considers runs that completed
    (not stopped early).
    
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
    
    final_models = []
    final_model_names = []
    best_configs = {}
    
    # Best config from CLASSIFIER approach (ignore early-stopped)
    clf_completed = results_df[~results_df['clf_stopped_early']]
    if len(clf_completed) > 0:
        best_clf_row = clf_completed.loc[clf_completed['valid_mase_clf'].idxmin()]
        best_clf_config_id = best_clf_row['config_id']
        best_clf_config = config_registry[best_clf_config_id]
        best_configs['clf'] = best_clf_config_id
        
        if verbose:
            print(f"\nBest config (classifier, completed): {best_clf_config_id}")
            print(f"  Validation MASE: {best_clf_row['valid_mase_clf']:.4f}")
        
        nf_best_clf = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=best_clf_config.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-BestCLF',
        )
        final_models.append(nf_best_clf)
        final_model_names.append(f'{model_name}-BestCLF')
    else:
        best_configs['clf'] = None
        if verbose:
            print("\nNo completed runs with classifier callback (all stopped early)")
    
    # Best config from REGRESSOR approach (ignore early-stopped)
    reg_completed = results_df[~results_df['reg_stopped_early']]
    if len(reg_completed) > 0:
        best_reg_row = reg_completed.loc[reg_completed['valid_mase_reg'].idxmin()]
        best_reg_config_id = best_reg_row['config_id']
        best_reg_config = config_registry[best_reg_config_id]
        best_configs['reg'] = best_reg_config_id
        
        if verbose:
            print(f"\nBest config (regressor, completed): {best_reg_config_id}")
            print(f"  Validation MASE: {best_reg_row['valid_mase_reg']:.4f}")
        
        nf_best_reg = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=best_reg_config.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-BestREG',
        )
        final_models.append(nf_best_reg)
        final_model_names.append(f'{model_name}-BestREG')
    else:
        best_configs['reg'] = None
        if verbose:
            print("\nNo completed runs with regressor callback (all stopped early)")
    
    # Best config WITHOUT callback (all configs)
    best_nocb_row = results_df.loc[results_df['valid_mase_nocb'].idxmin()]
    best_nocb_config_id = best_nocb_row['config_id']
    best_nocb_config = config_registry[best_nocb_config_id]
    best_configs['nocb'] = best_nocb_config_id
    
    if verbose:
        print(f"\nBest config (no callback): {best_nocb_config_id}")
        print(f"  Validation MASE: {best_nocb_row['valid_mase_nocb']:.4f}")
    
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
        
        # Summary comparison
        sn_mase = test_results['SeasonalNaive']
        print(f"\nBaseline (SeasonalNaive): {sn_mase:.4f}")
        
        for approach, col in [('BestCLF', f'{model_name}-BestCLF'), 
                               ('BestREG', f'{model_name}-BestREG'),
                               ('BestNoCB', f'{model_name}-BestNoCB')]:
            if col in test_results:
                approach_mase = test_results[col]
                vs_sn = "BETTER" if approach_mase < sn_mase else "WORSE"
                pct = 100 * (sn_mase - approach_mase) / sn_mase
                print(f"{approach:12s}: {approach_mase:.4f} ({vs_sn}, {pct:+.1f}% vs SN)")
    
    test_results['best_clf_config_id'] = best_configs.get('clf')
    test_results['best_reg_config_id'] = best_configs.get('reg')
    test_results['best_nocb_config_id'] = best_configs.get('nocb')
    
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
