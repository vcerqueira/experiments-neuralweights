"""HPO search utilities with Weightcast meta-model based early stopping."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase

from src.config import TRY_MPS
from src.neural.nf_arch import ModelsConfig
from src.utils import build_meta_xy
from src.weightcast.callbacks import WeightcastClassifier, WeightcastRegressor
from src.weightcast.learner_classifier import CatBoostAUCClassifier
from src.weightcast.learner_regressor import CatBoostRegressionModel


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
        meta_model: Union[CatBoostAUCClassifier, CatBoostRegressionModel, None] = None,
        feature_columns: list[str] = None,
        n_trials: int = 10,
        stopping_threshold: float = 0.70,
        exceedance_threshold: float = 0.0,
        cb_n_steps: int = 100,
        min_steps: int = 30,
        verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run HPO search with Weightcast early stopping callbacks.

    Args:
        target_dataset: Name of the target dataset.
        metadata: Full metadata DataFrame.
        category_mappings: Category mappings for encoding.
        config_list: List of configs to try (each must have 'config_id').
        model_name: Neural network model name.
        train: Training DataFrame.
        valid: Validation DataFrame.
        horizon: Forecast horizon.
        n_lags: Number of input lags.
        freq: Time series frequency.
        seas_len: Seasonality length.
        meta_model: Pre-trained meta-model. If None, trains a new classifier.
        feature_columns: Feature columns for the meta-model.
        n_trials: Maximum number of configs to try.
        stopping_threshold: Probability threshold for early stopping.
        exceedance_threshold: For regressor - value threshold for P(Y > threshold).
        cb_n_steps: Check callback every N steps.
        min_steps: Minimum steps before callback activates.
        verbose: Whether to print progress.

    Returns:
        Tuple of (results DataFrame, config registry dict).
    """
    mase_func = partial(mase, seasonality=seas_len)

    # Train meta-model if not provided
    if meta_model is None:
        meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)
        meta_model, feature_columns = train_meta_classifier(meta_train, calibrate=True)

    is_classifier = isinstance(meta_model, CatBoostAUCClassifier)

    # Baseline MASE on validation
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

        # Create Weightcast callback (classifier or regressor)
        if is_classifier:
            weightcast_cb = WeightcastClassifier(
                meta_classifier=meta_model,
                feature_columns=feature_columns,
                config_data=config_sample,
                category_mappings=category_mappings,
                stopping_threshold=stopping_threshold,
                every_n_steps=cb_n_steps,
                min_steps=min_steps,
                verbose=verbose,
            )
        else:
            weightcast_cb = WeightcastRegressor(
                meta_model=meta_model,
                feature_columns=feature_columns,
                config_data=config_sample,
                category_mappings=category_mappings,
                stopping_threshold=stopping_threshold,
                exceedance_threshold=exceedance_threshold,
                every_n_steps=cb_n_steps,
                min_steps=min_steps,
                verbose=verbose,
            )

        # Model with Weightcast callback
        nf_model_wc = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[weightcast_cb],
            alias=f'{model_name}-WC',
        )

        # Model without callback (baseline)
        nf_model_nocb = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=config_sample.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-NoCB',
        )

        nf = NeuralForecast(models=[nf_model_wc, nf_model_nocb], freq=freq)
        nf.fit(df=train)

        # Retrieve actual callback after training
        CallbackClass = WeightcastClassifier if is_classifier else WeightcastRegressor
        actual_cb = CallbackClass.get_cb(nf)

        fcst = nf.predict()
        fcst['ds'] = valid['ds']

        holdout = valid.merge(fcst, how='left', on=['unique_id', 'ds'])
        model_aliases = [f'{model_name}-WC', f'{model_name}-NoCB']
        mase_model = mase_func(holdout, models=model_aliases, train_df=train)

        result = {
            'config_id': cfg_id,
            'config_max_steps': config_sample['max_steps'],
            'dataset': target_dataset,
            'model': model_name,
            'valid_mase_wc': float(mase_model[f'{model_name}-WC'].mean()),
            'valid_mase_nocb': float(mase_model[f'{model_name}-NoCB'].mean()),
            'valid_mase_sn': mase_sn,
            'wc_stopped_early': actual_cb.stopped_early,
            'wc_stop_step': actual_cb.stop_step,
            'wc_n_predictions': len(actual_cb.predictions),
            'wc_prob_exceed': actual_cb.predictions[-1]['prob_exceed'] if actual_cb.predictions else np.nan,
        }

        result['wc_exceeds_baseline'] = result['valid_mase_wc'] > mase_sn
        result['nocb_exceeds_baseline'] = result['valid_mase_nocb'] > mase_sn

        search_results.append(result)

    return pd.DataFrame(search_results), config_registry


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

    Selects best config from each approach (Weightcast, no callback)
    based on validation MASE, trains on full training data, and evaluates on test.

    For Weightcast, only considers runs that completed (not stopped early).

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

    # Best config from Weightcast approach (ignore early-stopped)
    wc_completed = results_df[~results_df['wc_stopped_early']]
    if len(wc_completed) > 0:
        best_wc_row = wc_completed.loc[wc_completed['valid_mase_wc'].idxmin()]
        best_wc_config_id = best_wc_row['config_id']
        best_wc_config = config_registry[best_wc_config_id]
        best_configs['wc'] = best_wc_config_id

        if verbose:
            print(f"\nBest config (Weightcast, completed): {best_wc_config_id}")
            print(f"  Validation MASE: {best_wc_row['valid_mase_wc']:.4f}")

        nf_best_wc = ModelsConfig.create_model_instance(
            model_class=model_name,
            model_config=best_wc_config.copy(),
            horizon=horizon,
            input_size=n_lags,
            try_mps=TRY_MPS,
            callbacks=[],
            alias=f'{model_name}-BestWC',
        )
        final_models.append(nf_best_wc)
        final_model_names.append(f'{model_name}-BestWC')
    else:
        best_configs['wc'] = None
        if verbose:
            print("\nNo completed runs with Weightcast callback (all stopped early)")

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

    test_results['best_wc_config_id'] = best_configs.get('wc')
    test_results['best_nocb_config_id'] = best_configs.get('nocb')

    return test_results

#
# def save_search_results(
#         results_df: pd.DataFrame,
#         test_results: dict,
#         target_dataset: str,
#         output_dir: Path,
# ) -> tuple[Path, Path]:
#     """Save search results and test evaluation to CSV files.
#
#     Args:
#         results_df: Search results DataFrame.
#         test_results: Test evaluation results dict.
#         target_dataset: Name of target dataset.
#         output_dir: Output directory.
#
#     Returns:
#         Tuple of (search_results_path, test_results_path).
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     search_path = output_dir / f"search_{target_dataset}.csv"
#     results_df.to_csv(search_path, index=False)
#
#     test_path = output_dir / f"test_{target_dataset}.csv"
#     test_df = pd.DataFrame([test_results])
#     test_df['dataset'] = target_dataset
#     test_df.to_csv(test_path, index=False)
#
#     return search_path, test_path
