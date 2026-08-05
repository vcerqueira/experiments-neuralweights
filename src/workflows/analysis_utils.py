from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_search_results(
        results_dir: Path,
        models: list[str],
        mode: str,
        meta_type: str = 'clf',
) -> dict[str, pd.DataFrame]:
    """Load search results for all models.
    
    Args:
        results_dir: Directory containing result CSVs.
        models: List of model names.
        mode: 'ind' or 'transfer'.
        meta_type: 'clf' or 'reg'.
    
    Returns:
        Dict mapping model name to search results DataFrame.
    """
    return {
        m: pd.read_csv(results_dir / f'controlled_search_{m}_{mode}_{meta_type}.csv')
        for m in models
    }


def load_test_results(
        results_dir: Path,
        models: list[str],
        mode: str,
        meta_type: str = 'clf',
) -> dict[str, pd.DataFrame]:
    """Load test results for all models.
    
    Args:
        results_dir: Directory containing result CSVs.
        models: List of model names.
        mode: 'ind' or 'transfer'.
        meta_type: 'clf' or 'reg'.
    
    Returns:
        Dict mapping model name to test results DataFrame.
    """
    return {
        m: pd.read_csv(results_dir / f'controlled_test_{m}_{mode}_{meta_type}.csv')
        for m in models
    }


def build_performance_table(
        test_dfs: dict[str, pd.DataFrame],
        models: list[str],
        wc_col_suffix: str = 'BestWC',
        nocb_col_suffix: str = 'BestNoCB',
) -> pd.DataFrame:
    """Build performance table with Weightcast and NoCB columns for each model.
    
    Args:
        test_dfs: Dict mapping model name to test results DataFrame.
        models: List of model names.
        wc_col_suffix: Column suffix for Weightcast results.
        nocb_col_suffix: Column suffix for no-callback results.
    
    Returns:
        DataFrame with performance metrics indexed by dataset.
    """
    frames = []
    for model in models:
        wc_col = f'{model}-{wc_col_suffix}'
        nocb_col = f'{model}-{nocb_col_suffix}'
        
        cols_to_get = ['dataset']
        if wc_col in test_dfs[model].columns:
            cols_to_get.append(wc_col)
        if nocb_col in test_dfs[model].columns:
            cols_to_get.append(nocb_col)
        if 'SeasonalNaive' in test_dfs[model].columns:
            cols_to_get.append('SeasonalNaive')
        
        df = test_dfs[model][cols_to_get].copy()
        
        rename_map = {}
        if wc_col in df.columns:
            rename_map[wc_col] = f'{model}_WC'
        if nocb_col in df.columns:
            rename_map[nocb_col] = f'{model}_NoCB'
        df = df.rename(columns=rename_map)
        
        frames.append(df.set_index('dataset'))

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def build_model_comparison_table(
        model: str,
        perf_ind: pd.DataFrame,
        perf_transfer: pd.DataFrame,
        method_name: str = 'Weightcast',
) -> pd.DataFrame:
    """Build a table for a single model combining in-domain and transfer performance.
    
    Args:
        model: Model name.
        perf_ind: In-domain performance DataFrame.
        perf_transfer: Transfer performance DataFrame.
        method_name: Name for the Weightcast method column.
    
    Returns:
        DataFrame with MultiIndex columns.
    """
    wc_col = f'{model}_WC'
    nocb_col = f'{model}_NoCB'

    df = pd.DataFrame({
        ('In-Domain', method_name): perf_ind[wc_col] if wc_col in perf_ind.columns else np.nan,
        ('In-Domain', 'No CB'): perf_ind[nocb_col] if nocb_col in perf_ind.columns else np.nan,
        ('Transfer', method_name): perf_transfer[wc_col] if wc_col in perf_transfer.columns else np.nan,
        ('Transfer', 'No CB'): perf_transfer[nocb_col] if nocb_col in perf_transfer.columns else np.nan,
        ('Baseline', 'S. Naive'): perf_ind['SeasonalNaive'] if 'SeasonalNaive' in perf_ind.columns else np.nan,
    })
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def to_latex_multicolumn(
        df: pd.DataFrame,
        model: str,
        meta_type: str = 'clf',
        float_format: str = "%.3f",
) -> str:
    """Convert DataFrame with MultiIndex columns to LaTeX.
    
    Args:
        df: DataFrame to convert.
        model: Model name for caption.
        meta_type: 'clf' or 'reg' for caption.
        float_format: Float formatting string.
    
    Returns:
        LaTeX string.
    """
    meta_label = 'Classifier' if meta_type == 'clf' else 'Regressor'
    return df.to_latex(
        float_format=float_format,
        multicolumn=True,
        multicolumn_format='c',
        na_rep='--',
        caption=f'MASE performance for {model} ({meta_label}): In-Domain vs Transfer.',
        label=f'tab:{model.lower()}_{meta_type}_performance',
    )


def add_mode_column(dfs: dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    """Combine model DataFrames and add mode column.
    
    Args:
        dfs: Dict mapping model name to DataFrame.
        mode: Mode label to add.
    
    Returns:
        Combined DataFrame with mode column.
    """
    frames = []
    for model, df in dfs.items():
        df = df.copy()
        df['mode'] = mode
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_search_metrics(
        search_all: pd.DataFrame,
        auc_col: str = 'wc_search_auc',
        stopped_col: str = 'wc_stopped_early',
        stop_step_col: str = 'wc_stop_step',
) -> pd.DataFrame:
    """Compute aggregated metrics from search results.
    
    Args:
        search_all: Combined search results DataFrame.
        auc_col: Column name for AUC values.
        stopped_col: Column name for early stopping flag.
        stop_step_col: Column name for stop step.
    
    Returns:
        DataFrame with AUC, % Rejected, and % Steps Used metrics.
    """
    # AUC by model and mode
    auc_summary = search_all.groupby(['model', 'mode', 'dataset'])[auc_col].first().reset_index()
    auc_by_model = auc_summary.groupby(['model', 'mode'])[auc_col].mean().reset_index()
    auc_by_model.columns = ['model', 'mode', 'value']
    auc_by_model['metric'] = 'AUC'

    # % rejected configs
    rejection_summary = search_all.groupby(['model', 'mode', 'dataset']).agg(
        n_rejected=(stopped_col, 'sum'),
        n_total=(stopped_col, 'count'),
    ).reset_index()
    rejection_summary['pct_rejected'] = rejection_summary['n_rejected'] / rejection_summary['n_total']
    rejection_by_model = rejection_summary.groupby(['model', 'mode'])['pct_rejected'].mean().reset_index()
    rejection_by_model.columns = ['model', 'mode', 'value']
    rejection_by_model['metric'] = '% Rejected'

    # % training steps used
    search_all = search_all.copy()
    search_all['actual_steps'] = np.where(
        search_all[stopped_col],
        search_all[stop_step_col],
        search_all['config_max_steps']
    )
    steps_summary = search_all.groupby(['model', 'mode', 'dataset']).agg(
        total_actual_steps=('actual_steps', 'sum'),
        total_planned_steps=('config_max_steps', 'sum'),
    ).reset_index()
    steps_summary['pct_steps_used'] = steps_summary['total_actual_steps'] / steps_summary['total_planned_steps']
    steps_by_model = steps_summary.groupby(['model', 'mode'])['pct_steps_used'].mean().reset_index()
    steps_by_model.columns = ['model', 'mode', 'value']
    steps_by_model['metric'] = '% Steps Used'

    return pd.concat([auc_by_model, rejection_by_model, steps_by_model], ignore_index=True)
