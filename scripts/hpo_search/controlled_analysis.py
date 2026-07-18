from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

MODELS = ['MLP', 'NHITS']
# MODELS = ['MLP', 'NHITS', 'PatchTST']
RESULTS_DIR = Path('./assets/results_search')
OUTPUT_DIR = Path('./assets/outputs')

# =============================================================================
# 1. Load all data
# =============================================================================

search_ind = {m: pd.read_csv(RESULTS_DIR / f'controlled_search_{m}_ind.csv') for m in MODELS}
search_transfer = {m: pd.read_csv(RESULTS_DIR / f'controlled_search_{m}_transfer.csv') for m in MODELS}

test_ind = {m: pd.read_csv(RESULTS_DIR / f'controlled_test_{m}_ind.csv') for m in MODELS}
test_transfer = {m: pd.read_csv(RESULTS_DIR / f'controlled_test_{m}_transfer.csv') for m in MODELS}


# =============================================================================
# 2. Performance tables by mode (ind vs transfer)
# =============================================================================

def build_performance_table(test_dfs: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    """Build performance table with BestCLF and BestNoCB columns for each model."""
    frames = []
    for model in models:
        df = test_dfs[model][['dataset', f'{model}-BestCLF', f'{model}-BestNoCB', 'SeasonalNaive']].copy()
        df = df.rename(columns={
            f'{model}-BestCLF': f'{model}_CLF',
            f'{model}-BestNoCB': f'{model}_NoCB',
        })
        frames.append(df.set_index('dataset'))

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


perf_ind = build_performance_table(test_ind, MODELS)
perf_transfer = build_performance_table(test_transfer, MODELS)


def add_mode_column(dfs: dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    """Combine model DataFrames and add mode column."""
    frames = []
    for model, df in dfs.items():
        df = df.copy()
        df['mode'] = mode
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


search_all = pd.concat([
    add_mode_column(search_ind, 'ind'),
    add_mode_column(search_transfer, 'transfer'),
], ignore_index=True)

# =============================================================================
# 4. Compute metrics by model, mode, and dataset
# =============================================================================

# AUC by model and mode
auc_summary = search_all.groupby(['model', 'mode', 'dataset'])['clf_search_auc'].first().reset_index()
auc_by_model = auc_summary.groupby(['model', 'mode'])['clf_search_auc'].mean().reset_index()
auc_by_model.columns = ['model', 'mode', 'value']
auc_by_model['metric'] = 'AUC'

# % rejected configs
rejection_summary = search_all.groupby(['model', 'mode', 'dataset']).agg(
    n_rejected=('clf_stopped_early', 'sum'),
    n_total=('clf_stopped_early', 'count'),
).reset_index()
rejection_summary['pct_rejected'] = rejection_summary['n_rejected'] / rejection_summary['n_total']
rejection_by_model = rejection_summary.groupby(['model', 'mode'])['pct_rejected'].mean().reset_index()
rejection_by_model.columns = ['model', 'mode', 'value']
rejection_by_model['metric'] = '% Rejected'

# % training steps used
search_all['actual_steps'] = np.where(
    search_all['clf_stopped_early'],
    search_all['clf_stop_step'],
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

combined_metrics = pd.concat([auc_by_model, rejection_by_model, steps_by_model], ignore_index=True)
combined_metrics['metric'] = pd.Categorical(
    combined_metrics['metric'],
    categories=['AUC', '% Rejected', '% Steps Used'],
    ordered=True
)
combined_metrics['mode'] = combined_metrics['mode'].map({'ind': 'Configuration In-Domain',
                                                         'transfer': 'Configuration Transfer'})
combined_metrics['mode'] = pd.Categorical(
    combined_metrics['mode'],
    categories=['Configuration In-Domain', 'Configuration Transfer'],
    ordered=True
)

MODEL_COLORS = {
    'MLP': '#2563eb',
    'NHITS': '#dc2626',
    'PatchTST': '#16a34a',
}

p_combined = (
        p9.ggplot(combined_metrics, p9.aes(x='metric', y='value', fill='model'))
        + p9.geom_bar(stat='identity', position='dodge', width=0.7)
        + p9.facet_wrap('~mode', ncol=2)
        + p9.labs(x='', y='Value', fill='Model')
        + p9.scale_fill_manual(values=MODEL_COLORS)
        + p9.theme_538(base_family='Palatino', base_size=14)
        + p9.theme(
    panel_background=p9.element_rect(fill='white'),
    plot_background=p9.element_rect(fill='white'),
    legend_background=p9.element_rect(fill='white'),
    legend_box_background=p9.element_rect(fill='white'),
    strip_background=p9.element_rect(fill='white'),
    legend_position='top',
    axis_text_x=p9.element_text(size=10),
)
)
p_combined.save(OUTPUT_DIR / 'controlled_metrics_combined.pdf', width=10, height=5, verbose=False)
