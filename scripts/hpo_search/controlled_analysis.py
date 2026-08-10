from pathlib import Path

import pandas as pd
import plotnine as p9

from src.config import DATASET_MAPPING
from src.workflows.analysis_utils import (
    load_search_results,
    load_test_results,
    build_performance_table,
    build_model_comparison_table,
    to_latex_multicolumn,
    add_mode_column,
    compute_search_metrics,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

META_TYPE = 'clf'  # 'clf' / 'reg'
MODELS = ['MLP', 'NHITS', 'PatchTST']
RESULTS_DIR = Path('./assets/results_search')
OUTPUT_DIR = Path('./assets/outputs')

META_LABEL = 'Classifier' if META_TYPE == 'clf' else 'Regressor'
METHOD_NAME = 'Weightcast'

search_ind = load_search_results(RESULTS_DIR, MODELS, 'ind', META_TYPE)
search_transfer = load_search_results(RESULTS_DIR, MODELS, 'transfer', META_TYPE)
test_ind = load_test_results(RESULTS_DIR, MODELS, 'ind', META_TYPE)
test_transfer = load_test_results(RESULTS_DIR, MODELS, 'transfer', META_TYPE)

perf_ind = build_performance_table(test_ind, MODELS)
perf_transfer = build_performance_table(test_transfer, MODELS)

perf_ind.rename(index=DATASET_MAPPING, inplace=True)
perf_transfer.rename(index=DATASET_MAPPING, inplace=True)

for model in MODELS:
    table = build_model_comparison_table(model, perf_ind, perf_transfer, method_name=METHOD_NAME)
    latex_str = to_latex_multicolumn(table, model, meta_type=META_TYPE)
    print(f"\n{model}:")
    print(latex_str)

search_all = pd.concat([
    add_mode_column(search_ind, 'ind'),
    add_mode_column(search_transfer, 'transfer'),
], ignore_index=True)

combined_metrics = compute_search_metrics(
    search_all,
    auc_col='wc_search_auc',
    stopped_col='wc_stopped_early',
    stop_step_col='wc_stop_step',
)

combined_metrics['metric'] = pd.Categorical(
    combined_metrics['metric'],
    categories=['AUC', '% Rejected', '% Steps Used'],
    ordered=True
)
combined_metrics['mode'] = combined_metrics['mode'].map({
    'ind': 'Configuration In-Domain',
    'transfer': 'Configuration Transfer'
})
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
        + p9.labs(x='', y='Value', fill='Model', title='')
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f'controlled_metrics_combined_{META_TYPE}.pdf'
p_combined.save(output_path, width=10, height=5, verbose=False)
