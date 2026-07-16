from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

model_name = 'PatchTST'
results_dir = Path('./assets/results_cv')
plot_path = Path('./assets/outputs') / f'metal_clf_step_{model_name}.pdf'

METRICS = {
    'auc_mean': 'AUC',
    'll_mean': 'Log loss',
    # 'brier_mean': 'Brier score',
}

df = pd.read_csv(results_dir / f'metal_clf_step_{model_name}.csv')
df = df.assign(
    _sort_key=np.where(df['step'] == -1, np.inf, df['step']),
).sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

step_max = df.loc[df['step'] >= 0, 'step'].max()
df['step_x'] = np.where(df['step'] == -1, step_max + 100, df['step'])

plot_df = df.melt(
    id_vars=['step', 'step_x'],
    value_vars=list(METRICS.keys()),
    var_name='metric_col',
    value_name='value',
)
plot_df['metric'] = plot_df['metric_col'].map(METRICS)
plot_df['metric'] = pd.Categorical(
    plot_df['metric'],
    categories=list(METRICS.values()),
    ordered=True,
)

x_breaks = df['step_x'].tolist()
x_labels = ['Final' if step == -1 else str(int(step)) for step in df['step']]

p = (
        p9.ggplot(plot_df, p9.aes(x='step_x', y='value', color='metric', group='metric'))
        + p9.geom_line(size=1.2)
        + p9.geom_point(size=2.5)
        + p9.labs(
    x='Training step',
    y='Score',
    color=None,
)
        + p9.scale_x_continuous(breaks=x_breaks, labels=x_labels)
        # + p9.scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.2))
        + p9.scale_color_manual(values={
    'AUC': '#2563eb',
    'Log loss': '#dc2626',
    # 'Brier score': '#16a34a',
})
        + p9.theme_538(base_family='Palatino', base_size=14)
        + p9.theme(
    plot_margin=0.025,
    panel_background=p9.element_rect(fill='white'),
    plot_background=p9.element_rect(fill='white'),
    legend_box_background=p9.element_rect(fill='white'),
    legend_background=p9.element_rect(fill='white'),
    axis_text_y=p9.element_text(size=9),
    axis_text_x=p9.element_text(size=9),
    legend_title=p9.element_blank(),
    legend_position='top',
)
)

p.save(plot_path, width=6, height=5, verbose=False)
