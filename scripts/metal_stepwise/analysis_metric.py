from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

MODELS = ['MLP', 'NHITS', 'PatchTST']
results_dir = Path('./assets/results_cv')
output_dir = Path('./assets/outputs')

METRICS = {
    'auc_mean': 'AUC',
    'll_mean': 'Log loss',
}

MODEL_COLORS = {
    'MLP': '#2563eb',
    'NHITS': '#dc2626',
    'PatchTST': '#16a34a',
}

frames = []
for model in MODELS:
    df = pd.read_csv(results_dir / f'metal_clf_step_{model}.csv')
    df = df.assign(
        _sort_key=np.where(df['step'] == -1, np.inf, df['step']),
    ).sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

    step_max = df.loc[df['step'] >= 0, 'step'].max()
    df['step_x'] = np.where(df['step'] == -1, step_max + 100, df['step'])
    df['model'] = model
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
combined['model'] = pd.Categorical(combined['model'], categories=MODELS, ordered=True)

x_ref = combined.loc[combined['model'] == MODELS[0]].sort_values('step_x')
x_breaks = x_ref['step_x'].tolist()
x_labels = ['Final' if step == -1 else str(int(step)) for step in x_ref['step']]

theme = (
        p9.theme_538(base_family='Palatino', base_size=14)
        + p9.theme(
    plot_margin=0.025,
    panel_background=p9.element_rect(fill='white'),
    plot_background=p9.element_rect(fill='white'),
    legend_box_background=p9.element_rect(fill='white'),
    legend_background=p9.element_rect(fill='white'),
    axis_text_y=p9.element_text(size=10),
    axis_text_x=p9.element_text(size=10),
    legend_title=p9.element_blank(),
    legend_position='top',
)
)

for metric_col, metric_label in METRICS.items():
    plot_df = combined[['step', 'step_x', 'model', metric_col]].rename(
        columns={metric_col: 'value'}
    )

    plot_path = output_dir / f'metal_clf_step_{metric_col}.pdf'
    p = (
            p9.ggplot(plot_df, p9.aes(x='step_x', y='value', color='model', group='model'))
            + p9.geom_line(size=1.2)
            + p9.geom_point(size=2.5)
            + p9.labs(
        x='Training step',
        y=metric_label,
        color=None,
    )
            + p9.scale_x_continuous(breaks=x_breaks, labels=x_labels)
            + p9.scale_color_manual(values=MODEL_COLORS)
            + theme
    )
    p.save(plot_path, width=7, height=5, verbose=False)
    
