from pathlib import Path

import pandas as pd
import plotnine as p9

from src.utils import read_all_metadata

MODEL_NAME = 'MLP'
OUTPUT_DIR = Path('./assets/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_FEATURES = ['entropy', 'sv_min', 'num_layers', 'stable_rank']

metadata, _ = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
)

metadata['class'] = (metadata['mase'] < metadata['mase_sn']).map({True: 'Better', False: 'Worse'})
metadata['class'] = pd.Categorical(metadata['class'], categories=['Better', 'Worse'])

for feature in TOP_FEATURES:
    if feature not in metadata.columns:
        print(f"  Skipping {feature} (not in columns)")
        continue

    df_plot = metadata[[feature, 'class']].dropna()

    p = (
            p9.ggplot(df_plot, p9.aes(x=feature, fill='class', color='class'))
            + p9.geom_density(alpha=0.4)
            + p9.scale_fill_brewer(type='qual', palette='Set1')
            + p9.scale_color_brewer(type='qual', palette='Set1')
            + p9.labs(
        x=feature,
        y='Density',
        fill='vs Seasonal Naive',
        color='vs Seasonal Naive',
    )
            + p9.theme_538(base_family='Palatino', base_size=12)
            + p9.theme(plot_margin=.025,
                       panel_background=p9.element_rect(fill='white'),
                       plot_background=p9.element_rect(fill='white'),
                       legend_box_background=p9.element_rect(fill='white'),
                       strip_background=p9.element_rect(fill='white'),
                       legend_background=p9.element_rect(fill='white'),
                       # axis_text_x=p9.element_text(size=9, angle=0),
                       axis_text_y=p9.element_text(size=9),
                       legend_title=p9.element_blank())

    )

    save_path = OUTPUT_DIR / f'density_{feature}.pdf'
    p.save(save_path, width=7, height=7)
