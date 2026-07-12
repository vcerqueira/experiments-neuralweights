from pathlib import Path

import pandas as pd
import plotnine as p9

from src.utils import read_all_metadata, build_meta_xy
from src.algorithms import CatBoostRegressionModel, CatBoostAUCClassifier

MODEL = 'MLP'
OUTPUT_DIR = Path('./assets/outputs')
TOP_N = 15

metadata, _ = read_all_metadata(
    './assets',
    MODEL,
    processed_file=f'./assets/metadata_{MODEL}.csv',
    # sample_n=100000
)

data_reg = build_meta_xy(
    metadata,
    task='regression',
    use_step_as_feature=True,
    performance_diff=True,
    y_clip=(-2.5, 2.5),
)

data_clf = build_meta_xy(
    metadata,
    task='classification',
    use_step_as_feature=True,
)

# reg = CatBoostRegressionModel(conformal=True)
# reg.fit(data_reg.X, data_reg.y)

clf = CatBoostAUCClassifier(
    calibrate=True,
    calibration_method='platt',
    cal_size=0.05,
)
clf.fit(data_clf.X, data_clf.y)

# importances_reg = reg.feature_importance()
importances_clf = clf.feature_importance().head(TOP_N)

imp_df = importances_clf.reset_index()
imp_df.columns = ['Feature', 'Importance']

feature_order = imp_df.sort_values('Importance', ascending=True)['Feature'].tolist()
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=feature_order)

p = (p9.ggplot(imp_df, p9.aes(x='Feature', y='Importance')) +
     p9.geom_bar(stat='identity', width=0.75, show_legend=False, fill='steelblue') +
     p9.coord_flip() +
     p9.scale_fill_brewer(type='qual', palette='Set1') +
     p9.scale_fill_brewer(type='qual', palette='Set1') +
     p9.labs(
         x='',
         y='Importance',
     ) +
     p9.theme_538(base_family='Palatino', base_size=14) +
     p9.theme(
         plot_margin=0.025,
         panel_background=p9.element_rect(fill='white'),
         plot_background=p9.element_rect(fill='white'),
         legend_box_background=p9.element_rect(fill='white'),
         strip_background=p9.element_rect(fill='white'),
         legend_background=p9.element_rect(fill='white'),
         axis_text_y=p9.element_text(size=13),
         legend_title=p9.element_blank(),
     ))

p.save(OUTPUT_DIR / f'feature_importance_{MODEL}.pdf', height=4, width=4)
