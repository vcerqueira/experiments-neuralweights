from pathlib import Path

import pandas as pd
import plotnine as p9

from src.workflows.metadata_utils import read_all_metadata, build_meta_xy
from src.weightcast.learner_classifier import CatBoostAUCClassifier
from src.weightcast.learner_regressor import CatBoostRegressionModel

MODEL = 'NHITS'
OUTPUT_DIR = Path('./assets/outputs')
TOP_N = 15

metadata, _ = read_all_metadata(
    './assets',
    MODEL,
    processed_file=f'./assets/metadata_{MODEL}.csv',
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

reg = CatBoostRegressionModel(conformal=False)
reg.fit(data_reg.X, data_reg.y)

clf = CatBoostAUCClassifier(
    calibrate=False,
    calibration_method='platt',
    cal_size=0.05,
)
clf.fit(data_clf.X, data_clf.y)

importances_reg = reg.feature_importance().head(TOP_N)
importances_clf = clf.feature_importance().head(TOP_N)

def plot_importance(import_scores):

    imp_df = import_scores.reset_index()
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

    return p

p1 =plot_importance(importances_reg)
p2 = plot_importance(importances_clf)

p1.save(OUTPUT_DIR / f'feature_importance_{MODEL}_reg.pdf', height=4, width=4)
p2.save(OUTPUT_DIR / f'feature_importance_{MODEL}_clf.pdf', height=4, width=4)
