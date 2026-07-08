from pathlib import Path

import pandas as pd

from src.utils import read_all_metadata, build_meta_xy
from src.algorithms import CatBoostRegressionModel, CatBoostAUCClassifier
from src.plots import plot_feature_importance

model = 'MLP'
plot_path = Path("./assets/outputs") / f"metal_reg_importance_{model}_logo.pdf"

metadata = read_all_metadata(
    './assets', model,
    processed_file=f'./assets/metadata_{model}.csv',
    sample_n=100000
)

data_reg = build_meta_xy(metadata,
                         task="regression",
                         use_step_as_feature=True,
                         performance_diff=True,
                         y_clip=(-2.5, 2.5))

data_clf = build_meta_xy(metadata,
                         task="classification",
                         use_step_as_feature=True)

reg = CatBoostRegressionModel(conformal=True)
reg.fit(data_reg.X, data_reg.y)

clf = CatBoostAUCClassifier(calibrate=True,
                            calibration_method='platt',
                            cal_size=0.05)

clf.fit(data_clf.X, data_clf.y)

importances_reg = reg.feature_importance()
importances_clf = clf.feature_importance()

import_df = pd.concat([importances_reg, importances_clf], axis=1)

# todo grouped barplot
plot_feature_importance(
    importances_reg,
    title=f"Feature Importance — {model} (all datasets)",
    save_path=plot_path,
)
