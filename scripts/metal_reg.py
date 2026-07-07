from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error as mae,
                             roc_auc_score,
                             log_loss,
                             brier_score_loss)

from src.utils import read_metadata, corr_coef
from src.algorithms import CatBoostRegressionModel
from src.plots import plot_feature_importance

model_name = 'MLP'
dataset_name = 'monash_m1_monthly'
data_dir = Path('./assets/results')
plot_path = Path("./assets/outputs") / f"metal_reg_importance_{model_name}_{dataset_name}.pdf"

metadata = read_metadata(data_dir, model_name, dataset_name, detailed=False)

df_after_train = metadata.query('step==-1').reset_index(drop=True)
mase_sn_by_dataset = df_after_train.groupby('dataset')['mase_sn'].first()

# metadata splitting
y_reg = df_after_train['mase'] / df_after_train['mase_sn']
X = df_after_train.drop(columns=['mase', 'mase_sn', 'model', 'config_id', 'step', 'dataset'])

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y_reg, test_size=0.3)

# modeling
y_baseline = np.repeat(y_tr.median(), len(y_ts))

model = CatBoostRegressionModel(conformal=True)
# is clipping "safe" when doing CP?
y_tr = np.clip(y_tr, a_min=0, a_max=2)
model.fit(X_tr, np.clip(y_tr, a_min=0, a_max=2))
# model.fit(X_tr, y_tr)

# negative predicted values, why? -- solved when clipping
preds = model.predict(X_ts)

mae_score_bl = mae(y_ts, y_baseline)
mae_score = mae(y_ts, preds)
nmae = mae_score / mae_score_bl
print(nmae)
print(corr_coef(y_ts, preds))

thr = mase_sn_by_dataset[dataset_name]
y_exc_bin = (y_ts > thr).astype(int)

pred_exc = model.prob_exceeds(X_ts, thr)

auc_exc = roc_auc_score(y_exc_bin, pred_exc)
ll_exc = log_loss(y_exc_bin, pred_exc)
brier_exc = brier_score_loss(y_exc_bin, pred_exc)
print(auc_exc)


importances = model.feature_importance()
print(importances)

plot_feature_importance(
    importances,
    title=f"Feature Importance — {model_name} / {dataset_name}",
    save_path=plot_path,
)

