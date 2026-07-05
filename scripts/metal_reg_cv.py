from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error as mae,
                             roc_auc_score,
                             log_loss,
                             brier_score_loss)
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata
from src.algorithms import CatBoostRegressionModel
from src.plots import plot_feature_importance

model_name = 'MLP'
data_dir = Path('./assets/results')
plot_path = Path("./assets/outputs") / f"metal_reg_importance_{model_name}_logo.pdf"
target_dataset = 'monash_m3_monthly'
PERFORMANCE_DIFF = True

if PERFORMANCE_DIFF:
    y_clip_min, y_clip_max = -2.5, 2.5
else:
    y_clip_min, y_clip_max = 0, 5

metadata = read_all_metadata(data_dir, model_name, detailed=False)

df_after_train = metadata.query('step==-1').reset_index(drop=True)
print(df_after_train['dataset'].value_counts())

mase_sn_by_dataset = df_after_train.groupby('dataset')['mase_sn'].first()
if PERFORMANCE_DIFF:
    y_reg = df_after_train['mase_sn'] - df_after_train['mase']
else:
    y_reg = df_after_train['mase']

groups = df_after_train['dataset']
X = df_after_train.drop(columns=['mase', 'mase_sn', 'model', 'config_id', 'step', 'dataset'])

logo = LeaveOneGroupOut()
y_true_folds: list[np.ndarray] = []
pred_folds: list[np.ndarray] = []
baseline_folds: list[np.ndarray] = []
fold_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
fold_metrics: list[dict[str, object]] = []

for train_idx, test_idx in logo.split(X, y_reg, groups):
    held_out = groups.iloc[test_idx[0]]
    y_tr = np.clip(y_reg.iloc[train_idx], a_min=y_clip_min, a_max=y_clip_max)
    y_ts = y_reg.iloc[test_idx].to_numpy()

    reg = CatBoostRegressionModel(conformal=True)
    reg.fit(X.iloc[train_idx], y_tr)

    preds = reg.predict(X.iloc[test_idx])
    y_baseline = np.repeat(np.mean(y_tr), len(y_ts))

    y_true_folds.append(y_ts)
    pred_folds.append(preds)
    baseline_folds.append(y_baseline)

    nmae_fold = mae(y_ts, preds) / mae(y_ts, y_baseline)

    thr = 0 if PERFORMANCE_DIFF else mase_sn_by_dataset[held_out]
    y_exc_bin = (y_ts > thr).astype(int)
    pred_exc = reg.prob_exceeds(X.iloc[test_idx], thr)

    auc_exc = roc_auc_score(y_exc_bin, pred_exc)
    ll_exc = log_loss(y_exc_bin, pred_exc)
    brier_exc = brier_score_loss(y_exc_bin, pred_exc)

    fold_results[held_out] = (y_ts, preds, pred_exc)
    fold_metrics.append({
        'dataset': held_out,
        'nmae': nmae_fold,
        'auc_exc': auc_exc,
        'll_exc': ll_exc,
        'brier_exc': brier_exc,
    })
    print(f"{held_out}: nMAE = {nmae_fold:.3f}, exceedance AUC = {auc_exc:.3f}")

y_all = np.concatenate(y_true_folds)
preds_all = np.concatenate(pred_folds)
baseline_all = np.concatenate(baseline_folds)
nmae = mae(y_all, preds_all) / mae(y_all, baseline_all)
print(f"\nOverall LOO-dataset nMAE = {nmae:.3f}")

metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df[['nmae', 'auc_exc', 'll_exc', 'brier_exc']].agg(['mean', 'std']))

if target_dataset in fold_results:
    y_m3, preds_m3, pred_exc_m3 = fold_results[target_dataset]
    thr_m3 = mase_sn_by_dataset[target_dataset]
    y_exc_m3 = (y_m3 > thr_m3).astype(int)
    auc_exc_m3 = roc_auc_score(y_exc_m3, pred_exc_m3) if len(np.unique(y_exc_m3)) > 1 else np.nan
    nmae_m3 = metrics_df.loc[metrics_df['dataset'] == target_dataset, 'nmae'].iloc[0]
    print(f"\n{target_dataset} LOO nMAE = {nmae_m3:.3f}, exceedance AUC = {auc_exc_m3:.3f}")

reg = CatBoostRegressionModel(conformal=True)
reg.fit(X, np.clip(y_reg, a_min=y_clip_min, a_max=y_clip_max))

importances = reg.feature_importance()
print(importances)

plot_feature_importance(
    importances,
    title=f"Feature Importance — {model_name} (all datasets)",
    save_path=plot_path,
)
