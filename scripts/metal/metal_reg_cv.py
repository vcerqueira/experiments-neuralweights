from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error as mae,
                             roc_auc_score,
                             log_loss,
                             brier_score_loss)
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata, corr_coef
from src.algorithms import CatBoostRegressionModel
from src.plots import plot_feature_importance, plot_calibration_curve

model_name = 'MLP'
data_dir = Path('./assets/results')
plot_path = Path("./assets/outputs") / f"metal_reg_importance_{model_name}_logo.pdf"
target_dataset = 'monash_tourism_monthly'
PERFORMANCE_DIFF = True

if PERFORMANCE_DIFF:
    y_clip_min, y_clip_max = -2.5, 2.5
else:
    y_clip_min, y_clip_max = 0, 5

# metadata = read_all_metadata(data_dir, model_name, detailed=False)
metadata = pd.read_csv('./assets/metadata.csv')
object_cols = metadata.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    metadata[col] = metadata[col].astype('category').cat.codes

df_after_train = metadata.sample(100000).reset_index(drop=True)
# df_after_train = metadata.query('step==-1').reset_index(drop=True)
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
exc_true_folds: list[np.ndarray] = []
exc_raw_folds: list[np.ndarray] = []
exc_isotonic_folds: list[np.ndarray] = []
exc_platt_folds: list[np.ndarray] = []
fold_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
fold_metrics: list[dict[str, object]] = []

for train_idx, test_idx in logo.split(X, y_reg, groups):
    held_out = groups.iloc[test_idx[0]]
    y_tr = np.clip(y_reg.iloc[train_idx], a_min=y_clip_min, a_max=y_clip_max)
    y_ts = y_reg.iloc[test_idx].to_numpy()

    reg = CatBoostRegressionModel(conformal=True, conformal_cal_size=0.2)
    reg.fit(X.iloc[train_idx], y_tr)

    preds = reg.predict(X.iloc[test_idx])
    y_baseline = np.repeat(np.mean(y_tr), len(y_ts))

    y_true_folds.append(y_ts)
    pred_folds.append(preds)
    baseline_folds.append(y_baseline)

    nmae_fold = mae(y_ts, preds) / mae(y_ts, y_baseline)
    cc_k = corr_coef(y_ts, preds, 'kendall')
    cc_s = corr_coef(y_ts, preds, 'spearman')

    thr = 0 if PERFORMANCE_DIFF else mase_sn_by_dataset[held_out]
    y_exc_bin = (y_ts > thr).astype(int)
    pred_exc_raw = reg.prob_exceeds(X.iloc[test_idx], thr, calibration_method="none")
    pred_exc_isotonic = reg.prob_exceeds(X.iloc[test_idx], thr, calibration_method="isotonic")
    pred_exc_platt = reg.prob_exceeds(X.iloc[test_idx], thr, calibration_method="platt")

    exc_true_folds.append(y_exc_bin)
    exc_raw_folds.append(pred_exc_raw)
    exc_isotonic_folds.append(pred_exc_isotonic)
    exc_platt_folds.append(pred_exc_platt)

    auc_exc = roc_auc_score(y_exc_bin, pred_exc_raw)
    ll_raw = log_loss(y_exc_bin, pred_exc_raw)
    ll_iso = log_loss(y_exc_bin, pred_exc_isotonic)
    ll_platt = log_loss(y_exc_bin, pred_exc_platt)
    brier_raw = brier_score_loss(y_exc_bin, pred_exc_raw)
    brier_iso = brier_score_loss(y_exc_bin, pred_exc_isotonic)
    brier_platt = brier_score_loss(y_exc_bin, pred_exc_platt)

    fold_results[held_out] = (y_ts, preds, pred_exc_isotonic)
    fold_metrics.append({
        'dataset': held_out,
        'nmae': nmae_fold,
        'kendall': cc_k,
        'spearman': cc_s,
        'auc_exc': auc_exc,
        'll_raw': ll_raw,
        'll_iso': ll_iso,
        'll_platt': ll_platt,
        'brier_raw': brier_raw,
        'brier_iso': brier_iso,
        'brier_platt': brier_platt,
    })
    print(f"{held_out}: nMAE={nmae_fold:.3f}, AUC={auc_exc:.3f}, "
          f"LL(raw/iso/platt)={ll_raw:.3f}/{ll_iso:.3f}/{ll_platt:.3f}")

y_all = np.concatenate(y_true_folds)
preds_all = np.concatenate(pred_folds)
baseline_all = np.concatenate(baseline_folds)
nmae = mae(y_all, preds_all) / mae(y_all, baseline_all)
print(f"\nOverall LOO-dataset nMAE = {nmae:.3f}")

metrics_df = pd.DataFrame(fold_metrics)
print("\n--- Metrics Summary (mean ± std) ---")
summary_cols = ['nmae', 'auc_exc', 'kendall', 'spearman', 'll_raw', 'll_iso',
                'll_platt', 'brier_raw', 'brier_iso',
                'brier_platt']
print(metrics_df[summary_cols].agg(['mean', 'std']).T)

exc_true_all = np.concatenate(exc_true_folds)
exc_raw_all = np.concatenate(exc_raw_folds)
exc_iso_all = np.concatenate(exc_isotonic_folds)
exc_platt_all = np.concatenate(exc_platt_folds)

calib_plot_path = Path("./assets/outputs") / f"metal_reg_calibration_{model_name}_logo.pdf"
plot_calibration_curve(
    exc_true_all,
    exc_raw_all,
    y_prob_calibrated={"isotonic": exc_iso_all, "platt": exc_platt_all},
    n_bins=10,
    title=f"Calibration Curve — Exceedance Probability ({model_name})",
    save_path=calib_plot_path,
)
print(f"\nCalibration curve saved to {calib_plot_path}")

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
