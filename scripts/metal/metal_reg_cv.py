from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error as mae,
                             roc_auc_score,
                             log_loss,
                             brier_score_loss)
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata, corr_coef, build_meta_xy
from src.algorithms import CatBoostRegressionModel
from src.plots import plot_calibration_curve

model = 'NHITS'
results_dir = Path('./assets/results_cv')
PERFORMANCE_DIFF = True
Y_CLIP = (-2.5, 2.5)
# Y_CLIP = (0, 4)

metadata, category_mappings = read_all_metadata(
    './assets', model,
    processed_file=f'./assets/metadata_{model}.csv',
    # sample_n=200000
)

data = build_meta_xy(metadata,
                     task="regression",
                     use_step_as_feature=True,
                     performance_diff=PERFORMANCE_DIFF,
                     y_clip=Y_CLIP)

X = data.X
y = pd.Series(data.y)
groups = data.groups
mase_sn_by_dataset = data.mase_sn_by_dataset

logo = LeaveOneGroupOut()
exc_true_folds: list[np.ndarray] = []
exc_raw_folds: list[np.ndarray] = []
exc_isotonic_folds: list[np.ndarray] = []
exc_platt_folds: list[np.ndarray] = []
fold_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
fold_metrics: list[dict[str, object]] = []

for train_idx, test_idx in logo.split(X, y, groups):
    held_out = groups.iloc[test_idx[0]]
    y_tr = y.iloc[train_idx].to_numpy()
    y_ts = y.iloc[test_idx].to_numpy()

    reg = CatBoostRegressionModel(conformal=True, conformal_cal_size=0.1)
    reg.fit(X.iloc[train_idx], y_tr)

    preds = reg.predict(X.iloc[test_idx])
    y_baseline = np.repeat(np.mean(y_tr), len(y_ts))

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

metrics_df = pd.DataFrame(fold_metrics)
print("\n--- Metrics Summary (mean ± std) ---")
summary_cols = ['nmae', 'auc_exc', 'kendall', 'spearman', 'll_raw', 'll_iso',
                'll_platt', 'brier_raw', 'brier_iso',
                'brier_platt']
print(metrics_df[summary_cols].agg(['mean', 'std']).T)

calib_plot_path = Path("./assets/outputs") / f"metal_reg_calibration_{model}_logo.pdf"
plot_calibration_curve(
    exc_true_folds[2],
    exc_raw_folds[2],
    y_prob_calibrated={"isotonic": exc_isotonic_folds[2], "platt": exc_platt_folds[2]},
    n_bins=10,
    title=f"",
    save_path=calib_plot_path,
    raw_label="Raw (conformal)",
)


metrics_df.set_index(['dataset'], inplace=True)

metrics_df.loc['average'] = metrics_df.mean(numeric_only=True)
metrics_df.loc['std'] = metrics_df.std(numeric_only=True)

metrics_df.to_csv(results_dir / f'cv_reg_scores_{model}.csv')
