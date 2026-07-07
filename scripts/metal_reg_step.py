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

model_name = 'MLP'
data_dir = Path('./assets/results')
plot_path = Path('./assets/outputs') / f'metal_reg_step_{model_name}.pdf'
PERFORMANCE_DIFF = True

if PERFORMANCE_DIFF:
    y_clip_min, y_clip_max = -5, 5
else:
    y_clip_min, y_clip_max = 0, 5

# metadata = read_all_metadata(data_dir, model_name, detailed=False)
metadata = pd.read_csv('./assets/metadata.csv')
object_cols = metadata.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    metadata[col] = metadata[col].astype('category').cat.codes

# df_after_train = metadata.sample(50000).reset_index(drop=True)

mase_sn_by_dataset = metadata.query('step==-1').groupby('dataset')['mase_sn'].first()

steps = np.linspace(start=0, stop=1000, num=11).astype(int).tolist()
steps.append(-1)


def run_logo_cv_for_step(
        metadata: pd.DataFrame,
        step: int,
        y_clip_min: float,
        y_clip_max: float,
        performance_diff: bool,
) -> dict[str, float]:
    """Run leave-one-dataset-out CV for a given training step and return aggregate metrics."""
    df_step = metadata.query(f'step == {step}').reset_index(drop=True)

    if df_step.empty:
        return {'step': step, 'nmae': np.nan, 'auc_exc_mean': np.nan, 'auc_exc_std': np.nan}

    if performance_diff:
        y_reg = df_step['mase_sn'] - df_step['mase']
    else:
        y_reg = df_step['mase']

    groups = df_step['dataset']
    X = df_step.drop(columns=['mase', 'mase_sn', 'model', 'config_id', 'step', 'dataset'])

    logo = LeaveOneGroupOut()
    y_true_folds: list[np.ndarray] = []
    pred_folds: list[np.ndarray] = []
    baseline_folds: list[np.ndarray] = []
    fold_aucs: list[float] = []
    fold_ll: list[float] = []
    fold_bs: list[float] = []
    fold_bs_bl: list[float] = []
    fold_ccs: list[float] = []
    fold_cck: list[float] = []

    for train_idx, test_idx in logo.split(X, y_reg, groups):
        held_out = groups.iloc[test_idx[0]]
        y_tr = np.clip(y_reg.iloc[train_idx], a_min=y_clip_min, a_max=y_clip_max)
        y_ts = y_reg.iloc[test_idx].to_numpy()

        reg = CatBoostRegressionModel(conformal=True, calibration_method="isotonic")
        reg.fit(X.iloc[train_idx], y_tr)

        preds = reg.predict(X.iloc[test_idx])
        y_baseline = np.repeat(np.mean(y_tr), len(y_ts))
        y_baseline_prob = np.repeat(0.5, len(y_ts))

        y_true_folds.append(y_ts)
        pred_folds.append(preds)
        baseline_folds.append(y_baseline)

        thr = 0 if performance_diff else mase_sn_by_dataset[held_out]
        y_exc_bin = (y_ts > thr).astype(int)
        pred_exc = reg.prob_exceeds(X.iloc[test_idx], thr, calibration_method="isotonic")

        cc_k = corr_coef(y_ts, preds, 'kendall')
        cc_s = corr_coef(y_ts, preds, 'spearman')

        fold_ll.append(log_loss(y_exc_bin, pred_exc))
        fold_bs.append(brier_score_loss(y_exc_bin, pred_exc))
        fold_bs_bl.append(brier_score_loss(y_exc_bin, y_baseline_prob))
        fold_aucs.append(roc_auc_score(y_exc_bin, pred_exc))
        fold_ccs.append(cc_s)
        fold_cck.append(cc_k)

    y_all = np.concatenate(y_true_folds)
    preds_all = np.concatenate(pred_folds)
    baseline_all = np.concatenate(baseline_folds)
    nmae = mae(y_all, preds_all) / mae(y_all, baseline_all)


    return {
        'step': step,
        'nmae': nmae,
        'spearman': np.mean(fold_ccs),
        'kendall': np.mean(fold_cck),
        'brier': np.mean(fold_bs),
        'brier_bl': np.mean(fold_bs_bl),
        'll': np.mean(fold_ll),
        'auc_exc_mean': np.mean(fold_aucs) if fold_aucs else np.nan,
        'auc_exc_std': np.std(fold_aucs) if fold_aucs else np.nan,
    }


results: list[dict[str, float]] = []
for step in steps:
    print(f'Running step {step}...')
    metrics = run_logo_cv_for_step(
        metadata,
        step=step,
        y_clip_min=y_clip_min,
        y_clip_max=y_clip_max,
        performance_diff=PERFORMANCE_DIFF,
    )
    results.append(metrics)
    print(f"  nMAE = {metrics['nmae']:.3f}, exceedance AUC = {metrics['auc_exc_mean']:.3f}")

results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print('\n', results_df)
