from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error as mae,
                             roc_auc_score,
                             log_loss,
                             brier_score_loss)
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata, build_meta_xy, corr_coef
from src.algorithms import CatBoostRegressionModel

model_name = 'MLP'
plot_path = Path('./assets/outputs') / f'metal_reg_step_{model_name}.pdf'

PERFORMANCE_DIFF = True
Y_CLIP = (-2.5, 2.5)

metadata = read_all_metadata(
    './assets', model_name,
    processed_file=f'./assets/metadata_{model_name}.csv',
)

steps = np.linspace(start=0, stop=1000, num=11).astype(int).tolist()
steps.append(-1)


def run_logo_cv_for_step(
        metadata: pd.DataFrame,
        step: int,
        performance_diff: bool = True,
        y_clip: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Run leave-one-dataset-out CV for regression at a given training step."""
    df_step = metadata.query(f'step == {step}').reset_index(drop=True)

    if df_step.empty:
        return {
            'step': step,
            'nmae': np.nan,
            'spearman': np.nan,
            'kendall': np.nan,
            'auc_exc': np.nan,
            'll_iso': np.nan,
            'brier_iso': np.nan,
        }

    data = build_meta_xy(
        df_step,
        task="regression",
        use_step_as_feature=False,
        performance_diff=performance_diff,
        y_clip=y_clip,
    )

    X = data.X
    y = data.y
    groups = data.groups
    mase_sn_by_dataset = data.mase_sn_by_dataset

    logo = LeaveOneGroupOut()
    fold_aucs: list[float] = []
    fold_lls: list[float] = []
    fold_briers: list[float] = []
    fold_spearmans: list[float] = []
    fold_kendalls: list[float] = []
    fold_nmaes: list[float] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = groups.iloc[test_idx[0]]
        y_tr = y[train_idx]
        y_ts = y[test_idx]

        reg = CatBoostRegressionModel(
            conformal=True,
            conformal_cal_size=0.15,
            calibration_method="platt",
        )
        reg.fit(X.iloc[train_idx], y_tr)

        preds = reg.predict(X.iloc[test_idx])
        y_baseline = np.repeat(np.mean(y_tr), len(y_ts))

        nmae = mae(y_ts, preds) / mae(y_ts, y_baseline)

        thr = 0 if performance_diff else mase_sn_by_dataset[held_out]
        y_exc_bin = (y_ts > thr).astype(int)
        pred_exc = reg.prob_exceeds(X.iloc[test_idx], thr, calibration_method="isotonic")

        fold_spearmans.append(corr_coef(y_ts, preds, 'spearman'))
        fold_kendalls.append(corr_coef(y_ts, preds, 'kendall'))
        fold_aucs.append(roc_auc_score(y_exc_bin, pred_exc))
        fold_lls.append(log_loss(y_exc_bin, pred_exc))
        fold_briers.append(brier_score_loss(y_exc_bin, pred_exc))
        fold_nmaes.append(nmae)

    return {
        'step': step,
        'nmae': np.mean(fold_nmaes),
        'spearman': np.mean(fold_spearmans),
        'kendall': np.mean(fold_kendalls),
        'auc_exc': np.mean(fold_aucs),
        'auc_exc_std': np.std(fold_aucs),
        'll_iso': np.mean(fold_lls),
        'brier_iso': np.mean(fold_briers),
    }


results: list[dict[str, float]] = []
for step in steps:
    print(f'Running step {step}...')
    metrics = run_logo_cv_for_step(
        metadata,
        step=step,
        performance_diff=PERFORMANCE_DIFF,
        y_clip=Y_CLIP,
    )
    results.append(metrics)
    print(f"  nMAE = {metrics['nmae']:.3f}, AUC = {metrics['auc_exc']:.3f}, "
          f"Spearman = {metrics['spearman']:.3f}")

results_df = pd.DataFrame(results)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print('\n', results_df)
