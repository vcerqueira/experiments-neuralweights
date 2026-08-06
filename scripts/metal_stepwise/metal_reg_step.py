from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error as mae,
                             roc_auc_score,
                             log_loss,
                             r2_score,
                             brier_score_loss)
from sklearn.model_selection import LeaveOneGroupOut

from src.workflows.metadata_utils import read_all_metadata, build_meta_xy, corr_coef
from src.weightcast.learner_regressor import CatBoostRegressionModel
from src.workflows.cb_config import CATBOOST_CONFIGS_REG

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_name = 'PatchTST'
results_dir = Path('./assets/results_cv')
plot_path = Path('./assets/outputs') / f'metal_reg_step_{model_name}.pdf'

PERFORMANCE_DIFF = True
Y_CLIP = (-2.5, 2.5)

metadata, _ = read_all_metadata(
    './assets', model_name,
    processed_file=f'./assets/metadata_{model_name}.csv',
)

steps = np.linspace(start=0, stop=1500, num=16).astype(int).tolist()
steps.append(-1)


def run_logo_cv_for_step(
        metadata: pd.DataFrame,
        step: int,
        model_name: str,
        performance_diff: bool = True,
        y_clip: tuple[float, float] | None = None,
) -> dict[str, float]:
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
    fold_r2: list[float] = []
    fold_nmaes: list[float] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = groups.iloc[test_idx[0]]
        y_tr = y[train_idx]
        y_ts = y[test_idx]

        reg = CatBoostRegressionModel(
            conformal=True,
            conformal_cal_size=0.1,
            calibration_method="platt",
            catboost_params=CATBOOST_CONFIGS_REG[model_name][held_out],
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
        fold_r2.append(r2_score(y_ts, preds))
        fold_aucs.append(roc_auc_score(y_exc_bin, pred_exc))
        fold_lls.append(log_loss(y_exc_bin, pred_exc))
        fold_briers.append(brier_score_loss(y_exc_bin, pred_exc))
        fold_nmaes.append(nmae)

    return {
        'step': step,
        'nmae': np.mean(fold_nmaes),
        'spearman': np.mean(fold_spearmans),
        'r2': np.mean(fold_r2),
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
        model_name=model_name,
    )
    results.append(metrics)
    print(f"  nMAE = {metrics['nmae']:.3f}, AUC = {metrics['auc_exc']:.3f}, "
          f"Spearman = {metrics['spearman']:.3f}")

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(results_dir / f'metal_reg_step_{model_name}.csv', index=False)
