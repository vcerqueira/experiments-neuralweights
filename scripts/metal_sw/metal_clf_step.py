from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata, build_meta_xy
from src.algorithms import CatBoostAUCClassifier

model_name = 'MLP'
data_dir = Path('./assets/results')
plot_path = Path('./assets/outputs') / f'metal_clf_step_{model_name}.pdf'

metadata = read_all_metadata(
    './assets', model_name,
    processed_file=f'./assets/metadata_{model_name}.csv',
)

steps = np.linspace(start=0, stop=1000, num=11).astype(int).tolist()
steps.append(-1)


def run_logo_cv_for_step(
        metadata: pd.DataFrame,
        step: int,
) -> dict[str, float]:
    """Run leave-one-dataset-out CV for classification at a given training step."""
    df_step = metadata.query(f'step == {step}').reset_index(drop=True)
    # if step != -1:
    #     df_step = metadata.query(f'step <= {step}').reset_index(drop=True)
    # else:
    #     df_step = metadata

    if df_step.empty:
        return {
            'step': step,
            'auc_mean': np.nan,
            'auc_std': np.nan,
            'll_mean': np.nan,
            'brier_mean': np.nan,
        }

    data = build_meta_xy(
        df_step,
        task="classification",
        use_step_as_feature=False,
    )

    X = data.X
    y = data.y
    groups = data.groups

    logo = LeaveOneGroupOut()
    fold_aucs: list[float] = []
    fold_lls: list[float] = []
    fold_briers: list[float] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        clf = CatBoostAUCClassifier(
            calibrate=True,
            calibration_method='platt',
            cal_size=0.1,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])

        y_ts = y[test_idx]
        preds = clf.predict_proba_positive(X.iloc[test_idx])

        fold_aucs.append(roc_auc_score(y_ts, preds))
        fold_lls.append(log_loss(y_ts, preds))
        fold_briers.append(brier_score_loss(y_ts, preds))

    return {
        'step': step,
        'auc_mean': np.mean(fold_aucs),
        'auc_std': np.std(fold_aucs),
        'll_mean': np.mean(fold_lls),
        'll_std': np.std(fold_lls),
        'brier_mean': np.mean(fold_briers),
        'brier_std': np.std(fold_briers),
    }


results: list[dict[str, float]] = []
for step in steps:
    print(f'Running step {step}...')
    metrics = run_logo_cv_for_step(
        metadata,
        step=step,
    )
    results.append(metrics)
    print(f"  AUC = {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}, "
          f"LogLoss = {metrics['ll_mean']:.3f}, Brier = {metrics['brier_mean']:.3f}")

results_df = pd.DataFrame(results)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print('\n', results_df)
