"""
feature incompatibility--need to just use w features
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata, build_meta_xy
from src.algorithms.binary import CatBoostAUCClassifier

source_model = 'MLP'
target_model = 'PatchTST'
results_dir = Path('./assets/results_cv')

metadata, category_mappings = read_all_metadata(
    './assets', source_model,
    processed_file=f'./assets/metadata_{source_model}.csv',
    # sample_n=20000
)

target_metadata, target_category_mappings = read_all_metadata(
    './assets', source_model,
    processed_file=f'./assets/metadata_{target_model}.csv',
    # sample_n=20000
)

source_data = build_meta_xy(metadata,
                            task="classification",
                            use_step_as_feature=True)

source_X = source_data.X
source_y = pd.Series(source_data.y)
source_groups = source_data.groups

target_data = build_meta_xy(target_metadata,
                            task="classification",
                            use_step_as_feature=True)

target_X = target_data.X
target_y = pd.Series(target_data.y)
target_groups = target_data.groups

logo = LeaveOneGroupOut()
fold_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
fold_scores: list[tuple[str, float, float, float]] = []
for train_idx, test_idx in logo.split(source_X, source_y, source_groups):
    held_out = source_groups.iloc[test_idx[0]]
    print(held_out)

    clf = CatBoostAUCClassifier(calibrate=True,
                                calibration_method='platt',
                                cal_size=0.15)

    clf.fit(source_X.iloc[train_idx], source_y.iloc[train_idx])

    test_mask = target_groups == held_out

    X_test = target_X.iloc[test_mask.values,:].reset_index(drop=True)
    y_ts = target_y.iloc[test_mask.values].reset_index(drop=True)

    preds_raw = clf.predict_proba(X_test, calibrated=False)[:, 1]
    preds = clf.predict_proba(X_test)[:, 1]

    fold_results[held_out] = (y_ts, preds_raw, preds)

    fold_auc = roc_auc_score(y_ts, preds)
    fold_ll = log_loss(y_ts, preds)
    fold_br = brier_score_loss(y_ts, preds)

    fold_scores.append((held_out, fold_auc, fold_ll, fold_br))
    print(f"{held_out}: AUC = {fold_auc:.3f}")

auc_df = pd.DataFrame(fold_scores, columns=['dataset', 'auc', 'll', 'brier'])
auc_df.mean(numeric_only=True)
auc_df.std(numeric_only=True)
print(auc_df.mean(numeric_only=True))

auc_df.set_index(['dataset'], inplace=True)

auc_df.loc['average'] = auc_df.mean(numeric_only=True)
auc_df.loc['std'] = auc_df.std(numeric_only=True)

auc_df.to_csv(results_dir / f'cv_meta_tl_{source_model}_{target_model}.csv')
