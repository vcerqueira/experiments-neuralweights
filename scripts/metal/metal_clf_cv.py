from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata, build_meta_xy
from src.algorithms.binary import CatBoostAUCClassifier
from src.plots import plot_roc_curve

model = 'MLP'
data_dir = Path('./assets/results')
plot_path = Path("./assets/outputs") / f"metal_clf_roc_{model}_logo.pdf"
plot_path_m3 = Path("./assets/outputs") / f"metal_clf_roc_{model}_monash_m3_monthly_logo.pdf"
target_dataset = 'monash_m3_monthly'

metadata, category_mappings = read_all_metadata(
    './assets', model,
    processed_file=f'./assets/metadata_{model}.csv',
    sample_n=100000
)

data = build_meta_xy(metadata,
                     task="classification",
                     use_step_as_feature=True)

X = data.X
y = data.y
groups = data.groups

print(data.groups.value_counts())

logo = LeaveOneGroupOut()
fold_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
fold_scores: list[tuple[str, float, float, float]] = []
for train_idx, test_idx in logo.split(X, y, groups):
    held_out = groups.iloc[test_idx[0]]
    print(held_out)

    clf = CatBoostAUCClassifier(calibrate=True,
                                calibration_method='platt',
                                cal_size=0.15)

    clf.fit(X.iloc[train_idx], y.iloc[train_idx])

    y_ts = y.iloc[test_idx].to_numpy()
    preds = clf.predict_proba(X.iloc[test_idx])[:, 1]

    # if all(y_ts == 0):
    #     y_ts = np.concatenate([y_ts, np.array([1])])
    #     preds = np.concatenate([preds, np.array([0])])

    fold_results[held_out] = (y_ts, preds)

    fold_auc = roc_auc_score(y_ts, preds)
    fold_ll = log_loss(y_ts, preds)
    fold_br = brier_score_loss(y_ts, preds)

    fold_scores.append((held_out, fold_auc, fold_ll, fold_br))
    print(f"{held_out}: AUC = {fold_auc:.3f}")

y_m3, preds_m3 = fold_results[target_dataset]
auc_m3 = roc_auc_score(y_m3, preds_m3)
print(f"{target_dataset} LOO AUC = {auc_m3:.3f}")

plot_roc_curve(
    y_m3,
    preds_m3,
    auc_m3,
    title=f"ROC Curve — {model} / {target_dataset} (leave-one-dataset-out)",
    save_path=plot_path_m3,
)

auc_df = pd.DataFrame(fold_scores, columns=['dataset', 'auc', 'll', 'brier'])
auc_df.mean(numeric_only=True)
auc_df.std(numeric_only=True)
print(auc_df.mean(numeric_only=True))
