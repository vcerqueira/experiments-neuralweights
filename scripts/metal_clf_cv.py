from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from src.utils import read_all_metadata
from src.algorithms.binary import CatBoostAUCClassifier
from src.plots import plot_roc_curve

model = 'MLP'
dir = Path('./assets/results')
plot_path = Path("./assets/outputs") / f"metal_clf_roc_{model}_logo.pdf"
plot_path_m3 = Path("./assets/outputs") / f"metal_clf_roc_{model}_monash_m3_monthly_logo.pdf"
target_dataset = 'monash_m3_monthly'

metadata = read_all_metadata(dir, model, detailed=False)

df_after_train = metadata.query('step==-1').reset_index(drop=True)
print(df_after_train['dataset'].value_counts())

y_clf = (df_after_train['mase'] < df_after_train['mase_sn']).astype(int)
groups = df_after_train['dataset']
X = df_after_train.drop(columns=['mase', 'mase_sn', 'model', 'config_id', 'step', 'dataset'])

logo = LeaveOneGroupOut()
y_true_folds: list[np.ndarray] = []
y_pred_folds: list[np.ndarray] = []
fold_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
fold_aucs: list[tuple[str, float]] = []

for train_idx, test_idx in logo.split(X, y_clf, groups):
    held_out = groups.iloc[test_idx[0]]

    clf = CatBoostAUCClassifier()
    clf.fit(X.iloc[train_idx], y_clf.iloc[train_idx])

    y_ts = y_clf.iloc[test_idx].to_numpy()
    preds = clf.predict_proba(X.iloc[test_idx])[:, 1]

    y_true_folds.append(y_ts)
    y_pred_folds.append(preds)
    fold_results[held_out] = (y_ts, preds)

    fold_auc = roc_auc_score(y_ts, preds)
    fold_aucs.append((held_out, fold_auc))
    print(f"{held_out}: AUC = {fold_auc:.3f}")

y_ts = np.concatenate(y_true_folds)
preds = np.concatenate(y_pred_folds)
auc = roc_auc_score(y_ts, preds)
print(f"\nOverall LOO-dataset AUC = {auc:.3f}")

plot_roc_curve(
    y_ts,
    preds,
    auc,
    title=f"ROC Curve — {model} (leave-one-dataset-out)",
    save_path=plot_path,
)

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

auc_df = pd.DataFrame(fold_aucs, columns=['dataset', 'auc'])
auc_df.mean(numeric_only=True)
auc_df.std(numeric_only=True)

