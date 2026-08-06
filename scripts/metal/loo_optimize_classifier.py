import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from src.workflows.metadata_utils import read_all_metadata, build_meta_xy
from src.weightcast.learner_classifier import CatBoostAUCClassifier

MODEL_NAME = 'PatchTST'
N_TRIALS = 50
SAMPLE_N = 50000

metadata, _ = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
    sample_n=SAMPLE_N,
)

data = build_meta_xy(metadata, task="classification", use_step_as_feature=True)
X, y, groups = data.X, pd.Series(data.y), data.groups

by_dataset: dict[str, dict] = {}
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    held_out = groups.iloc[test_idx[0]]
    print(f"\nOptimizing (held out: {held_out})...")

    clf = CatBoostAUCClassifier(
        calibrate=False,
        optimize=True,
        n_trials=N_TRIALS,
        calibration_method='platt',
        cal_size=0.15,
    )
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])

    by_dataset[held_out] = clf.best_params_

print(by_dataset)
