from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.utils import read_metadata
from src.algorithms.binary import CatBoostAUCClassifier
from src.plots import plot_roc_curve

model = 'MLP'
dataset_name = 'monash_m1_monthly'
dir = Path('./assets/results')
plot_path = Path("./assets/outputs") / f"metal_clf_roc_{model}_{dataset_name}.pdf"

metadata = read_metadata(dir, model, dataset_name, detailed=False)

df_after_train = metadata.query('step==-1').reset_index(drop=True)

# metadata splitting

y_clf = (df_after_train['mase'] < df_after_train['mase_sn']).astype(int)
X = df_after_train.drop(columns=['mase', 'mase_sn', 'model', 'config_id', 'step', 'dataset'])

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y_clf, test_size=0.2)

# modeling

clf = CatBoostAUCClassifier()
clf.fit(X_tr, y_tr)

preds = clf.predict_proba(X_ts)[:, 1]
auc = roc_auc_score(y_ts, preds)
print(auc)

plot_roc_curve(
    y_ts,
    preds,
    auc,
    title=f"ROC Curve — {model} / {dataset_name}",
    save_path=plot_path,
)
