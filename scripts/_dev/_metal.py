from pathlib import Path

import numpy as np

from src.utils import read_metadata
from src.algorithms.binary import CatBoostAUCClassifier
from src.algorithms.regression import CatBoostRegressionModel

model = 'MLP'
dataset_name = 'monash_m1_monthly'
dir = Path('./assets/results')

metadata = read_metadata(dir, model, dataset_name, detailed=False)

df_after_train = metadata.query('step==-1').reset_index(drop=True)

print(df_after_train.iloc[0])

(df_after_train['mase'] > df_after_train['mase_sn']).value_counts()

y_clf = (df_after_train['mase'] < df_after_train['mase_sn']).astype(int)
y_reg = df_after_train['mase']
X = df_after_train.drop(columns=['mase', 'mase_sn', 'model', 'config_id', 'step', 'dataset'])

clf = CatBoostAUCClassifier(optimize=False)
clf.fit(X, y_clf)
clf.predict_proba(X)


reg = CatBoostRegressionModel(optimize=False, conformal=False, conformal_cal_size=0.2)
reg.fit(X, y_reg)

reg.predict(X_reg)
reg.prob_exceeds(X_reg, threshold=1.0)  # P(MASE > 1)
reg.predict_quantile(X_reg, q=0.9)  # 90th pct of predictive distribution
reg.predict_cdf(X_reg, y_grid=np.linspace(0, 3, 50))
