import pandas as pd

from src.config import DATASET_MAPPING

MODELS = ['MLP', 'NHITS', 'PatchTST']
results_dir = './assets/results_cv'

# ---- classifier

auc_dfs = []
for model in MODELS:
    df = pd.read_csv(f'{results_dir}/cv_clf_scores_{model}.csv', index_col='dataset')
    auc_dfs.append(df['auc'].rename(model))

table = pd.concat(auc_dfs, axis=1)
table = table.drop('std')
table = table.rename(index=DATASET_MAPPING)
print(table)

print(table.to_latex(caption='cap', label='lab:tab_auc', float_format='%.3f'))

# ---- regression

# todo
