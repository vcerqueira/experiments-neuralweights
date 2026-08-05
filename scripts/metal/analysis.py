import pandas as pd

from src.config import DATASET_MAPPING

MODELS = ['MLP', 'NHITS', 'PatchTST']
results_dir = './assets/results_cv'

# ---- classifier

auc_dfs = []
for model in MODELS:
    df = pd.read_csv(f'{results_dir}/cv_clf_scores_{model}.csv', index_col='dataset')
    auc_dfs.append(df['auc'].rename(model))

table_clf = pd.concat(auc_dfs, axis=1)
table_clf = table_clf.drop('std')
table_clf = table_clf.rename(index=DATASET_MAPPING)
print(table_clf)

print(table_clf.to_latex(caption='Classifier AUC by model (LOO-CV)', label='tab:clf_auc', float_format='%.3f'))

# ---- regression

REG_METRICS = {
    'auc_exc': 'AUC',
    'nmae': 'nMAE',
    'kendall': 'Kendall',
    'spearman': 'Spearman',
    'r2': 'R2',
    'll_iso': 'Log Loss',
    'brier_iso': 'Brier',
}

for metric_col, metric_name in REG_METRICS.items():
    metric_dfs = []
    for model in MODELS:
        df = pd.read_csv(f'{results_dir}/cv_reg_scores_{model}.csv', index_col='dataset')
        metric_dfs.append(df[metric_col].rename(model))

    table_reg = pd.concat(metric_dfs, axis=1)
    table_reg = table_reg.drop('std', errors='ignore')
    table_reg = table_reg.rename(index=DATASET_MAPPING)

    print(f"\n{metric_name}:")
    print(table_reg)

# KEY_METRICS = ['auc_exc', 'nmae', 'kendall']
KEY_METRICS = [*REG_METRICS]

combined_rows = []
for model in MODELS:
    df = pd.read_csv(f'{results_dir}/cv_reg_scores_{model}.csv', index_col='dataset')
    df = df.drop('std', errors='ignore')
    df = df.rename(index=DATASET_MAPPING)
    for metric in KEY_METRICS:
        combined_rows.append(df[metric].rename((model, REG_METRICS[metric])))

table_combined = pd.concat(combined_rows, axis=1)
table_combined.columns = pd.MultiIndex.from_tuples(table_combined.columns)
print(table_combined)

print(table_combined.to_latex(
    caption='Regressor performance by model (LOO-CV)',
    label='tab:reg_metrics',
    float_format='%.3f',
    multicolumn=True,
    multicolumn_format='c',
))
