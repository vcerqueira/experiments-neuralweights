import pandas as pd

MODELS = ['MLP', 'NHITS', 'PatchTST']
results_dir = './assets/results_cv'

DATASET_MAPPING = {
    'monash_hospital': 'Hospital',
    'monash_m1_monthly': 'M1-M',
    'monash_m1_quarterly': 'M1-Q',
    'monash_m3_monthly': 'M3-M',
    'monash_m3_quarterly': 'M3-Q',
    'monash_tourism_monthly': 'T-M',
    'monash_tourism_quarterly': 'T-Q',
    'average': 'Average',
}

auc_dfs = []
for model in MODELS:
    df = pd.read_csv(f'{results_dir}/cv_clf_scores_{model}.csv', index_col='dataset')
    auc_dfs.append(df['auc'].rename(model))

table = pd.concat(auc_dfs, axis=1)
table = table.drop('std')
table = table.rename(index=DATASET_MAPPING)
print(table)

print(table.to_latex(caption='cap',
                     label='lab:tab_auc',
                     float_format='%.3f'))
