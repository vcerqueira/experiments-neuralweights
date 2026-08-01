import pandas as pd

model = 'PatchTST'
n_trials = '_30'
df1=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_hospital.csv')
df2=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_m1_monthly.csv')
df3=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_m1_quarterly.csv')
df4=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_m3_monthly.csv')
df5=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_m3_quarterly.csv')
df6=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_tourism_monthly.csv')
df7=pd.read_csv(f'assets/results_search_open/open_test_{model}{n_trials}_monash_tourism_quarterly.csv')

df=pd.concat([df1,df2,df3,df4,df5,df6,df7]).reset_index(drop=True).set_index('dataset')
# df=pd.concat([df1,df2,df3]).reset_index(drop=True).set_index('dataset')
# df=pd.concat([df1,df2]).reset_index(drop=True).set_index('dataset')

dfp=df.loc[:,~df.columns.str.endswith('steps')]
dfs=df.loc[:,df.columns.str.endswith('steps')]


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(dfp.mean().round(4))
print(dfp.round(3))
print(dfp.rank(axis=1).mean())

dfp.round(3)

print(dfs.sum() / (dfs.sum().sum()))
