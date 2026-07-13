import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('./assets/results_search/search_MLP.csv')

df.iloc[1]

# table with final performance by model and mode

# grouped barplots with

# AUC by model
df.groupby(['dataset','model']).mean(numeric_only=True).mean()

# % of rejected configs by model

# tot training steps by model, as % of the total steps planned

