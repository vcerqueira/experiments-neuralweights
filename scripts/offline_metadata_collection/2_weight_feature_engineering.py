from pathlib import Path

from src.utils import read_all_metadata
from src.weights.weight_summarizer import NeuralWeightsFeatureEng

model = 'MLP'
data_dir = Path('./assets/results')

metadata = read_all_metadata(data_dir, model, detailed=True)
# metadata.iloc[0]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# metadata_grouped = metadata.groupby(['dataset', 'config_id', 'model'])
## example group
# idx = metadata_grouped.groups[('monash_m1_monthly', '00398d6088206dd39e4e', 'MLP')]
# df = metadata.loc[idx,]


metadata_smr = NeuralWeightsFeatureEng.summarise_detail_df(metadata, model='MLP')
metadata_smr['step'] = metadata_smr['step'].astype(int)

metadata_smr.to_csv('assets/metadata.csv', index=False)
