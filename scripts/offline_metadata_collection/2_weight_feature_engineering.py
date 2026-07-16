from pathlib import Path

from src.utils import read_all_metadata
from src.weights.weight_summarizer import NeuralWeightsFeatureEng

model_name = 'NHITS'
data_dir = Path('../neuralweights-files/results')
print(data_dir.absolute())

metadata = read_all_metadata(data_dir, model_name, detailed=True)

# metadata_grouped = metadata.groupby(['dataset', 'config_id', 'model'])
## example group
# idx = metadata_grouped.groups[('monash_m1_monthly', '00398d6088206dd39e4e', 'MLP')]
# df = metadata.loc[idx,]

metadata_smr = NeuralWeightsFeatureEng.summarise_detail_df(metadata, model=model_name)
metadata_smr['step'] = metadata_smr['step'].astype(int)

metadata_smr.to_csv(f'assets/metadata_{model_name}.csv', index=False)
