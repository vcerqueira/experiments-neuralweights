import pandas as pd

from src.loaders import ChronosDataset, LongHorizonDatasetR


def load_dataset_splits(target):
    if target in ChronosDataset.FREQUENCY_MAP_DATASETS:
        df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
        train, test = ChronosDataset.time_wise_split(df, horizon)
    else:
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target, resample_to='D'
        )
        train, test = ChronosDataset.time_wise_split(df, horizon)

    return train, test, horizon, n_lags, freq, seas_len


def read_metadata(data_dir, model, dataset_name, detailed=False):
    data_type = 'cbd' if detailed else 'cbs'

    pattern = f"{model},{dataset_name},*,{data_type}.csv"

    config_files = list(data_dir.glob(pattern))

    metadata = pd.concat([pd.read_csv(f) for f in config_files]).reset_index(drop=True)

    return metadata
