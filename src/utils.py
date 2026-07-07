from pathlib import Path

import pandas as pd

from src.loaders import ChronosDataset, LongHorizonDatasetR


def load_dataset_splits(target, get_valid: bool = False):
    if target in ChronosDataset.FREQUENCY_MAP_DATASETS:
        df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)

    else:
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target, resample_to='D'
        )

    df = ChronosDataset.prune_uids_by_size(df, min_n_instances=2 * (n_lags + horizon))
    train, test = ChronosDataset.time_wise_split(df, horizon)

    if get_valid:
        train_in, valid = ChronosDataset.time_wise_split(train, horizon)
    else:
        train_in = train
        valid = pd.DataFrame()

    df = ChronosDataset.prune_uids_by_size(train_in, min_n_instances=2 * (n_lags + horizon))

    return train_in, valid, test, horizon, n_lags, freq, seas_len


class MetadataReader:
    """Read experiment metadata CSVs named `{model},{dataset},{config_id},{cbs|cbd}.csv`."""

    def __init__(self, data_dir, model: str, detailed: bool = False):
        self.data_dir = Path(data_dir)
        self.model = model
        self.detailed = detailed

    @property
    def data_type(self) -> str:
        return 'cbd' if self.detailed else 'cbs'

    def glob_pattern(self, dataset_name: str | None = None) -> str:
        if dataset_name is None:
            return f"{self.model},*,*,{self.data_type}.csv"
        return f"{self.model},{dataset_name},*,{self.data_type}.csv"

    @staticmethod
    def parse_dataset(path: Path, model: str) -> str | None:
        """Extract dataset_name from `{model},{dataset_name},{cfg_id},{cbs|cbd}.csv`."""
        stem_parts = path.stem.split(',')
        if len(stem_parts) != 4:
            return None
        file_model, dataset_name, _cfg_id, data_type = stem_parts
        if file_model != model or data_type not in {'cbs', 'cbd'}:
            return None
        return dataset_name

    def list_datasets(self) -> list[str]:
        """Return sorted unique dataset names available for this model."""
        datasets = {
            dataset
            for path in self.data_dir.glob(self.glob_pattern())
            if (dataset := self.parse_dataset(path, self.model)) is not None
        }
        return sorted(datasets)

    def read(self, dataset_name: str) -> pd.DataFrame:
        """Read metadata for a single dataset."""
        config_files = list(self.data_dir.glob(self.glob_pattern(dataset_name)))
        if not config_files:
            return pd.DataFrame()

        metadata = pd.concat([pd.read_csv(f) for f in config_files]).reset_index(drop=True)
        # metadata['dataset_name'] = dataset_name
        return metadata

    def read_all(self) -> pd.DataFrame:
        """Read metadata for every available dataset."""
        dataset_names = self.list_datasets()
        if not dataset_names:
            return pd.DataFrame()

        return pd.concat(
            [self.read(dataset_name) for dataset_name in dataset_names],
            ignore_index=True,
        )


def list_metadata_datasets(data_dir, model, detailed=False) -> list[str]:
    return MetadataReader(data_dir, model, detailed).list_datasets()


def read_metadata(data_dir, model, dataset_name, detailed=False):
    return MetadataReader(data_dir, model, detailed).read(dataset_name)


def read_all_metadata(data_dir, model, detailed=False):
    return MetadataReader(data_dir, model, detailed).read_all()


def corr_coef(y_true, y_pred, method='spearman'):
    cc = pd.DataFrame({'t': y_true, 'p': y_pred}).corr(method).values[0, 1]

    return cc
