from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.loaders import ChronosDataset, LongHorizonDatasetR


@dataclass
class MetaLearningData:
    """Container for meta-learning X, y pairs and related data."""
    X: pd.DataFrame
    y: np.ndarray
    groups: pd.Series
    feature_columns: list[str]
    task: Literal["classification", "regression"]
    mase_sn_by_dataset: Optional[pd.Series] = None


def build_meta_xy(
        metadata: pd.DataFrame,
        task: Literal["classification", "regression"] = "classification",
        *,
        use_step_as_feature: bool = False,
        performance_diff: bool = True,
        y_clip: Optional[tuple[float, float]] = None,
        drop_columns: Optional[list[str]] = None,
) -> MetaLearningData:
    """Build X, y pairs for meta-learning from metadata.
    
    Args:
        metadata: DataFrame with columns including 'mase', 'mase_sn', 'dataset', etc.
        task: Either "classification" (predict if better than baseline) or 
              "regression" (predict performance or performance difference).
        use_step_as_feature: If True, include 'step' column as a feature.
        performance_diff: For regression, if True predict (mase_sn - mase),
                         otherwise predict mase directly.
        y_clip: For regression, clip y values to (min, max). E.g., (-2.5, 2.5).
        drop_columns: Additional columns to drop from features.
    
    Returns:
        MetaLearningData with X, y, groups, feature_columns, and mase_sn_by_dataset.
    
    Examples:
        # Classification: predict if model beats baseline
        data = build_meta_xy(metadata, task="classification")
        clf.fit(data.X, data.y)
        
        # Regression: predict performance difference
        data = build_meta_xy(metadata, task="regression", y_clip=(-2.5, 2.5))
        reg.fit(data.X, data.y)
        
        # With step as feature
        data = build_meta_xy(metadata, task="classification", use_step_as_feature=True)
    """
    metadata = metadata.copy()

    base_drop_cols = ['mase', 'mase_sn', 'has_esd', 'model', 'config_id', 'dataset']
    if not use_step_as_feature:
        base_drop_cols.append('step')

    if drop_columns:
        base_drop_cols.extend(drop_columns)

    base_drop_cols = [c for c in base_drop_cols if c in metadata.columns]

    groups = metadata['dataset'].copy()
    mase_sn_by_dataset = metadata.groupby('dataset')['mase_sn'].first()

    if task == "classification":
        y = (metadata['mase'] > metadata['mase_sn']).astype(int)
    elif task == "regression":
        if performance_diff:
            y = (metadata['mase_sn'] - metadata['mase'])
        else:
            y = metadata['mase']

        if y_clip is not None:
            y = np.clip(y, a_min=y_clip[0], a_max=y_clip[1])
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'regression'.")

    X = metadata.drop(columns=base_drop_cols)
    feature_columns = X.columns.tolist()

    return MetaLearningData(
        X=X,
        y=y,
        groups=groups,
        feature_columns=feature_columns,
        task=task,
        mase_sn_by_dataset=mase_sn_by_dataset,
    )


def encode_cats(
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Encode object/categorical columns as integer codes.
    
    Args:
        df: DataFrame to encode.
        columns: Specific columns to encode. If None, encodes all object columns.
    
    Returns:
        Tuple of (encoded DataFrame, category mappings dict).
        The mappings dict maps column -> {value -> code}.
    
    Example:
        df_encoded, mappings = encode_categorical_columns(df)
        # mappings = {'col1': {'a': 0, 'b': 1}, 'col2': {'x': 0, 'y': 1}}
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    category_mappings: dict[str, dict[str, int]] = {}

    for col in columns:
        if col in df.columns:
            if col not in ['dataset', 'model', 'config_id']:
                cat_type = df[col].astype('category')
                category_mappings[col] = {v: i for i, v in enumerate(cat_type.cat.categories)}
                df[col] = cat_type.cat.codes

    return df, category_mappings


def load_dataset_splits(target, get_valid: bool = False):
    if target in ChronosDataset.FREQUENCY_MAP_DATASETS:
        df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)

    else:
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target, resample_to='D'
        )

    df = ChronosDataset.prune_uids_by_size(df, min_n_instances=2 * (n_lags + horizon))
    train_full, test = ChronosDataset.time_wise_split(df, horizon)

    if get_valid:
        train_in, valid = ChronosDataset.time_wise_split(train_full, horizon)
    else:
        train_in = train_full
        valid = pd.DataFrame()

    train_in = ChronosDataset.prune_uids_by_size(train_in, min_n_instances=2 * (n_lags + horizon))
    train_full = ChronosDataset.prune_uids_by_size(train_full, min_n_instances=2 * (n_lags + horizon))

    return train_full, train_in, valid, test, horizon, n_lags, freq, seas_len


class MetadataReader:
    """Read experiment metadata from individual CSVs or a single processed file.
    
    Supports two modes:
    - Individual files: `{model},{dataset},{config_id},{cbs|cbd}.csv`
    - Processed file: `metadata_{model}.csv` (pre-aggregated)
    """

    def __init__(
            self,
            data_dir,
            model: str,
            detailed: bool = False,
            processed_file: str | Path | None = None,
    ):
        """Initialize MetadataReader.
        
        Args:
            data_dir: Directory containing metadata files.
            model: Model name (e.g., 'MLP').
            detailed: If True, read detailed (cbd) files; otherwise summary (cbs).
            processed_file: Path to a pre-processed metadata CSV. If provided,
                reads from this file instead of individual config files.
        """
        self.data_dir = Path(data_dir)
        self.model = model
        self.detailed = detailed
        self.processed_file = Path(processed_file) if processed_file else None
        self._cached_df: pd.DataFrame | None = None

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
        if self.processed_file is not None:
            df = self._read_processed()
            if 'dataset' in df.columns:
                return sorted(df['dataset'].unique().tolist())
            return []

        datasets = {
            dataset
            for path in self.data_dir.glob(self.glob_pattern())
            if (dataset := self.parse_dataset(path, self.model)) is not None
        }
        return sorted(datasets)

    def _read_processed(self) -> pd.DataFrame:
        """Read and cache the processed metadata file."""
        if self._cached_df is None:
            if self.processed_file and self.processed_file.exists():
                self._cached_df = pd.read_csv(self.processed_file)
            else:
                self._cached_df = pd.DataFrame()
        return self._cached_df

    def read(self, dataset_name: str) -> pd.DataFrame:
        """Read metadata for a single dataset."""
        if self.processed_file is not None:
            df = self._read_processed()
            if 'dataset' in df.columns:
                return df[df['dataset'] == dataset_name].reset_index(drop=True)
            return pd.DataFrame()

        config_files = list(self.data_dir.glob(self.glob_pattern(dataset_name)))
        if not config_files:
            return pd.DataFrame()

        metadata = pd.concat([pd.read_csv(f) for f in config_files]).reset_index(drop=True)
        return metadata

    def read_all(
            self,
            sample_n: int | None = None,
            sample_frac: float | None = None,
            random_state: int = 42,
    ) -> pd.DataFrame:
        """Read metadata for every available dataset.
        
        Args:
            sample_n: If provided, randomly sample this many rows.
            sample_frac: If provided, randomly sample this fraction of rows.
            random_state: Random seed for sampling.
        
        Returns:
            DataFrame with metadata, optionally sampled.
        """
        if self.processed_file is not None:
            df = self._read_processed().copy()
        else:
            dataset_names = self.list_datasets()
            if not dataset_names:
                return pd.DataFrame()
            df = pd.concat(
                [self.read(dataset_name) for dataset_name in dataset_names],
                ignore_index=True,
            )

        if sample_n is not None:
            sample_n = min(sample_n, len(df))
            df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
        elif sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

        return df

    def clear_cache(self):
        """Clear the cached processed DataFrame."""
        self._cached_df = None


def list_metadata_datasets(
        data_dir,
        model,
        detailed=False,
        processed_file=None,
) -> list[str]:
    return MetadataReader(data_dir, model, detailed, processed_file).list_datasets()


def read_metadata(
        data_dir,
        model,
        dataset_name,
        detailed=False,
        processed_file=None,
):
    return MetadataReader(data_dir, model, detailed, processed_file).read(dataset_name)


def read_all_metadata(
        data_dir,
        model,
        detailed=False,
        processed_file=None,
        sample_n=None,
        sample_frac=None,
        random_state=42,
):
    """Read all metadata with optional sampling.
    
    Args:
        data_dir: Directory containing metadata files.
        model: Model name.
        detailed: If True, read detailed files.
        processed_file: Path to pre-processed CSV (e.g., 'metadata_MLP.csv').
        sample_n: Sample this many rows.
        sample_frac: Sample this fraction of rows.
        random_state: Random seed.
    
    Examples:
        # Read from individual files
        df = read_all_metadata('./assets/results', 'MLP')
        
        # Read from processed file
        df = read_all_metadata('./assets', 'MLP', processed_file='./assets/metadata_MLP.csv')
        
        # Read with sampling
        df = read_all_metadata('./assets', 'MLP', processed_file='./assets/metadata_MLP.csv', sample_n=50000)
    """

    BAD_COLS = [
        'log_norm.1',
        'input_size_multiplier',
        'scaler_type',
        'batch_size',
        'windows_batch_size',
        'random_seed',
        'start_padding_enabled',
    ]

    df = MetadataReader(data_dir, model, detailed, processed_file).read_all(
        sample_n=sample_n,
        sample_frac=sample_frac,
        random_state=random_state,
    )

    df = df.drop(columns=BAD_COLS)

    df, _ = encode_cats(df)

    return df


def corr_coef(y_true, y_pred, method='spearman'):
    cc = pd.DataFrame({'t': y_true, 'p': y_pred}).corr(method).values[0, 1]

    return cc
