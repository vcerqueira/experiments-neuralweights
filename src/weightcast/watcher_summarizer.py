import math
import pandas as pd

from src.neural.param_samples import ConfigSampler
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.config import N_SAMPLES, SEED


class NeuralWeightsFeatureEng:
    stats_metrics = [
        "log_norm",
        "alpha",
        "alpha_weighted",
        "lambda_max",
        "log_alpha_norm",
        "log_spectral_norm",
        "norm",
        "stable_rank",
        # "mp_softrank",
        "entropy",
        "has_esd",
        "num_pl_spikes",
        "rank_loss",
        "sigma",
        "sv_max",
        "sv_min",
        "xmax",
        "xmin",
        "D",
        "is_undertrained",
    ]

    get_first_metrics = [
        "config_id",
        "step",
    ]

    @classmethod
    def snapshop_detail_stats(cls, df, add_performance: bool=True):
        df = cls.pre_feature_engineering(df)

        stats = df[cls.stats_metrics].mean()
        # stats['config_id'] = df['config_id'].values[0]
        # stats['step'] = df['step'].values[0]
        if add_performance:
            stats['mase'] = df['mase'].values[0]
            stats['mase_sn'] = df['mase_sn'].values[0]

        # df[stats_metrics].std()
        return stats

    @staticmethod
    def pre_feature_engineering(df):
        df['is_undertrained'] = (df['warning'] == 'under-trained').astype(int)

        return df

    @staticmethod
    def bin_learning_rate(lr):
        LR_LOG_CENTERS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
        LR_LABELS = [f'1e{exp:g}' for exp in LR_LOG_CENTERS]

        center_idx = min(
            range(len(LR_LOG_CENTERS)),
            key=lambda i: abs(math.log10(lr) - LR_LOG_CENTERS[i]),
        )
        return LR_LABELS[center_idx]

    @classmethod
    def get_config_data(cls, df, model):
        config_pool = NEURAL_CONFIG_POOL[model]
        config_df = ConfigSampler.generate_samples(config_pool=config_pool,
                                                   num_samples=N_SAMPLES,
                                                   random_state=SEED,
                                                   return_df=True)

        df = df.merge(config_df, how='left', on='config_id')
        df['start_padding_enabled'] = df['start_padding_enabled'].astype(bool).astype(int)
        df['learning_rate'] = df['learning_rate'].map(cls.bin_learning_rate)

        return df

    @classmethod
    def summarise_detail_df(cls, details: pd.DataFrame, model):
        df_grouped = details.groupby(['dataset', 'config_id', 'model', 'step'])

        df_smr = df_grouped.apply(cls.snapshop_detail_stats)
        df_smr = df_smr.reset_index()

        df_smr = cls.get_config_data(df_smr, model)

        return df_smr
