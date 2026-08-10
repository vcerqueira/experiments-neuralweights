"""Deterministic random sampling from Ray Tune-style config pools."""
import random
import hashlib
import json
from typing import Dict

import numpy as np
import pandas as pd

from src.config import SEED, N_SAMPLES


class ConfigSampler:
    """Sample hyperparameter configs and assign stable ``config_id`` hashes.

    Example:
        >>> from src.neural.config_pool import NEURAL_CONFIG_POOL
        >>> samples = ConfigSampler.generate_samples(NEURAL_CONFIG_POOL['MLP'], num_samples=10)
    """

    BAD_CONFIGS = []

    @classmethod
    def generate_samples(cls,
                         config_pool: Dict,
                         num_samples: int = N_SAMPLES,
                         random_state: int = SEED,
                         remove_bad_configs: bool = True,
                         return_df: bool = False):
        """Draw ``num_samples`` uninformed random configs from ``config_pool``.

        Args:
            config_pool: Mapping of param name -> Ray Tune sampleable / constant.
            num_samples: Number of configs to draw.
            random_state: Seed for reproducibility.
            remove_bad_configs: Drop entries listed in ``BAD_CONFIGS``.
            return_df: If True, return a DataFrame indexed by ``config_id``.

        Returns:
            List of config dicts (each with ``config_id``), or a DataFrame.
        """
        cls.set_seeds(random_state)

        sample_list = []
        for i in range(num_samples):
            sample = {
                k: (v.sample() if hasattr(v, 'sample') else v)
                for k, v in config_pool.items()
            }

            sample['config_id'] = cls.get_config_id(sample)

            # if sample['batch_size'] > 32:
            #     continue

            sample_list.append(sample)

        if remove_bad_configs:
            sample_list = [sample for sample in sample_list if sample['config_id'] not in cls.BAD_CONFIGS]

        if return_df:
            df = pd.DataFrame(sample_list).set_index('config_id')
            return df

        return sample_list

    @staticmethod
    def set_seeds(seed: int = SEED):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_config_id(config):
        hash_len = 20

        config_str = json.dumps(config, sort_keys=True)
        config_id = hashlib.md5(config_str.encode()).hexdigest()[:hash_len]

        return config_id
