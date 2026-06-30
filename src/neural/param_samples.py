import random
import hashlib
import json
from typing import Dict

import numpy as np
import pandas as pd

from src.config import SEED, N_SAMPLES


class ConfigSampler:
    # from ray import tune
    #
    # config_space_ = {
    #     "n_pool_kernel_size": tune.choice([2, 3, 5]),
    #     "learning_rate": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([16, 32, 64]),
    #     "activation": "relu"
    # }
    #
    # sample_list_ = ConfigSampler.generate_samples(config_space_, num_samples=4)
    #
    # df = pd.DataFrame(sample_list_).set_index('config_id')

    MISC_BAD_CONFIGS = ['fd1d8da7684e79a15d7d',
                        '2b034fe8fd6d9ee361de',
                        '5955c5f639e11e662d78',
                        'aefb0e3b650af6c4f8ea',
                        'b682f1fd0bea1965880f',
                        '856f1172fc2f3108b2eb',
                        '6c6aa783172ef07d645d',
                        'e23b2735de790f35e3b4',
                        '0ae460bbd17da6e08ef9',
                        'be8cf16288da90941ed6',
                        'a36bf9c40a499e62ee5a',
                        'e04ff4877de4e241445d',
                        '09316b60a187415cd7dc',
                        'fd9e39298706aa752095',
                        '4c5aee5fefa6297da54d', '0eeaaea4899c7a6c4174', '37f7ff872091667cd1f0', '240da28a18633e4b5dd5',
                        '5b87b26f02d3730a269d', '1fd7a21a3c318eb289e7', '6d7d7ca2c907280ac965', '52e7ad6bc1aff7967f5e',
                        '05ae47290b9dca4b81e9', '3fa586f3cef025886861', '25d8652d0fc54beacd67', '0aa23c36a3f490eb201d',
                        'b7142a74df76be802340', 'd0939307381a4b0d11b4', '343baaa75d2db517bf79', '29b99429f27f24bd2e13',
                        '6c2f15c6e364957ce015', '25b25cd264864d3b1e23', 'f087f3a809f7256a1e9b', 'f16693060d9038687465',
                        '23f76bf58cc90aa11181', '42f19db28cbd5d17c23a', '51471701b67d9c28a234', 'cdbab8bf3126bbefcc0f',
                        '181aa3526fbe024da3cc', 'bdd3a37a4eb17d1292b8', '293a811bfdd26a2321f8', '865ef649a51e520e4291',
                        '402d5cf19c1e3439922e', '96c361c80ec43d9b5e86', 'beae40907062bb1da545', 'cb859d4d44408796a5b1'

                        ]

    ECL_BAD_IDS = ['35f1273d18fb12279de6', '3c6dbaa91b91d440b9ea', '37ff49dd45d4f53513e0', '9a0a1887d8e83656ec3b',
                   '82807f1f5be20c1d0ec0', 'a9c3a2b6762c4586fe67', '60c702f9cba673707b66', '3acc74f72e6ff19edea9',
                   'a99aec9a19f241c8fef9', '9a3d0f890864e2b62d19', '959f0832fa6bfa3b0cf3', 'ad0c8986174fd8c0b53d',
                   '6a5d9c886cf892cac3e7', '29b99429f27f24bd2e13', 'ff0a805318367cbc8229', '1fd7a21a3c318eb289e7',
                   'e773d7613b612a1e0108', 'c0122bd5db811cc8ba3f', 'bdd3a37a4eb17d1292b8', '0aa23c36a3f490eb201d',
                   'f087f3a809f7256a1e9b', '42f19db28cbd5d17c23a', '293a811bfdd26a2321f8', '3fdbc68e6b4f6f056a09',
                   '25d8652d0fc54beacd67', '2c0ce73dc8e135c1d657']

    TOURISM_M_BAD_IDS = ['37ff49dd45d4f53513e0']

    BAD_CONFIGS = MISC_BAD_CONFIGS + ECL_BAD_IDS + TOURISM_M_BAD_IDS

    @classmethod
    def generate_samples(cls,
                         config_pool: Dict,
                         num_samples: int = N_SAMPLES,
                         random_state: int = SEED,
                         remove_bad_configs: bool = True,
                         return_df: bool = False):

        """
        Uninformed Random Sampling
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
