"""Hyperparameter search spaces for NeuralForecast architectures.

``NEURAL_CONFIG_POOL`` uses Ray Tune distributions for offline random sampling
(``ConfigSampler``). ``CONFIG_SAMPLERS`` exposes Optuna-compatible callables for
AutoModel open search.
"""
from typing import Callable

from ray import tune
import optuna

NEURAL_CONFIG_POOL = {
    'NHITS': {
        "input_size_multiplier": tune.choice([1, 2]),
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [3, 2, 1],  #
                [6, 2, 1],  #
                [8, 4, 1],
                3 * [1],
                3 * [2],
                3 * [4],
                [24, 8, 2],  #
                [12, 6, 3],  #
                [16, 8, 1]
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [6, 2, 1],  #
                [24, 8, 2],  #
                [1, 1, 1],
            ]
        ),
        "mlp_units": tune.choice(
            [
                3 * [[64, 64]],
                3 * [[64, 64, 64]],
                3 * [[128, 128]],
                3 * [[128, 128, 128]],
                3 * [[256, 256]],
                3 * [[256, 256, 256]],
                3 * [[512, 512]],
                3 * [[512, 512, 512]],
            ]
        ),

        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None,
                                    "robust",
                                    "revin",
                                    "standard"]),
        "max_steps": tune.quniform(lower=500,
                                   # upper=1500,
                                   upper=2000,
                                   q=100),
        "pooling_mode": tune.choice(['MaxPool1d', 'AvgPool1d']),
        "interpolation_mode": tune.choice(['linear', 'nearest', 'cubic']),
        "start_padding_enabled": tune.choice([True, False]),
        "dropout_prob_theta": tune.choice([0.0, 0.1, 0.2]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        # "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    },

    'PatchTST': {
        "input_size_multiplier": tune.choice([1, 2, 3]),
        "hidden_size": tune.choice([16, 32, 128, 256]),
        "linear_hidden_size": tune.choice([64, 128, 256]),
        "n_heads": tune.choice([2, 4, 8, 16]),
        "encoder_layers": tune.choice([1, 2, 3]),
        "patch_len": tune.choice([16, 24]),
        "stride": tune.choice([2, 4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "revin": tune.choice([False, True]),
        "max_steps": tune.choice([500, 1000, 2000, 5000]),
        "activation": tune.choice(["relu", "gelu"]),
        "res_attention": tune.choice([True, False]),
        "batch_normalization": tune.choice([True, False]),
        "learn_pos_embed": tune.choice([True, False]),
        "start_padding_enabled": tune.choice([True, False]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
    },

    'MLP': {
        "input_size_multiplier": tune.choice([1, 2, 3]),
        "hidden_size": tune.choice([64, 128, 256, 512, 1024]),
        "num_layers": tune.randint(2, 7),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.choice([500, 1000, 2000, 5000]),
        "start_padding_enabled": tune.choice([True, False]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
    },

    'KAN': {
        "input_size_multiplier": tune.choice([1, 2, 3]),
        "grid_size": tune.choice([5, 10, 15]),
        "spline_order": tune.choice([2, 3, 4]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "scale_noise": tune.choice([0.1, 0.05, 0.2, 0.01]),
        "enable_standalone_scale_spline": tune.choice([True, False]),
        "n_hidden_layers": tune.choice([1, 2, 3, 4]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=2000, q=100),
        "start_padding_enabled": tune.choice([True, False]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(lower=1, upper=20),
    },

    'TFT': {
        "input_size_multiplier": tune.choice([1, 2, 3]),
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "n_head": tune.choice([2, 4, 8]),
        "n_rnn_layers": tune.choice([1, 2]),
        "attn_dropout": tune.choice([0, 0.1, 0.2, 0.3]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.3]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "rnn_type": tune.choice(["lstm", "gru"]),
        "grn_activation": tune.choice(['ELU', 'LeakyReLU', "ReLU", 'Softplus']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "start_padding_enabled": tune.choice([True, False]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
    },

    'Informer': {
        "input_size_multiplier": tune.choice([1, 2, 3]),
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "n_head": tune.choice([2, 4, 8]),
        "conv_hidden_size": tune.choice([8, 16, 32]),
        "encoder_layers": tune.choice([1, 2, 3]),
        "decoder_layers": tune.choice([1, 2]),
        "factor": tune.choice([1, 2, 3]),
        "dropout": tune.choice([0.05, 0.1, 0.15]),
        "distil": tune.choice([True, False]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["robust", "standard"]),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "random_seed": tune.randint(1, 20),
    },

    'GRU': {
        "input_size_multiplier": tune.choice([-1, 1, 2, 3]),
        "inference_input_size_multiplier": tune.choice([-1, 1]),
        "encoder_hidden_size": tune.choice([32, 64, 128, 256, 512]),
        "encoder_n_layers": tune.randint(1, 4),
        "encoder_bias": tune.choice([True, False]),
        "recurrent": tune.choice([True, False]),
        "encoder_dropout": tune.choice([0, 0.1, 0.2, 0.3]),
        "start_padding_enabled": tune.choice([True, False]),
        "context_size": tune.choice([2, 5, 10, 25, 50]),
        "decoder_hidden_size": tune.choice([16, 32, 64, 128, 256]),
        "scaler_type": tune.choice(["robust", "standard"]),
        "decoder_layers": tune.choice([1, 2, 3]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000, 1500, 2000]),
        "batch_size": tune.choice([16, 32, 64]),
        "random_seed": tune.randint(1, 20),
    },

}


def mlp_config_sampler(input_size: int) -> Callable[[optuna.Trial], dict]:
    """Create config sampler for MLP model."""

    def sampler(trial: optuna.Trial) -> dict:
        return {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 1024]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "standard"]),
            "max_steps": trial.suggest_categorical("max_steps", [500, 1000, 2000, 5000]),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

    return sampler


def nhits_config_sampler(input_size: int) -> Callable[[optuna.Trial], dict]:
    """Create config sampler for NHITS model."""

    def sampler(trial: optuna.Trial) -> dict:
        return {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2]),
            "n_pool_kernel_size": trial.suggest_categorical("n_pool_kernel_size", [
                [2, 2, 1], [3, 2, 1], [6, 2, 1], [8, 4, 1],
                [1, 1, 1], [2, 2, 2], [4, 4, 4], [24, 8, 2], [16, 8, 1]
            ]),
            "n_freq_downsample": trial.suggest_categorical("n_freq_downsample", [
                [168, 24, 1], [24, 12, 1], [60, 8, 1], [40, 20, 1],
                [6, 2, 1], [24, 8, 2], [1, 1, 1],
            ]),
            "mlp_units": trial.suggest_categorical("mlp_units", [
                3 * [[64, 64]], 3 * [[128, 128]], 3 * [[256, 256]], 3 * [[512, 512]],
            ]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "revin", "standard"]),
            "max_steps": trial.suggest_int("max_steps", 500, 2000, step=100),
            "pooling_mode": trial.suggest_categorical("pooling_mode", ['MaxPool1d', 'AvgPool1d']),
            "interpolation_mode": trial.suggest_categorical("interpolation_mode", ['linear', 'nearest', 'cubic']),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "dropout_prob_theta": trial.suggest_categorical("dropout_prob_theta", [0.0, 0.1, 0.2]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

    return sampler


def patchtst_config_sampler(input_size: int) -> Callable[[optuna.Trial], dict]:
    """Create config sampler for PatchTST model."""

    def sampler(trial: optuna.Trial) -> dict:
        return {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2, input_size * 3]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 128, 256]),
            "linear_hidden_size": trial.suggest_categorical("linear_hidden_size", [64, 128, 256]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8, 16]),
            "encoder_layers": trial.suggest_categorical("encoder_layers", [1, 2, 3]),
            "patch_len": trial.suggest_categorical("patch_len", [16, 24]),
            "stride": trial.suggest_categorical("stride", [2, 4, 8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "standard"]),
            "revin": trial.suggest_categorical("revin", [False, True]),
            "max_steps": trial.suggest_categorical("max_steps", [500, 1000, 2000, 5000]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            "res_attention": trial.suggest_categorical("res_attention", [True, False]),
            "batch_normalization": trial.suggest_categorical("batch_normalization", [True, False]),
            "learn_pos_embed": trial.suggest_categorical("learn_pos_embed", [True, False]),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

    return sampler


# Optuna config factories: CONFIG_SAMPLERS[model](input_size=n_lags) -> (trial) -> dict
CONFIG_SAMPLERS = {
    'MLP': mlp_config_sampler,
    'NHITS': nhits_config_sampler,
    'PatchTST': patchtst_config_sampler,
}
