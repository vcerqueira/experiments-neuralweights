"""NeuralForecast model configuration and hyperparameter sampling.

Modules:
    config_pool: Ray Tune / Optuna search spaces per architecture.
    param_samples: Deterministic random sampling of config pools.
    nf_arch: Factory for NeuralForecast model instances (MLP, NHITS, PatchTST).
"""
