"""Global experiment settings shared across scripts.

Attributes:
    DRY_RUN: If True, use tiny sample sizes / epoch limits for debugging.
    RESULTS_DIR: Default directory for miscellaneous outputs.
    SEED: Global RNG seed for config sampling and reproducibility.
    CB_N_STEPS: Default WeightWatcher / Weightcast check interval (optimizer steps).
    ENGINE: Lightning accelerator (`'mps'`, `'gpu'`, or `'cpu'`).
    N_SAMPLES: Number of hyperparameter configs to sample from each pool.
    MAX_SAMPLES: Cap used in metadata collection loops.
    DATASET_MAPPING: Short display names for tables and plots.
"""
from pathlib import Path

DRY_RUN = False

RESULTS_DIR = Path().resolve() / 'results'

SEED = 1108
CB_N_STEPS = 10
ENGINE = 'mps'  # 'gpu' for CUDA
if DRY_RUN:
    LIMIT_EPOCHS = True
    N_SAMPLES = 100
    MAX_SAMPLES = 50
else:
    LIMIT_EPOCHS = False
    N_SAMPLES = 3000
    MAX_SAMPLES = 500

DATASET_MAPPING = {
    'monash_hospital': 'Hospital',
    'monash_m1_monthly': 'M1-M',
    'monash_m1_quarterly': 'M1-Q',
    'monash_m3_monthly': 'M3-M',
    'monash_m3_quarterly': 'M3-Q',
    'monash_tourism_monthly': 'T-M',
    'monash_tourism_quarterly': 'T-Q',
    'average': 'Average',
}
