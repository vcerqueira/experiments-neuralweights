from pathlib import Path

DRY_RUN = False

# RESULTS_DIR = Path().resolve().parent.parent / 'hypertuning-files' / 'results-all-compiled'
# RESULTS_DIR = Path().resolve().parent / 'results'
RESULTS_DIR = Path().resolve() / 'results'

SEED = 1108
CB_N_STEPS = 10
ENGINE = 'mps'
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
