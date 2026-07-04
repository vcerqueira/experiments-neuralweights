from pathlib import Path

DRY_RUN = False

# RESULTS_DIR = Path().resolve().parent.parent / 'hypertuning-files' / 'results-all-compiled'
# RESULTS_DIR = Path().resolve().parent / 'results'
RESULTS_DIR = Path().resolve() / 'results'

SEED = 1108
CB_N_STEPS = 10
TRY_MPS = True
if DRY_RUN:
    LIMIT_EPOCHS = True
    N_SAMPLES = 100
    MAX_SAMPLES = 50
else:
    LIMIT_EPOCHS = False
    N_SAMPLES = 3000
    MAX_SAMPLES = 500


DATASETS = [

]
