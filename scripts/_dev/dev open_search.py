import warnings
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoPatchTST

from neuralforecast.losses.pytorch import MAE
from utilsforecast.losses import mase

from src.early_stopping import ClassifierEarlyStopCallback
from src.search import train_meta_classifier
from src.utils import read_all_metadata, load_dataset_splits

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

STOPPING_THRESHOLD = 0.70
N_TRIALS = 2
CB_N_STEPS = 100
MIN_CB_N_STEPS = 50
MODEL_NAME = 'MLP'
OUTPUT_DIR = Path('./assets/results_search')

AUTO_MODEL_CLASSES = {
    'MLP': AutoMLP,
    'NHITS': AutoNHITS,
    'PatchTST': AutoPatchTST,
}


def make_config_fn(
        model_name: str,
        meta_classifier,
        feature_columns: list[str],
        category_mappings: dict,
        input_size: int,
        stopping_threshold: float = 0.70,
        cb_n_steps: int = 100,
        min_steps: int = 50,
        max_steps: int = 1500,
        verbose: bool = True,
):
    """Factory that creates a config function with meta-model callback injection.
    
    The returned function is called by Optuna for each trial. It samples
    hyperparameters and creates a ClassifierEarlyStopCallback with access
    to the sampled config.
    """

    def config_mlp(trial):
        config = {
            "input_size": trial.suggest_categorical("input_size", [input_size, input_size * 2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 1024]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "standard"]),
            "max_steps": trial.suggest_categorical("max_steps", [500, 1000]),
            "start_padding_enabled": trial.suggest_categorical("start_padding_enabled", [True, False]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

        callback = ClassifierEarlyStopCallback(
            meta_classifier=meta_classifier,
            feature_columns=feature_columns,
            config_data=_prepare_config_data(config, model_name),
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            every_n_steps=cb_n_steps,
            min_steps=min_steps,
            verbose=verbose,
        )
        config["callbacks"] = [callback]
        return config

    def config_nhits(trial):
        config = {
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

        callback = ClassifierEarlyStopCallback(
            meta_classifier=meta_classifier,
            feature_columns=feature_columns,
            config_data=_prepare_config_data(config, model_name),
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            every_n_steps=cb_n_steps,
            min_steps=min_steps,
            verbose=verbose,
        )
        config["callbacks"] = [callback]
        return config

    def config_patchtst(trial):
        config = {
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
            "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }

        callback = ClassifierEarlyStopCallback(
            meta_classifier=meta_classifier,
            feature_columns=feature_columns,
            config_data=_prepare_config_data(config, model_name),
            category_mappings=category_mappings,
            stopping_threshold=stopping_threshold,
            every_n_steps=cb_n_steps,
            min_steps=min_steps,
            verbose=verbose,
        )
        config["callbacks"] = [callback]
        return config

    config_fns = {
        'MLP': config_mlp,
        'NHITS': config_nhits,
        'PatchTST': config_patchtst,
    }

    return config_fns[model_name]


def _prepare_config_data(config: dict, model_name: str) -> dict:
    """Prepare config_data dict for the callback with expected field names."""
    config_data = config.copy()
    config_data.pop('callbacks', None)

    input_size = config_data.pop('input_size', None)
    if input_size is not None:
        config_data['input_size_multiplier'] = 1

    config_data['model'] = model_name

    return config_data


print("Loading metadata...")
metadata, category_mappings = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
)

all_datasets = sorted(metadata['dataset'].unique().tolist())
# all_datasets = [all_datasets[2]]

all_search_results = []
all_test_results = []
for i, target_dataset in enumerate(all_datasets):
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(all_datasets)}] TARGET DATASET: {target_dataset}")
    print("=" * 70)

    train_full, train, valid, test, horizon, n_lags, freq, seas_len = load_dataset_splits(
        target_dataset, get_valid=True
    )

    meta_train = metadata[metadata['dataset'] != target_dataset].reset_index(drop=True)
    meta_classifier, clf_feature_columns = train_meta_classifier(meta_train, calibrate=True)

    mase_func = partial(mase, seasonality=seas_len)

    config_fn = make_config_fn(
        model_name=MODEL_NAME,
        meta_classifier=meta_classifier,
        feature_columns=clf_feature_columns,
        category_mappings=category_mappings,
        input_size=n_lags,
        stopping_threshold=STOPPING_THRESHOLD,
        cb_n_steps=CB_N_STEPS,
        min_steps=MIN_CB_N_STEPS,
        verbose=True,
    )

    AutoModelClass = AUTO_MODEL_CLASSES[MODEL_NAME]

    auto_model = AutoModelClass(
        h=horizon,
        loss=MAE(),
        config=config_fn,
        search_alg=optuna.samplers.TPESampler(seed=42),
        backend="optuna",
        num_samples=N_TRIALS,
        refit_with_val=True,
    )

    nf = NeuralForecast(models=[auto_model], freq=freq)
    nf.fit(df=train, val_size=horizon)

    trials_df = nf.models[0].results.trials_dataframe()
    print(trials_df)

    fcst = nf.predict()
    fcst['ds'] = test['ds']

    alias = AUTO_MODEL_CLASSES[MODEL_NAME].__name__

    holdout = test.merge(fcst, how='left', on=['unique_id', 'ds'])
    test_mase = mase_func(holdout, models=[alias], train_df=train_full)
    test_mase_value = float(test_mase[alias].mean())

    test_results = {
        'dataset': target_dataset,
        f'{MODEL_NAME}_test_mase': test_mase_value,
        'n_trials': N_TRIALS,
    }

    all_test_results.append(test_results)

    trials_df['dataset'] = target_dataset
    all_search_results.append(trials_df)

    print(f"\nTest MASE: {test_mase_value:.4f}")

all_search_df = pd.concat(all_search_results, ignore_index=True)
all_test_df = pd.DataFrame(all_test_results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
search_path = OUTPUT_DIR / f"open_search_{MODEL_NAME}.csv"
test_path = OUTPUT_DIR / f"open_test_{MODEL_NAME}.csv"
all_search_df.to_csv(search_path, index=False)
all_test_df.to_csv(test_path, index=False)

print(f"\nResults saved to {search_path} and {test_path}")
