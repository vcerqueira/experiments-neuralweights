import numpy as np
import pandas as pd
import optuna
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_auto import OptunaOptions

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Fake data ---
n_series, n_dates = 3, 120
dates = pd.date_range("2020-01-01", periods=n_dates, freq="ME")
df = pd.DataFrame({
    "unique_id": np.repeat(["s1", "s2", "s3"], n_dates),
    "ds": np.tile(dates, n_series),
    "y": np.random.randn(n_series * n_dates) * 10 + 50,
})

# --- Custom search space ---
def config_mlp(trial):
    return {
        "max_steps": 10,
        "input_size": 24,
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "val_check_steps": 50,
        "random_seed": trial.suggest_int("random_seed", 1, 10),
    }

# --- AutoMLP with TPE + MedianPruner ---
model = AutoMLP(
    h=12,
    loss=MAE(),
    config=config_mlp,
    search_alg=optuna.samplers.TPESampler(seed=0),
    backend="optuna",
    num_samples=10,
    optuna_options=OptunaOptions(
        create_study_kwargs={"pruner": optuna.pruners.MedianPruner()}
    ),
)


# --- Fit and predict ---
nf = NeuralForecast(models=[model], freq="ME")
nf.fit(df=df, val_size=24)

forecasts = nf.predict()
print(forecasts.head())

# --- Inspect tuning results ---
results = nf.models[0].results.trials_dataframe()
print(results.drop(columns="user_attrs_ALL_PARAMS"))