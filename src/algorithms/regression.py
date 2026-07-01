from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


class ConformalPredictiveDistribution:
    """Split conformal predictive distribution from calibration residuals."""

    def __init__(self, residuals: np.ndarray):
        self.residuals_ = np.asarray(residuals, dtype=float).ravel()
        if self.residuals_.size == 0:
            raise ValueError("Calibration residuals must be non-empty.")

    def predictive_samples(self, y_hat: ArrayLike) -> np.ndarray:
        """Return all conformal predictive atoms y_hat + r_j for each point."""
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        return y_arr[:, None] + self.residuals_[None, :]

    def cdf(self, y_hat: ArrayLike, y_grid: ArrayLike) -> np.ndarray:
        """Empirical predictive CDF F(t) = P(Y <= t | X) at each grid point."""
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        grid = np.asarray(y_grid, dtype=float).ravel()
        samples = self.predictive_samples(y_arr)
        n_cal = self.residuals_.size
        return (samples[:, None, :] <= grid[None, :, None]).sum(axis=2) / (n_cal + 1)

    def prob_exceeds(self, y_hat: ArrayLike, threshold: float) -> np.ndarray:
        """P(Y > threshold | X), e.g. P(MASE > 0.13)."""
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        samples = self.predictive_samples(y_arr)
        n_cal = self.residuals_.size
        return (samples > threshold).sum(axis=1) / (n_cal + 1)

    def quantile(self, y_hat: ArrayLike, q: float) -> np.ndarray:
        """Quantile of the conformal predictive distribution."""
        if not 0.0 < q < 1.0:
            raise ValueError("`q` must be between 0 and 1.")
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        samples = np.sort(self.predictive_samples(y_arr), axis=1)
        n_cal = self.residuals_.size
        idx = np.ceil(q * (n_cal + 1) - 1).astype(int)
        idx = np.clip(idx, 0, n_cal - 1)
        row_idx = np.arange(len(y_arr))
        return samples[row_idx, idx]


class CatBoostRegressionModel:
    """CatBoost regressor with optional Optuna tuning and conformal predictive distributions."""

    def __init__(
            self,
            *,
            optimize: bool = False,
            conformal: bool = False,
            n_trials: int = 50,
            val_size: float = 0.2,
            conformal_cal_size: float = 0.2,
            random_state: int = 42,
            early_stopping_rounds: int = 50,
            catboost_params: Optional[dict[str, Any]] = None,
            optuna_seed: Optional[int] = None,
    ):
        self.optimize = optimize
        self.conformal = conformal
        self.n_trials = n_trials
        self.val_size = val_size
        self.conformal_cal_size = conformal_cal_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.catboost_params = catboost_params or {}
        self.optuna_seed = random_state if optuna_seed is None else optuna_seed

        self.model_: Optional[CatBoostRegressor] = None
        self.cpd_: Optional[ConformalPredictiveDistribution] = None
        self.best_params_: dict[str, Any] = {}
        self.best_rmse_: Optional[float] = None

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> "CatBoostRegressionModel":
        X_input, y_arr = self._validate_xy(X, y)

        if self.conformal:
            X_fit, X_cal, y_fit, y_cal = self._holdout_split(
                X_input,
                y_arr,
                test_size=self.conformal_cal_size,
            )
        else:
            X_fit, y_fit = X_input, y_arr
            X_cal = y_cal = None

        if self.optimize:
            self.best_params_ = self._optimize_params(X_fit, y_fit, cat_features=cat_features)
        else:
            self.best_params_ = self._default_params()

        self.model_ = CatBoostRegressor(**self.best_params_)
        self.model_.fit(X_fit, y_fit, cat_features=cat_features)

        if self.conformal:
            assert X_cal is not None and y_cal is not None
            self.cpd_ = self._fit_conformal(X_cal, y_cal, cat_features=cat_features)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(self._validate_x(X))

    def prob_exceeds(self, X: ArrayLike, threshold: float) -> np.ndarray:
        """P(Y > threshold | X) from the conformal predictive distribution."""
        self._check_conformal()
        y_hat = self.predict(X)
        return self.cpd_.prob_exceeds(y_hat, threshold)

    def predict_cdf(self, X: ArrayLike, y_grid: ArrayLike) -> np.ndarray:
        """Predictive CDF evaluated on `y_grid` for each row in X."""
        self._check_conformal()
        y_hat = self.predict(X)
        return self.cpd_.cdf(y_hat, y_grid)

    def predict_quantile(self, X: ArrayLike, q: float) -> np.ndarray:
        """Quantile of the conformal predictive distribution for each row in X."""
        self._check_conformal()
        y_hat = self.predict(X)
        return self.cpd_.quantile(y_hat, q)

    def _fit_conformal(
            self,
            X_cal: ArrayLike,
            y_cal: np.ndarray,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> ConformalPredictiveDistribution:
        y_hat_cal = self.model_.predict(X_cal)
        residuals = y_cal - y_hat_cal
        return ConformalPredictiveDistribution(residuals)

    def _optimize_params(
            self,
            X: ArrayLike,
            y: np.ndarray,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> dict[str, Any]:
        X_train, X_val, y_train, y_val = self._holdout_split(X, y, test_size=self.val_size)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.optuna_seed),
        )
        study.optimize(
            lambda trial: self._objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                cat_features=cat_features,
            ),
            n_trials=self.n_trials,
        )

        self.best_rmse_ = study.best_value
        return {**self._default_params(), **study.best_params}

    def _objective(
            self,
            trial: optuna.Trial,
            X_train: ArrayLike,
            y_train: np.ndarray,
            X_val: ArrayLike,
            y_val: np.ndarray,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> float:
        params = {
            **self._default_params(),
            **self._suggest_params(trial),
            "iterations": trial.suggest_int("iterations", 100, 1000),
        }

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )

        y_pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, y_pred)))

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

    def _holdout_split(
            self,
            X: ArrayLike,
            y: np.ndarray,
            *,
            test_size: float,
    ) -> tuple[ArrayLike, ArrayLike, np.ndarray, np.ndarray]:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
        )

    def _default_params(self) -> dict[str, Any]:
        return {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'verbose': False,
            'allow_writing_files': False,
            'random_seed': 42,
            'depth': 4,
            'learning_rate': 0.29796039491167264,
            'l2_leaf_reg': 1.7551585658181852,
            'bagging_temperature': 0.9102608885378489,
            'random_strength': 8.97550269491597,
            'border_count': 68,
            'subsample': 0.8703838296422213,
            'iterations': 904,
            **self.catboost_params,
        }

    @staticmethod
    def _validate_xy(X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, np.ndarray]:
        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.ndim != 1:
            raise ValueError("`y` must be a 1d array-like object.")
        n_rows = CatBoostRegressionModel._row_count(X)
        if n_rows != len(y_arr):
            raise ValueError("`X` and `y` must have the same number of rows.")
        return X, y_arr

    @staticmethod
    def _validate_x(X: ArrayLike) -> ArrayLike:
        return X

    @staticmethod
    def _row_count(X: ArrayLike) -> int:
        if isinstance(X, pd.DataFrame):
            return len(X)
        return len(np.asarray(X))

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Call fit before predict.")

    def _check_conformal(self) -> None:
        self._check_fitted()
        if self.cpd_ is None:
            raise RuntimeError("Conformal prediction is disabled. Set conformal=True in fit.")
