from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


class CatBoostAUCClassifier:
    """Binary CatBoost classifier with optional Optuna tuning on holdout AUC."""

    def __init__(
            self,
            *,
            optimize: bool = False,
            n_trials: int = 50,
            val_size: float = 0.2,
            random_state: int = 42,
            early_stopping_rounds: int = 50,
            catboost_params: Optional[dict[str, Any]] = None,
            optuna_seed: Optional[int] = None,
    ):
        self.optimize = optimize
        self.n_trials = n_trials
        self.val_size = val_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.catboost_params = catboost_params or {}
        self.optuna_seed = random_state if optuna_seed is None else optuna_seed

        self.model_: Optional[CatBoostClassifier] = None
        self.best_params_: dict[str, Any] = {}
        self.best_auc_: Optional[float] = None

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> "CatBoostAUCClassifier":
        X_input, y_arr = self._validate_xy(X, y)

        if self.optimize:
            self.best_params_ = self._optimize_params(X_input, y_arr, cat_features=cat_features)
        else:
            self.best_params_ = self._default_params()

        self.model_ = CatBoostClassifier(**self.best_params_)
        self.model_.fit(
            X_input,
            y_arr,
            cat_features=cat_features,
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(self._validate_x(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(self._validate_x(X))

    def _optimize_params(
            self,
            X: ArrayLike,
            y: np.ndarray,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> dict[str, Any]:
        X_train, X_val, y_train, y_val = self._holdout_split(X, y)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
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

        self.best_auc_ = study.best_value
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

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )

        y_score = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, y_score))

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
    ) -> tuple[ArrayLike, ArrayLike, np.ndarray, np.ndarray]:
        return train_test_split(
            X,
            y,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y,
        )

    def _default_params(self) -> dict[str, Any]:
        return {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'allow_writing_files': False,
            'depth': 9,
            'learning_rate': 0.21557103267404404,
            'l2_leaf_reg': 2.60075265467506,
            'bagging_temperature': 0.17861251886208518,
            'random_strength': 0.40655745710716007,
            'border_count': 107,
            'subsample': 0.7510803400616529,
            'iterations': 931,
            "random_seed": self.random_state,
            **self.catboost_params,
        }

    @staticmethod
    def _validate_xy(X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, np.ndarray]:
        y_arr = np.asarray(y).ravel()
        if y_arr.ndim != 1:
            raise ValueError("`y` must be a 1d array-like object.")
        n_rows = CatBoostAUCClassifier._row_count(X)
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
            raise RuntimeError("Call fit before predict or predict_proba.")
