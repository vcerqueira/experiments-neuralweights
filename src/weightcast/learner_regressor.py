"""Regression meta-model for Weightcast with conformal exceedance probabilities."""
from typing import Any, Optional, Union, Literal

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

CalibrationMethod = Literal["isotonic", "platt", "none"]

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


class ConformalPredictiveDistribution:
    """Split conformal predictive distribution from calibration residuals.

    Exceedance/quantile computations avoid materializing the full
    (n_points × n_cal) predictive-atom matrix, which OOMs on large datasets.
    """

    def __init__(self, residuals: np.ndarray):
        self.residuals_ = np.asarray(residuals, dtype=float).ravel()
        if self.residuals_.size == 0:
            raise ValueError("Calibration residuals must be non-empty.")
        self._residuals_sorted = np.sort(self.residuals_)

    def predictive_samples(self, y_hat: ArrayLike) -> np.ndarray:
        """Return all conformal predictive atoms y_hat + r_j for each point.

        Warning: allocates an (n_points × n_cal) array. Prefer prob_exceeds /
        quantile for large n.
        """
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        return y_arr[:, None] + self.residuals_[None, :]

    def cdf(self, y_hat: ArrayLike, y_grid: ArrayLike) -> np.ndarray:
        """Empirical predictive CDF F(t) = P(Y <= t | X) at each grid point."""
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        grid = np.asarray(y_grid, dtype=float).ravel()
        n_cal = self.residuals_.size
        # For each (point, grid_t): count residuals <= t - y_hat
        # cutoffs shape: (n_points, n_grid)
        cutoffs = grid[None, :] - y_arr[:, None]
        n_le = np.searchsorted(self._residuals_sorted, cutoffs, side='right')
        return n_le / (n_cal + 1)

    def prob_exceeds(self, y_hat: ArrayLike, threshold: float) -> np.ndarray:
        """P(Y > threshold | X), e.g. P(MASE > 0.13).

        Uses sorted residuals + binary search: O(n_cal log n_cal + n log n_cal)
        memory/time instead of O(n × n_cal).
        """
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        n_cal = self.residuals_.size
        # y_hat + r > threshold  <=>  r > threshold - y_hat
        cutoffs = threshold - y_arr
        n_le = np.searchsorted(self._residuals_sorted, cutoffs, side='right')
        return (n_cal - n_le) / (n_cal + 1)

    def quantile(self, y_hat: ArrayLike, q: float) -> np.ndarray:
        """Quantile of the conformal predictive distribution."""
        if not 0.0 < q < 1.0:
            raise ValueError("`q` must be between 0 and 1.")
        y_arr = np.asarray(y_hat, dtype=float).ravel()
        n_cal = self.residuals_.size
        idx = np.ceil(q * (n_cal + 1) - 1).astype(int)
        idx = np.clip(idx, 0, n_cal - 1)
        return y_arr + self._residuals_sorted[idx]


class CatBoostRegressionModel:
    """CatBoost regressor with optional Optuna tuning and conformal predictive distributions.

    When ``conformal=True``, a holdout set of residuals forms a split conformal
    predictive distribution. ``prob_exceeds`` then estimates
    P(Y > threshold | X), optionally recalibrated with isotonic or Platt scaling.

    Args:
        optimize: If True, run Optuna HPO before fitting the final model.
        conformal: If True, reserve a calibration set for conformal residuals.
        calibration_method: Default method for ``prob_exceeds``
            (``'isotonic'``, ``'platt'``, or ``'none'``).
        n_trials: Optuna trials when ``optimize=True``.
        val_size: Holdout fraction for Optuna objective evaluation.
        conformal_cal_size: Fraction of data reserved for conformal residuals.
        random_state: RNG seed for splits and CatBoost.
        early_stopping_rounds: CatBoost early stopping during Optuna trials.
        catboost_params: Override / extend default CatBoost hyperparameters.
        optuna_seed: Sampler seed (defaults to ``random_state``).
    """

    def __init__(
            self,
            *,
            optimize: bool = False,
            conformal: bool = False,
            calibration_method: CalibrationMethod = "isotonic",
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
        self.calibration_method = calibration_method
        self.n_trials = n_trials
        self.val_size = val_size
        self.conformal_cal_size = conformal_cal_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.catboost_params = catboost_params or {}
        self.optuna_seed = random_state if optuna_seed is None else optuna_seed

        self.model_: Optional[CatBoostRegressor] = None
        self.cpd_: Optional[ConformalPredictiveDistribution] = None
        self.feature_names_: Optional[list[str]] = None
        self.best_params_: dict[str, Any] = {}
        self.best_rmse_: Optional[float] = None
        self._X_cal: Optional[ArrayLike] = None
        self._y_cal: Optional[np.ndarray] = None
        self._calibrators: dict[float, Any] = {}

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
            calibrate_threshold: Optional[float] = None,
    ) -> "CatBoostRegressionModel":
        """Fit the model.
        
        Args:
            X: Features.
            y: Target values.
            cat_features: List of categorical feature indices or names.
            calibrate_threshold: If provided and conformal=True, pre-fit the
                probability calibrator for this threshold. This avoids expensive
                lazy fitting during inference.
        """
        X_input, y_arr = self._validate_xy(X, y)
        self.feature_names_ = self._feature_names(X_input)

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
            self._X_cal = X_cal
            self._y_cal = y_cal
            self._calibrators = {}

            if calibrate_threshold is not None:
                self._get_or_fit_calibrator(calibrate_threshold, self.calibration_method)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(self._validate_x(X))

    def prob_exceeds(
            self,
            X: ArrayLike,
            threshold: float,
            *,
            calibration_method: Optional[CalibrationMethod] = None,
    ) -> np.ndarray:
        """P(Y > threshold | X) from the conformal predictive distribution.
        
        Args:
            X: Features to predict on.
            threshold: Value above which to compute exceedance probability.
            calibration_method: Override calibration method ("isotonic", "platt", "none").
                Defaults to self.calibration_method.
        """
        self._check_conformal()
        y_hat = self.predict(X)
        raw_probs = self.cpd_.prob_exceeds(y_hat, threshold)

        method = calibration_method if calibration_method is not None else self.calibration_method
        if method == "none":
            return raw_probs

        calibrator = self._get_or_fit_calibrator(threshold, method)
        return self._apply_calibrator(calibrator, raw_probs, method)

    def _get_or_fit_calibrator(
            self, threshold: float, method: CalibrationMethod
    ) -> Any:
        """Fit or retrieve a calibrator for the given threshold and method."""
        key = (threshold, method)
        if key not in self._calibrators:
            y_hat_cal = self.model_.predict(self._X_cal)
            raw_probs_cal = self.cpd_.prob_exceeds(y_hat_cal, threshold)
            y_exc_cal = (self._y_cal > threshold).astype(int)

            if method == "isotonic":
                calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                calibrator.fit(raw_probs_cal, y_exc_cal)
            elif method == "platt":
                calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
                calibrator.fit(raw_probs_cal.reshape(-1, 1), y_exc_cal)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            self._calibrators[key] = calibrator

        return self._calibrators[key]

    @staticmethod
    def _apply_calibrator(calibrator: Any, probs: np.ndarray, method: CalibrationMethod) -> np.ndarray:
        """Apply a fitted calibrator to raw probabilities."""
        if method == "isotonic":
            return calibrator.predict(probs)
        elif method == "platt":
            return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        return probs

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

    def feature_importance(
            self,
            importance_type: str = "FeatureImportance",
    ) -> pd.Series:
        """Return CatBoost feature importances indexed by input variable name."""
        self._check_fitted()

        scores = self.model_.get_feature_importance(type=importance_type)
        names = self.feature_names_ or self.model_.feature_names_
        if names is None:
            names = [f"f{i}" for i in range(len(scores))]

        importance_scr = pd.Series(scores, index=names, name=importance_type).sort_values(ascending=False)

        return importance_scr

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
            **self.catboost_params,
        }

    @staticmethod
    def _feature_names(X: ArrayLike) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        n_features = np.asarray(X).shape[1]
        return [f"f{i}" for i in range(n_features)]

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
