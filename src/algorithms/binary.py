from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]
CalibrationMethod = Literal["isotonic", "platt"]


class CatBoostAUCClassifier:
    """Binary CatBoost classifier with optional Optuna tuning and probability calibration."""

    def __init__(
            self,
            *,
            optimize: bool = False,
            calibrate: bool = False,
            calibration_method: CalibrationMethod = "isotonic",
            n_trials: int = 50,
            val_size: float = 0.2,
            cal_size: float = 0.2,
            random_state: int = 42,
            early_stopping_rounds: int = 50,
            catboost_params: Optional[dict[str, Any]] = None,
            optuna_seed: Optional[int] = None,
    ):
        self.optimize = optimize
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.n_trials = n_trials
        self.val_size = val_size
        self.cal_size = cal_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.catboost_params = catboost_params or {}
        self.optuna_seed = random_state if optuna_seed is None else optuna_seed

        self.model_: Optional[CatBoostClassifier] = None
        self.calibrator_: Optional[Any] = None
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

        if self.calibrate:
            X_fit, X_cal, y_fit, y_cal = self._calibration_split(X_input, y_arr)
        else:
            X_fit, y_fit = X_input, y_arr
            X_cal = y_cal = None

        if self.optimize:
            self.best_params_ = self._optimize_params(X_fit, y_fit, cat_features=cat_features)
        else:
            self.best_params_ = self._default_params()

        self.model_ = CatBoostClassifier(**self.best_params_)
        self.model_.fit(
            X_fit,
            y_fit,
            cat_features=cat_features,
        )

        if self.calibrate and X_cal is not None and y_cal is not None:
            self.calibrator_ = self._fit_calibrator(X_cal, y_cal)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(self._validate_x(X))

    def predict_proba(
            self,
            X: ArrayLike,
            *,
            calibrated: Optional[bool] = None,
    ) -> np.ndarray:
        """Return class probabilities.
        
        Args:
            X: Features.
            calibrated: If True, return calibrated probabilities. If False, return
                raw model probabilities. Defaults to self.calibrate.
        
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class.
        """
        self._check_fitted()
        raw_proba = self.model_.predict_proba(self._validate_x(X))

        use_calibration = calibrated if calibrated is not None else self.calibrate
        if not use_calibration or self.calibrator_ is None:
            return raw_proba

        calibrated_pos = self._apply_calibrator(raw_proba[:, 1])
        return np.column_stack([1 - calibrated_pos, calibrated_pos])

    def predict_proba_positive(
            self,
            X: ArrayLike,
            *,
            calibrated: Optional[bool] = None,
    ) -> np.ndarray:
        """Return probability of the positive class only.
        
        Convenience method that returns just P(y=1|X).
        """
        return self.predict_proba(X, calibrated=calibrated)[:, 1]

    def _fit_calibrator(self, X_cal: ArrayLike, y_cal: np.ndarray) -> Any:
        """Fit a calibrator on the calibration set."""
        raw_proba = self.model_.predict_proba(X_cal)[:, 1]

        if self.calibration_method == "isotonic":
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            calibrator.fit(raw_proba, y_cal)
        elif self.calibration_method == "platt":
            calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            calibrator.fit(raw_proba.reshape(-1, 1), y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        return calibrator

    def _apply_calibrator(self, proba: np.ndarray) -> np.ndarray:
        """Apply the fitted calibrator to raw probabilities."""
        if self.calibration_method == "isotonic":
            return self.calibrator_.predict(proba)
        elif self.calibration_method == "platt":
            return self.calibrator_.predict_proba(proba.reshape(-1, 1))[:, 1]
        return proba

    def _calibration_split(
            self,
            X: ArrayLike,
            y: np.ndarray,
    ) -> tuple[ArrayLike, ArrayLike, np.ndarray, np.ndarray]:
        """Split data into training and calibration sets."""
        return train_test_split(
            X,
            y,
            test_size=self.cal_size,
            random_state=self.random_state,
            stratify=y,
        )

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

    def feature_importance(
            self,
            importance_type: str = "FeatureImportance",
    ) -> pd.Series:
        """Return CatBoost feature importances indexed by feature name.
        
        Args:
            importance_type: Type of feature importance to compute. Options:
                - "FeatureImportance" (default): Based on loss change
                - "ShapValues": SHAP-based importance (slower)
                - "PredictionValuesChange": Average prediction change
        
        Returns:
            Series with feature importances sorted descending.
        """
        self._check_fitted()
        scores = self.model_.get_feature_importance(type=importance_type)
        names = self.model_.feature_names_
        if names is None:
            names = [f"f{i}" for i in range(len(scores))]
        return pd.Series(scores, index=names, name=importance_type).sort_values(ascending=False)


class RFAUCClassifier:
    """Binary sklearn Random Forest classifier with optional Optuna tuning and calibration.
    
    Supports sampling to limit training data size for faster training.
    """

    def __init__(
            self,
            *,
            optimize: bool = False,
            calibrate: bool = False,
            calibration_method: CalibrationMethod = "isotonic",
            n_trials: int = 50,
            val_size: float = 0.2,
            cal_size: float = 0.2,
            random_state: int = 42,
            max_train_samples: int = 10000,
            rf_params: Optional[dict[str, Any]] = None,
            optuna_seed: Optional[int] = None,
    ):
        self.optimize = optimize
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.n_trials = n_trials
        self.val_size = val_size
        self.cal_size = cal_size
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.rf_params = rf_params or {}
        self.optuna_seed = random_state if optuna_seed is None else optuna_seed

        self.model_: Optional[RandomForestClassifier] = None
        self.calibrator_: Optional[Any] = None
        self.best_params_: dict[str, Any] = {}
        self.best_auc_: Optional[float] = None
        self.label_encoders_: dict[str, LabelEncoder] = {}
        self.cat_features_: Optional[list[str]] = None
        self.feature_names_: Optional[list[str]] = None

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            cat_features: Optional[list[Union[int, str]]] = None,
    ) -> "RFAUCClassifier":
        X_input, y_arr = self._validate_xy(X, y)

        # Store feature names
        if isinstance(X_input, pd.DataFrame):
            self.feature_names_ = list(X_input.columns)

        # Store categorical feature info and encode them
        self.cat_features_ = cat_features
        X_encoded = self._encode_categoricals(X_input, cat_features, fit=True)

        if self.calibrate:
            X_fit, X_cal, y_fit, y_cal = self._calibration_split(X_encoded, y_arr)
        else:
            X_fit, y_fit = X_encoded, y_arr
            X_cal = y_cal = None

        # Sample if too many observations
        X_fit, y_fit = self._sample_data(X_fit, y_fit)

        if self.optimize:
            self.best_params_ = self._optimize_params(X_fit, y_fit)
        else:
            self.best_params_ = self._default_params()

        self.model_ = RandomForestClassifier(**self.best_params_)
        self.model_.fit(X_fit, y_fit)

        if self.calibrate and X_cal is not None and y_cal is not None:
            self.calibrator_ = self._fit_calibrator(X_cal, y_cal)

        return self

    def _sample_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample data if it exceeds max_train_samples."""
        n_samples = len(y)
        if n_samples <= self.max_train_samples:
            return X, y
        
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n_samples, size=self.max_train_samples, replace=False)
        return X[indices], y[indices]

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X_encoded = self._encode_categoricals(X, self.cat_features_, fit=False)
        return self.model_.predict(X_encoded)

    def predict_proba(
            self,
            X: ArrayLike,
            *,
            calibrated: Optional[bool] = None,
    ) -> np.ndarray:
        self._check_fitted()
        X_encoded = self._encode_categoricals(X, self.cat_features_, fit=False)
        raw_proba = self.model_.predict_proba(X_encoded)

        use_calibration = calibrated if calibrated is not None else self.calibrate
        if use_calibration and self.calibrator_ is not None:
            calibrated_pos = self._apply_calibrator(raw_proba[:, 1])
            return np.column_stack([1 - calibrated_pos, calibrated_pos])

        return raw_proba

    def predict_proba_positive(
            self,
            X: ArrayLike,
            *,
            calibrated: Optional[bool] = None,
    ) -> np.ndarray:
        return self.predict_proba(X, calibrated=calibrated)[:, 1]

    def _encode_categoricals(
            self,
            X: ArrayLike,
            cat_features: Optional[list[Union[int, str]]],
            fit: bool = False,
    ) -> np.ndarray:
        """Encode categorical features using LabelEncoder."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        if cat_features is None:
            return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        for col in cat_features:
            col_name = X.columns[col] if isinstance(col, int) else col
            if fit:
                le = LabelEncoder()
                X[col_name] = le.fit_transform(X[col_name].astype(str))
                self.label_encoders_[col_name] = le
            else:
                le = self.label_encoders_[col_name]
                # Handle unseen categories
                X[col_name] = X[col_name].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    def _fit_calibrator(self, X_cal: ArrayLike, y_cal: np.ndarray) -> Any:
        raw_proba = self.model_.predict_proba(X_cal)[:, 1]

        if self.calibration_method == "isotonic":
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            calibrator.fit(raw_proba, y_cal)
        elif self.calibration_method == "platt":
            calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            calibrator.fit(raw_proba.reshape(-1, 1), y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        return calibrator

    def _apply_calibrator(self, proba: np.ndarray) -> np.ndarray:
        if self.calibration_method == "isotonic":
            return self.calibrator_.predict(proba)
        elif self.calibration_method == "platt":
            return self.calibrator_.predict_proba(proba.reshape(-1, 1))[:, 1]
        return proba

    def _calibration_split(
            self,
            X: ArrayLike,
            y: np.ndarray,
    ) -> tuple[ArrayLike, ArrayLike, np.ndarray, np.ndarray]:
        return train_test_split(
            X,
            y,
            test_size=self.cal_size,
            random_state=self.random_state,
            stratify=y,
        )

    def _optimize_params(
            self,
            X: ArrayLike,
            y: np.ndarray,
    ) -> dict[str, Any]:
        X_train, X_val, y_train, y_val = self._holdout_split(X, y)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.optuna_seed),
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
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
    ) -> float:
        params = {
            **self._default_params(),
            **self._suggest_params(trial),
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, y_score))

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
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
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": self.random_state,
            "n_jobs": -1,
            "class_weight": "balanced",
            **self.rf_params,
        }

    @staticmethod
    def _validate_xy(X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, np.ndarray]:
        y_arr = np.asarray(y).ravel()
        if y_arr.ndim != 1:
            raise ValueError("`y` must be a 1d array-like object.")
        n_rows = RFAUCClassifier._row_count(X)
        if n_rows != len(y_arr):
            raise ValueError("`X` and `y` must have the same number of rows.")
        return X, y_arr

    @staticmethod
    def _row_count(X: ArrayLike) -> int:
        if isinstance(X, pd.DataFrame):
            return len(X)
        return len(np.asarray(X))

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Call fit before predict or predict_proba.")

    def feature_importance(self) -> pd.Series:
        """Return Random Forest feature importances."""
        self._check_fitted()
        scores = self.model_.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(scores))]
        return pd.Series(scores, index=names, name="importance").sort_values(ascending=False)
