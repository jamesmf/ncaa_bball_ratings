import typing as T

from optuna import Trial
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import sklearn.metrics
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
)
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.pipeline import Pipeline
from .tournament_dataset import MMadnessDataset


def get_suggestions(trial: Trial):
    model_cls = trial.suggest_categorical(
        "classifier",
        [
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "AdaBoostRegressor",
        ],
    )
    if model_cls == "RandomForestClassifier":
        kwargs = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 20, 100, log=True),
            "max_depth": trial.suggest_int(
                "rf_max_depth",
                4,
                8,
            ),
            "max_features": trial.suggest_int("rf_max_features", 4, 8),
            "max_samples": trial.suggest_float(
                "rf_max_samples",
                0.5,
                0.8,
            ),
            "min_samples_split": trial.suggest_int(
                "rf_min_samples_split",
                2,
                5,
            ),
        }
    elif model_cls == "AdaBoostClassifier":
        kwargs = {
            "n_estimators": trial.suggest_int(
                "adaboostclassifer_n_estimators", 20, 150, log=True
            ),
            "learning_rate": trial.suggest_float(
                "adaboostclassifer_learning_rate", 0.1, 1
            ),
        }
    elif model_cls == "AdaBoostRegressor":
        kwargs = {
            "n_estimators": trial.suggest_int(
                "adaboostregressor_n_estimators", 20, 150, log=True
            ),
            "learning_rate": trial.suggest_float(
                "adaboostregressor_learning_rate", 0.1, 1
            ),
            "loss": "linear",
        }
    else:
        print(type(model_cls))
        print(model_cls)
    feature_selection = trial.suggest_categorical(
        "feature_selection_cls",
        [None, "SelectPercentile"],
    )
    fs_kwargs = (
        {"percentile": trial.suggest_int("feature_selection_percentile", 10, 100)}
        if feature_selection is not None
        else {}
    )

    return {
        "model_cls": model_cls,
        "scaling": trial.suggest_categorical("scaling", [True, False]),
        "feature_selection": feature_selection,
        "feature_selection_kwargs": fs_kwargs,
        "kwargs": kwargs,
    }


def init_model(
    model_cls: str,
    scaling: bool,
    feature_selection: T.Optional[str],
    feature_selection_kwargs: dict[str, T.Any],
    kwargs: dict[str, T.Any],
) -> BaseEstimator:
    if model_cls == "RandomForestClassifier":
        model = RandomForestClassifier(
            **kwargs,
        )
    elif model_cls == "AdaBoostClassifier":
        model = AdaBoostClassifier(**kwargs)
    elif model_cls == "AdaBoostRegressor":
        model = AdaBoostRegressor(**kwargs)

    steps = [("model", model)]
    if scaling:
        steps.insert(0, ("scaler", MinMaxScaler()))
    if feature_selection == "SelectPercentile":
        steps.insert(
            0,
            (
                "feature_selection",
                SelectPercentile(mutual_info_regression, **feature_selection_kwargs),
            ),
        )

    if len(steps) > 1:
        model = Pipeline(steps)

    return model


def train(
    ds: MMadnessDataset, model: BaseEstimator
) -> tuple[BaseEstimator, np.ndarray, np.ndarray]:
    cols = ds.X.columns
    x_train, y_train = ds.X.reset_index(drop=True), ds.y * 1.0
    model.fit(x_train, y_train)
    x_test = pd.DataFrame(ds.X_test.reset_index(drop=True), columns=cols)
    y_test = ds.y_test * 1.0 if ds.y_test is not None else None
    if hasattr(model, "predict_proba"):
        preds_raw = model.predict_proba(x_test)[:, 1]
    else:
        preds_raw = model.predict(x_test)

    return model, preds_raw, y_test


def evaluate_model_on_years(
    trial: Trial,
    ds_all: dict[int, MMadnessDataset],
    years: list[int],
) -> float:
    losses = []
    mses = []
    suggestions = get_suggestions(trial)
    for year in years:
        ds = ds_all[year]
        model = init_model(**suggestions)
        model, preds_raw, y_test = train(ds, model)
        test_ll = sklearn.metrics.log_loss(y_test, preds_raw)
        test_mse = sklearn.metrics.mean_squared_error(y_test, preds_raw)

        losses.append(test_ll)
        mses.append(test_mse)
    # mean_loss = np.mean(losses)
    mean_mses = np.mean(mses)

    return mean_mses
