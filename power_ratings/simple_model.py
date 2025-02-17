import typing as T
import argparse

import numpy as np
import pandas as pd

import sklearn.calibration
import sklearn.pipeline
import sklearn.preprocessing
import mlflow
import plotly.express as px
import plotly.graph_objects as go

from power_ratings.tournament_dataset import MMadnessDataset
import power_ratings.tournament_dataset as td


class PassthroughClassifier(sklearn.base.BaseEstimator):
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return y

    def predict(self, X):
        return X

    def predict_proba(self, X):
        x0 = X[:, 0]
        s = np.stack([1 - x0, x0], axis=1)
        return s

    def __sklearn_is_fitted__(self):
        return True


class VotingRegressorNoCheck(sklearn.ensemble.VotingRegressor):

    def __init__(self, estimators: list[sklearn.base.BaseEstimator]):
        super().__init__(
            [(f"regressor_{i}", estimators[i]) for i in range(len(estimators))]
        )
        self.estimators_ = estimators

    def predict(self, X):
        return np.average(self._predict(X), axis=1)


def cross_val_predict_return_model_and_preds(
    model_base: sklearn.base.BaseEstimator, x: np.ndarray, y: np.ndarray
) -> tuple[list[sklearn.base.BaseEstimator], np.ndarray]:
    results = sklearn.model_selection.cross_validate(
        model_base,
        x,
        y,
        cv=sklearn.model_selection.StratifiedKFold(),
        # n_jobs=4,
        return_estimator=True,
        return_indices=True,
        return_train_score=True,
        scoring="explained_variance",
    )
    print("train score", results["train_score"])
    print("test score", results["test_score"])
    estimators = results["estimator"]
    indices = results["indices"]["test"]
    test_preds = []
    for estimator, test_inds in zip(estimators, indices):
        test_x = x[test_inds]
        test_preds.extend(list(zip(estimator.predict(test_x), test_inds)))
    preds_out = np.array([i[0] for i in sorted(test_preds, key=lambda x: x[1])])
    return VotingRegressorNoCheck(estimators), np.round(preds_out, 2)


def calibration_plot(ytrue: np.ndarray, ypred: np.ndarray, name) -> go.Figure:
    calibration_display = sklearn.calibration.CalibrationDisplay.from_predictions(
        ytrue, ypred, n_bins=15, strategy="uniform"
    )
    fig = px.scatter(
        pd.DataFrame(
            zip(calibration_display.prob_pred, calibration_display.prob_true),
            columns=["pred", "true"],
        ),
        x="pred",
        y="true",
    )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines")
    mlflow.log_figure(fig, f"{name}.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+")
    args = ap.parse_args()
    submissions = []
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"calibrated-model")
    mlflow.autolog(disable=True)
    run = mlflow.start_run()
    mlflow.log_param("years", str(args.years))
    for year in args.years:
        for prefix in ("W", "M"):
            ds_params = {
                "holdout_seasons": None,
                "prefix": prefix,
                "start_year": 2003,
                "curr_year": year,
                "extra_features": [
                    "T1CombinedRating",
                    "T1PossessionEfficiencyFactor",
                    "T1EloWithScore",
                    "T1EloWinLoss",
                    "T1EloDay30WithScore",
                    "T1EloDay30WinLoss",
                    "T1WP16",
                    "T1Seed",
                    "T1EloDelta21Days",
                    "T2WP16",
                    "T2CombinedRating",
                    "T2PossessionEfficiencyFactor",
                    "T2EloWithScore",
                    "T2EloWinLoss",
                    "T2EloDay30WithScore",
                    "T2EloDay30WinLoss",
                    "T2Seed",
                    "T2EloDelta21Days",
                    "round",
                ],
                "holdout_strategy": "prior",
            }

            dataset = MMadnessDataset(
                **ds_params,
            )

            core_cols = [
                "T1WinsPMEstimate",
                "WP16_diff",
                "elo1_diff",
                "elo2_diff",
            ]

            base_clf = sklearn.linear_model.LogisticRegression()
            cc = sklearn.calibration.CalibratedClassifierCV(
                estimator=base_clf, cv=5, ensemble=True, method="isotonic"
            )
            x = dataset.X[core_cols].values
            preds = cc.fit(x, dataset.y).predict_proba(x)[:, 1]

            # score_model = sklearn.ensemble.RandomForestRegressor(n_estimators=50)
            score_model = sklearn.pipeline.Pipeline(
                [
                    ("scaler", sklearn.preprocessing.RobustScaler()),
                    ("model", sklearn.linear_model.ElasticNet()),
                ]
            )
            score_estimator, score_preds = cross_val_predict_return_model_and_preds(
                score_model,
                x,
                dataset.reg_y,
            )

            base_clf2 = sklearn.linear_model.LogisticRegression()
            cc2 = sklearn.calibration.CalibratedClassifierCV(
                estimator=base_clf2, cv=5, ensemble=True, method="isotonic"
            )
            score_based_preds = cc2.fit(
                score_preds.reshape(-1, 1), dataset.y
            ).predict_proba(score_preds.reshape(-1, 1))[:, 1]

            calibration_plot(
                dataset.y,
                0.66 * preds + 0.34 * x[:, 0],
                f"{year}{prefix}_calibration_averaged_pm_estimate",
            )

            calibration_plot(
                dataset.y,
                score_based_preds,
                f"{year}{prefix}_calibration_score_based.html",
            )

            calibration_plot(
                dataset.y,
                0.2 * preds + 0.05 * x[:, 0] + 0.75 * score_based_preds,
                f"{year}{prefix}_combined",
            )

            output_preds_0 = dataset.submission.T1WinsPMEstimate
            output_preds_1 = cc.predict_proba(dataset.submission[core_cols].values)[
                :, 1
            ]
            x_sub = dataset.submission[core_cols].values
            output_preds_2 = cc2.predict_proba(
                score_estimator.predict(x_sub).reshape(-1, 1)
            )[:, 1]
            output_preds = (
                0.39 * output_preds_1 + 0.01 * output_preds_0 + 0.6 * output_preds_2
            )
            # output_preds = output_preds_2
            output_df = dataset.submission[["ID"]].copy()
            output_df["Pred"] = output_preds
            submissions.append(output_df)

    submission_df = pd.concat(submissions)
    submission_df.to_csv("submission.csv", index=False)
    mlflow.log_artifact("submission.csv")

    # calibration_display2 = sklearn.calibration.CalibrationDisplay.from_predictions(
    #     dataset.y, preds, n_bins=15, strategy="uniform"
    # )
    # fig2 = px.scatter(
    #     pd.DataFrame(
    #         zip(calibration_display2.prob_pred, calibration_display2.prob_true),
    #         columns=["pred", "true"],
    #     ),
    #     x="pred",
    #     y="true",
    # )
    # mlflow.log_figure(fig2, "calibration_tuned_pm_estimate.html")

    # preds2 = np.where(
    #     (x[:, 0] < 0.2) | (x[:, 0] > 0.8),
    #     0.33 * x[:, 0] + 0.67 * preds,
    #     0.5 * preds + 0.5 * x[:, 0],
    # )
    # calibration_display3 = sklearn.calibration.CalibrationDisplay.from_predictions(
    #     dataset.y, preds2, n_bins=15, strategy="uniform"
    # )
    # fig3 = fig = px.scatter(
    #     pd.DataFrame(
    #         zip(calibration_display3.prob_pred, calibration_display3.prob_true),
    #         columns=["pred", "true"],
    #     ),
    #     x="pred",
    #     y="true",
    # )
    # mlflow.log_figure(fig3, "calibration_manual_pm_estimate.html")

    # calibration_display4 = sklearn.calibration.CalibrationDisplay.from_predictions(
    #     dataset.y, x[:, 0], n_bins=15, strategy="uniform"
    # )
    # fig4 = px.scatter(
    #     pd.DataFrame(
    #         zip(calibration_display4.prob_pred, calibration_display4.prob_true),
    #         columns=["pred", "true"],
    #     ),
    #     x="pred",
    #     y="true",
    # )
    # mlflow.log_figure(fig4, "original_pm_estimate.html")
