import os
import typing as T
import time

import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.model_selection
import scipy.stats
import pandas as pd
import numpy as np


def feature_rename(df: pd.DataFrame, wl: str):
    """
    Rename the feature columns based on whether you're joining to the winning or losing team ID
    """
    return df.rename(columns={c: wl + c for c in df.columns})


def df_rename(df: pd.DataFrame, t1_prefix: str, t2_prefix: str):
    """
    Rename columns based on whether we care about the winning or losing team
    """
    renames = {}
    for col in df.columns:
        if col[0] == t1_prefix:
            renames[col] = "T1" + col[1:]
        if col[0] == t2_prefix:
            renames[col] = "T2" + col[1:]
    return df.rename(columns=renames)


def prob_t1_score_gt_t2(
    t1_off: float,
    t1_def: float,
    t2_off: float,
    t2_def: float,
    sigma1: float,
    sigma2: float,
    base: float,
    scaler: float,
) -> float:
    """If we modeled the score distribution of one team
    as Norm((t1_off - t2_def)*scaler + base, sigma), we
    can get the distribution of t1_score - t2_score to
    get a point estimate of how likely it is t1 wins
    """
    t1_score_mu = (t1_off - t2_def) * scaler + base
    t2_score_mu = (t2_off - t1_def) * scaler + base
    new_mean = t1_score_mu - t2_score_mu
    new_sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2)
    return 1 - scipy.stats.norm.cdf(0, new_mean, new_sigma)


def probabilistic_estimate_df(
    df: pd.DataFrame,
    base: float,
    scaler: float,
) -> pd.Series:
    """Use the probabilistic model we used to generate
    the team off/def scores to get a point estimate of
    the probability that team1 wins (has a higher score)

    Args:
        df (pd.DataFrame): tournament or submission df

    Returns:
        pd.Series: probabilities
    """
    return df.apply(
        lambda x: prob_t1_score_gt_t2(
            x["T1OffensiveRating"],
            x["T1DefensiveRating"],
            x["T2OffensiveRating"],
            x["T2DefensiveRating"],
            sigma1=x["T1ScoreVariance"],
            sigma2=x["T2ScoreVariance"],
            base=base,
            scaler=scaler,
        ),
        axis=1,
    )


class MMadnessDataset:
    def __init__(
        self,
        base_path: str = "data/",
        prefix: str = "W",
        start_year: int = 2002,
        curr_year: int = 2022,
        stage_num: str = "2",
        holdout_seasons: T.Optional[T.List[int]] = None,
        # core_features=[
        #     "CombinedRating",
        #     "OffensiveRating",
        #     "DefensiveRating",
        #     "EloWinLoss",
        #     "EloWithScore",
        # ],
        extra_features: T.List[str] = [],
        holdout_strategy: str = "all",
    ):
        self.holdout_seasons = holdout_seasons
        self.holdout_strategy = holdout_strategy
        self.extra_features = extra_features

        if prefix == "M":
            self.estimated_score_scaler = 0.59
            self.estimated_score_base = 69.5
        if prefix == "W":
            self.estimated_score_scaler = 0.79
            self.estimated_score_base = 64.0

        features_path = os.path.join(base_path, f"../output/{prefix}_data_complete.csv")
        tourney_path = os.path.join(base_path, f"{prefix}NCAATourneyCompactResults.csv")
        sub_path = os.path.join(
            base_path, f"{prefix}SampleSubmissionStage{stage_num}.csv"
        )
        seed_path = os.path.join(base_path, f"{prefix}NCAATourneySeeds.csv")
        team_name_path = os.path.join(base_path, f"{prefix}Teams.csv")

        self.seeds = pd.read_csv(seed_path)
        self.team_names = pd.read_csv(team_name_path)
        self.features = (
            pd.read_csv(features_path).drop_duplicates().set_index(["Season", "TeamID"])
        )
        self.tourneys = pd.read_csv(tourney_path)
        self.submission = pd.read_csv(sub_path)[["ID"]]

        self.tourneys = self.tourneys[self.tourneys.Season >= start_year]
        self.submission["Season"] = curr_year
        self.submission["Team1ID"] = self.submission.ID.apply(lambda x: int(x[5:9]))
        self.submission["Team2ID"] = self.submission.ID.apply(lambda x: int(x[10:14]))

        joined = pd.merge(
            self.tourneys,
            feature_rename(self.features, "W"),
            left_on=["Season", "WTeamID"],
            right_index=True,
            how="inner",
        )
        joined = pd.merge(
            joined,
            feature_rename(self.features, "L"),
            left_on=["Season", "LTeamID"],
            right_index=True,
            how="inner",
        )
        joined["diff"] = joined["WScore"] - joined["LScore"]
        joined["diff"] = joined[["diff", "NumOT"]].apply(
            lambda x: np.min([x["diff"], 3]) if x["NumOT"] > 0 else x["diff"], axis=1
        )

        self.submission = pd.merge(
            self.submission,
            feature_rename(self.features, "T1"),
            left_on=["Season", "Team1ID"],
            right_index=True,
            how="inner",
        )
        self.submission = pd.merge(
            self.submission,
            feature_rename(self.features, "T2"),
            left_on=["Season", "Team2ID"],
            right_index=True,
            how="inner",
        )

        pos = df_rename(joined, "W", "L")
        neg = df_rename(joined, "L", "W")
        neg["diff"] = -neg["diff"]

        self.combined = pd.concat((pos, neg))
        self.combined["target"] = self.combined["diff"] > 0

        # now calculate "difference" features for all the core features

        for specific_df in (self.combined, self.submission):
            specific_df[f"T1OD_diff"] = (
                specific_df[f"T1OffensiveRating"] - specific_df[f"T2DefensiveRating"]
            )
            specific_df[f"T2OD_diff"] = (
                specific_df[f"T2OffensiveRating"] - specific_df[f"T1DefensiveRating"]
            )
            specific_df["elo1_diff"] = (
                specific_df[f"T1EloWithScore"] - specific_df[f"T2EloWithScore"]
            )
            specific_df["elo2_diff"] = (
                specific_df[f"T1EloWinLoss"] - specific_df[f"T2EloWinLoss"]
            )
            specific_df["poss_eff_diff"] = (
                specific_df[f"T1PossessionEfficiency"]
                - specific_df[f"T2PossessionEfficiency"]
            )

            # calculate our game-level "T1 wins" estimate from our probabilistic model
            specific_df["T1WinsPMEstimate"] = probabilistic_estimate_df(
                specific_df, self.estimated_score_base, self.estimated_score_scaler
            )

        self.core_features = [
            "T1OD_diff",
            "T2OD_diff",
            "elo1_diff",
            "elo2_diff",
            "T1WinsPMEstimate",
            "poss_eff_diff",
        ]

    def get_mask(self, train=True):
        """Return mask of the data based on the holdout and the holdout strategy.
        If `train==True` then return the training data, else return the test set.

        If `self.holdout_strategy == 'all'` then the training set is all years
        not in `self.holdout_season`. If `self.holdout_strategy == 'prior'` then
        return everything prior to the minimum year in `self.holdout_season`

        Args:
            train (bool, optional): whether to return the training set mask (vs test). Defaults to True.

        Returns:
            pd.Series: the boolean mask of the training or test set
        """
        if self.holdout_seasons is None:
            return [True] * len(self.combined)
        if self.holdout_strategy == "all":
            if train:
                return ~self.combined.Season.isin(self.holdout_seasons)
            return self.combined.Season.isin(self.holdout_seasons)
        elif self.holdout_strategy == "prior":
            min_holdout = np.min(self.holdout_seasons)
            prior = self.combined.Season < min_holdout
            if train:
                return (prior) & (~self.combined.Season.isin(self.holdout_seasons))
            return self.combined.Season.isin(self.holdout_seasons)
        else:
            raise NotImplementedError(
                "Only holdout strategies implemented are ('all', 'prior')"
            )

    @property
    def X(self):
        if self.holdout_seasons is None:
            return self.combined[self.feature_names]
        return self.combined[self.get_mask()][self.feature_names]

    @property
    def y(self):
        if self.holdout_seasons is None:
            return self.combined.target
        return self.combined[self.get_mask()].target

    @property
    def reg_y(self):
        if self.holdout_seasons is None:
            return self.combined["diff"].apply(lambda x: np.max([np.min([x, 15]), -15]))
        return self.combined[self.get_mask()]["diff"].apply(
            lambda x: np.max([np.min([x, 15]), -15])
        )

    @property
    def X_test(self):
        if self.holdout_seasons is None:
            return self.submission[self.feature_names]
        return self.combined[self.get_mask(train=False)][self.feature_names]

    @property
    def y_test(self):
        if self.holdout_seasons is None:
            return None
        return self.combined[self.get_mask(train=False)].target

    @property
    def reg_y_test(self):
        if self.holdout_seasons is None:
            return None
        return self.combined[self.get_mask(train=False)]["diff"].apply(
            lambda x: np.max([np.min([x, 15]), -15])
        )

    @property
    def feature_names(self) -> T.List[str]:
        """return just the columns that constitute our feature representation

        Returns:
            T.List[str]:list of feature columns
        """
        if hasattr(self, "core_features"):
            return self.core_features + self.extra_features
        cols = [c for c in self.features.columns if c not in ("Season", "TeamID")]
        return [
            team_id + col_name for col_name in cols for team_id in ("T1", "T2")
        ] + self.extra_features

    def get_cv_splits(self) -> sklearn.model_selection.LeaveOneOut:
        groups = self.combined[self.get_mask()].Season
        cv = sklearn.model_selection.LeaveOneGroupOut()
        return cv.split(self.X, groups=groups)

    def get_final_preds(
        self, model, strategy: str = "last_game"
    ) -> T.Sequence[pd.DataFrame]:
        if self.holdout_seasons is None:
            preds = model.predict_proba(self.submission[self.feature_names])[:, 1]
            sub = self.submission.copy()
        else:
            preds = model.predict_proba(self.X_test)[:, 1]
            sub = self.combined[self.get_mask(train=False)].copy()
            sub = sub.rename(columns={"T1TeamID": "Team1ID", "T2TeamID": "Team2ID"})
            sub["ID"] = ""
        sub["Pred"] = preds
        sub = pd.merge(
            sub,
            self.seeds,
            how="left",
            left_on=["Team1ID", "Season"],
            right_on=["TeamID", "Season"],
        )
        sub = pd.merge(
            sub,
            self.seeds,
            how="left",
            left_on=["Team2ID", "Season"],
            right_on=["TeamID", "Season"],
            suffixes=("_1", "_2"),
        )
        strat_1 = sub.copy()
        strat_2 = sub.copy()
        if strategy == "last_game":
            last_game_mask = (sub["Seed_1"].apply(lambda x: x[0] in ("W", "X"))) & (
                sub["Seed_2"].apply(lambda x: x[0] in ("Y", "Z"))
            )
            strat_1.loc[last_game_mask, "Pred"] = 1e-5
            strat_2.loc[last_game_mask, "Pred"] = 1 - 1e-5
        if strategy in ("upset_none_1", "upset_none_2"):
            # one submission goes highly confident in first round, other hedges
            round1_seed1_left = (sub["Seed_1"].apply(lambda x: x[1:3] == "01")) & (
                sub["Seed_2"].apply(lambda x: x[1:3] == "16")
            )
            round1_seed1_right = (sub["Seed_2"].apply(lambda x: x[1:3] == "01")) & (
                sub["Seed_1"].apply(lambda x: x[1:3] == "16")
            )
            strat_1.loc[round1_seed1_left, "Pred"] = 0.999
            strat_2.loc[round1_seed1_left, "Pred"] = (
                strat_2.loc[round1_seed1_left, "Pred"]
                .apply(lambda x: np.min([0.9, x]))
                .values
            )
            strat_1.loc[round1_seed1_right, "Pred"] = 0.001
            strat_2.loc[round1_seed1_right, "Pred"] = (
                strat_2.loc[round1_seed1_left, "Pred"]
                .apply(lambda x: np.max([0.1, x]))
                .values
            )
        if strategy in ("upset_none_2"):
            round1_seed2_left = (sub["Seed_1"].apply(lambda x: x[1:3] == "02")) & (
                sub["Seed_2"].apply(lambda x: x[1:3] == "15")
            )
            round1_seed2_right = (sub["Seed_2"].apply(lambda x: x[1:3] == "02")) & (
                sub["Seed_1"].apply(lambda x: x[1:3] == "15")
            )
            strat_1.loc[round1_seed2_left, "Pred"] = 0.999
            strat_2.loc[round1_seed2_left, "Pred"] = (
                strat_2.loc[round1_seed1_left, "Pred"]
                .apply(lambda x: np.min([0.92, x]))
                .values
            )
            strat_1.loc[round1_seed2_right, "Pred"] = 0.001
            strat_2.loc[round1_seed2_right, "Pred"] = (
                strat_2.loc[round1_seed1_left, "Pred"]
                .apply(lambda x: np.max([0.08, x]))
                .values
            )

        sub = pd.merge(
            sub,
            self.team_names,
            how="left",
            left_on=["Team1ID"],
            right_on=["TeamID"],
        )
        sub = pd.merge(
            sub,
            self.team_names,
            how="left",
            left_on=["Team2ID"],
            right_on=["TeamID"],
            suffixes=("_1", "_2"),
        )
        cp = ["ID", "Pred"]
        return strat_1[cp], strat_2[cp], sub
