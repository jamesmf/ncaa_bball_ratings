import os
import typing as T
import time

import sklearn
import sklearn.linear_model
import sklearn.pipeline
import sklearn.impute
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.model_selection
import scipy.stats
import pandas as pd
import numpy as np

from .constants import M_PRE_BASE, M_PRE_SCALER, W_PRE_BASE, W_PRE_SCALER


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
    new_sigma = np.sqrt(sigma1**2 + sigma2**2)
    return 1 - scipy.stats.norm.cdf(0, new_mean, new_sigma)


def score_sum_distribution(
    t1_off: float,
    t1_def: float,
    t2_off: float,
    t2_def: float,
    sigma1: float,
    sigma2: float,
    base: float,
    scaler: float,
) -> T.List[float]:
    """If we modeled the score distribution of one team
    as Norm((t1_off - t2_def)*scaler + base, sigma), we
    can get the distribution of t1_score - t2_score to
    get a point estimate of how likely it is t1 wins
    """
    t1_score_mu = (t1_off - t2_def) * scaler + base
    t2_score_mu = (t2_off - t1_def) * scaler + base
    new_mean = t1_score_mu + t2_score_mu
    new_sigma = np.sqrt(sigma1**2 + sigma2**2)
    return [new_mean, *scipy.stats.norm.interval(0.25, new_mean, new_sigma)]


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


def score_estimate_df(
    df: pd.DataFrame,
    base: float,
    scaler: float,
) -> T.Tuple[np.ndarray]:
    """Use the probabilistic model we used to generate
    the team off/def scores to get a point estimate of
    the probability that team1 wins (has a higher score)

    Args:
        df (pd.DataFrame): tournament or submission df

    Returns:
        pd.Series: probabilities
    """
    result = np.array(
        df.apply(
            lambda x: score_sum_distribution(
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
        ).tolist()
    )
    return result


def seeds_to_round(x1, x2):
    """
    From the two seeds, infer the round. Uses seed naming similar to
    Kaggle's dataset, W01, W02, ... Z16 with a/b for play-in
    """
    if str(x1) in ("nan", "none", "None"):
        return 1
    s1 = int(str(x1)[1:3])
    s2 = int(str(x2)[1:3])
    combined = sorted([s1, s2])
    combined_str = f"{combined[0]}.{combined[1]}"
    region_str = f"{'.'.join(sorted([x1[0], x2[0]]))}"

    # cross-region matchups occur in round 5, 6
    if region_str in (
        "W.X",
        "Y.Z",
    ):
        return 5
    if region_str in ("W.Z", "W.Y", "X.Y", "X.Z"):
        return 6

    # playins are between evenly ranked teams
    if s1 == s2:
        return 0
    if combined_str in ("1.16", "2.15", "3.14", "4.13", "5.12", "6.11", "7.10", "8.9"):
        return 1
    if combined_str in (
        "1.8",
        "1.9",
        "8.16",
        "9.16",
        "4.5",
        "4.12",
        "5.13",
        "12.13",
        "3.6",
        "3.11",
        "6.14",
        "11.14",
        "2.7",
        "2.10",
        "7.15",
        "10.15",
    ):
        return 2
    if combined_str in (
        "1.5",
        "1.12",
        "1.4",
        "1.13",
        "5.16",
        "12.16",
        "4.16",
        "13.16",
        "5.9",
        "9.12",
        "4.9",
        "9.13",
        "5.8",
        "8.12",
        "4.8",
        "8.13",
        "2.3",
        "2.6",
        "2.11",
        "2.14",
        "3.7",
        "6.7",
        "7.11",
        "7.14",
        "3.10",
        "6.10",
        "10.11",
        "10.14",
        "3.15",
        "6.15",
        "11.15",
        "14.15",
    ):
        return 3
    return 4


class MMadnessDataset:
    def __init__(
        self,
        base_path: str = "data/",
        prefix: str = "W",
        start_year: int = 2002,
        curr_year: int = 2024,
        sample_weight_method: T.Literal["last_3", "linear"] = "linear",
        holdout_seasons: T.Optional[T.List[int]] = None,
        extra_features: T.List[str] = [],
        holdout_strategy: str = "all",
    ):
        self.curr_year = curr_year
        self.holdout_seasons = holdout_seasons
        self.holdout_strategy = holdout_strategy
        self.extra_features = extra_features
        self.sample_weight_method = sample_weight_method
        self.prefix = prefix
        self.data_dir = os.path.join(base_path, str(self.curr_year))

        if prefix == "M":
            self.estimated_score_scaler = M_PRE_SCALER
            self.estimated_score_base = M_PRE_BASE
        if prefix == "W":
            self.estimated_score_scaler = W_PRE_SCALER
            self.estimated_score_base = W_PRE_BASE

        features_path = os.path.join(base_path, f"../output/{prefix}_data_complete.csv")
        print(features_path)
        tourney_path = os.path.join(
            self.data_dir, f"{prefix}NCAATourneyCompactResults.csv"
        )
        seed_path = os.path.join(base_path, f"{prefix}NCAATourneySeeds.csv")
        team_name_path = os.path.join(base_path, f"{prefix}Teams.csv")

        self.seeds = pd.read_csv(seed_path)
        self.team_names = pd.read_csv(team_name_path)
        print(pd.read_csv(features_path).drop_duplicates().columns)
        self.features = (
            pd.read_csv(features_path).drop_duplicates().set_index(["Season", "TeamID"])
        )

        self.tourneys = pd.read_csv(tourney_path)
        self.submission = self.build_submission_df()

        self.tourneys = self.tourneys[self.tourneys.Season >= start_year]

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
        self.submission = pd.merge(
            self.submission,
            self.seeds,
            how="left",
            left_on=["Team1ID", "Season"],
            right_on=["TeamID", "Season"],
        )
        self.submission = pd.merge(
            self.submission,
            self.seeds,
            how="left",
            left_on=["Team2ID", "Season"],
            right_on=["TeamID", "Season"],
            suffixes=("_1", "_2"),
        )
        self.submission["T1Seed"] = (
            self.submission["Seed_1"]
            .fillna("")
            .apply(lambda x: int(x[1:3]) if x and str(x) != "nan" else 17)
        )
        self.submission["T2Seed"] = (
            self.submission["Seed_2"]
            .fillna("")
            .apply(lambda x: int(x[1:3]) if x and str(x) != "nan" else 17)
        )
        self.submission["round"] = (
            self.submission[["Seed_1", "Seed_2"]]
            .fillna("")
            .apply(lambda x: seeds_to_round(*x) if x[0] and x[1] else 1, axis=1)
        )

        pos = df_rename(joined, "W", "L")
        neg = df_rename(joined, "L", "W")
        neg["diff"] = -neg["diff"]

        self.combined = pd.concat((pos, neg))
        self.combined["target"] = self.combined["diff"] > 0

        self.combined = pd.merge(
            self.combined,
            self.seeds,
            how="left",
            left_on=["T1TeamID", "Season"],
            right_on=["TeamID", "Season"],
        )
        self.combined = pd.merge(
            self.combined,
            self.seeds,
            how="left",
            left_on=["T2TeamID", "Season"],
            right_on=["TeamID", "Season"],
            suffixes=("_1", "_2"),
        )
        self.combined["T1Seed"] = self.combined["Seed_1"].apply(lambda x: int(x[1:3]))
        self.combined["T2Seed"] = self.combined["Seed_2"].apply(lambda x: int(x[1:3]))
        self.combined["round"] = self.combined[["Seed_1", "Seed_2"]].apply(
            lambda x: seeds_to_round(*x), axis=1
        )

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
                specific_df[f"T1PossessionEfficiencyFactor"]
                - specific_df[f"T2PossessionEfficiencyFactor"]
            )
            specific_df["WP16_diff"] = specific_df[f"T1WP16"] - specific_df[f"T2WP16"]

            # calculate our game-level "T1 wins" estimate from our probabilistic model
            specific_df["T1WinsPMEstimate"] = probabilistic_estimate_df(
                specific_df, self.estimated_score_base, self.estimated_score_scaler
            )
            specific_df[["EstScoreMean", "EstScoreLower", "EstScoreUpper"]] = (
                score_estimate_df(
                    specific_df, self.estimated_score_base, self.estimated_score_scaler
                )
            )
            specific_df["elo21d_diff"] = (
                specific_df[f"T1EloDelta21Days"] - specific_df[f"T2EloDelta21Days"]
            )

        self.core_features = [
            "T1OD_diff",
            "T2OD_diff",
            "elo1_diff",
            "elo2_diff",
            "elo21d_diff",
            "T1WinsPMEstimate",
            "poss_eff_diff",
            "WP16_diff",
        ]

        self.score_diff_predictor = sklearn.pipeline.Pipeline(
            [
                ("imputer", sklearn.impute.SimpleImputer()),
                ("cls", sklearn.ensemble.AdaBoostRegressor()),
            ]
        )
        _ = self.score_diff_predictor.fit(self.X[self.core_features], self.reg_y)
        self.score_preds_test = self.score_diff_predictor.predict(
            self.X_test[self.core_features]
        )

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
    def sample_weights(self):
        seasons = self.combined[self.get_mask()].Season.unique()
        if self.sample_weight_method == "last_3":
            last_n_seasons = set(sorted(seasons, reverse=True)[:3])
            return self.combined[self.get_mask()].Season.apply(
                lambda x: 1 if x in last_n_seasons else 0.5
            )
        if self.sample_weight_method == "linear":
            range = seasons.max() - seasons.min()
            return (self.combined[self.get_mask()].Season - seasons.min()) / range + 0.5
        return None

    @property
    def reg_y(self):
        if self.holdout_seasons is None:
            return self.combined["diff"].apply(lambda x: np.max([np.min([x, 30]), -30]))
        return self.combined[self.get_mask()]["diff"].apply(
            lambda x: np.max([np.min([x, 30]), -30])
        )

    @property
    def reg_y_log(self):

        if self.holdout_seasons is None:
            score_diff = self.combined["diff"]
            return score_diff.apply(lambda x: np.sign(x) * np.log(np.abs(x)))
        else:
            score_diff = self.combined[self.get_mask()]["diff"]
        return score_diff.apply(lambda x: np.sign(x) * np.log(np.abs(x)))

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
    def reg_y_test_log(self):
        if self.holdout_seasons is None:
            return None
        score_diff = self.combined[self.get_mask(train=False)]["diff"]
        return score_diff.apply(lambda x: np.sign(x) * np.log(np.abs(x)))

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
        self,
        model,
        strategy: str = "last_game",
        preds: T.Optional[np.ndarray] = None,
    ) -> T.Sequence[pd.DataFrame]:
        if self.holdout_seasons is None:
            if preds is None:
                preds = model.predict_proba(self.submission[self.feature_names])[:, 1]
            sub = self.submission.copy()
        else:
            if preds is None:
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

    def build_submission_df(self) -> pd.DataFrame:
        """
        build a "submission" dataset for kaggle using the old submission format,
        which was a handy starting point
        """
        teams = pd.read_csv(os.path.join(self.data_dir, f"{self.prefix}Teams.csv"))
        matchups = [
            [id1, id2]
            for id1 in teams.TeamID.values
            for id2 in teams.TeamID.values
            if id1 < id2
        ]
        matchup_df = pd.DataFrame(matchups, columns=["Team1ID", "Team2ID"])
        matchup_df["Season"] = int(self.curr_year)
        matchup_df["ID"] = (
            matchup_df["Team1ID"]
            .apply(str)
            .str.cat(matchup_df["Team2ID"].apply(str), "_")
        )
        matchup_df["ID"] = matchup_df["ID"].apply(lambda x: f"{self.curr_year}_{x}")
        return matchup_df
