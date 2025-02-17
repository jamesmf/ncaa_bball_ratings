import argparse
import datetime
import json
import typing as T
import os
import logging

import mlflow
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import linregress, kendalltau, spearmanr
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression


from .featurizers import FeatureBase
from .tournament_dataset import df_rename, feature_rename, probabilistic_estimate_df
from .constants import (
    M_PRE_SCALER,
    M_PRE_BASE,
    W_PRE_SCALER,
    W_PRE_BASE,
    OVERTIME_SCORE_BONUS,
)

logging.basicConfig()
# some assumptions gleaned from prior data analysis


# if we want to use predetermined values instead of distributions,
# we can set this flag
USE_PREDETERMINED = True


class PMFeatureGenerator:

    def __init__(
        self,
        prefix: T.Literal["M", "W"],
        max_year: int,
        base_data_dir: str = "./data",
        min_year: int = 1998,
    ):
        self.prefix = prefix
        self.max_year = max_year
        self.min_year = min_year
        self.data_dir = os.path.join(base_data_dir, str(self.max_year))
        self.feature_base = FeatureBase(
            prefix=prefix, start_year=self.min_year, base_path=self.data_dir
        )

    def get_season_list(
        self,
    ):
        return sorted(list(range(self.min_year, self.max_year + 1)), reverse=True)

    def read_in_data(
        self,
        seasons: T.Optional[T.List[int]] = None,
        starting_daynum: int = 1,
    ) -> T.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if seasons is None:
            seasons = self.get_season_list()

        games_df = pd.read_csv(
            f"{self.data_dir}/{self.prefix}RegularSeasonCompactResults.csv"
        )
        teamnames = pd.read_csv(f"{self.data_dir}/{self.prefix}Teams.csv")
        elo_df = self.feature_base.elo_features.reset_index()
        elo_df["elo"] = elo_df["elo_32_day0_True"]
        elo_df["elo_delta"] = elo_df["elo_32_day30_True_21d_diff"]

        games_df = games_df[games_df.Season.isin(seasons)]
        games_df = games_df[games_df.DayNum >= starting_daynum]
        games_df = pd.merge(
            games_df,
            elo_df,
            how="inner",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        games_df["WTeamID"] = games_df[["Season", "WTeamID"]].apply(
            lambda x: "_".join([str(v) for v in x]), axis=1
        )
        games_df = games_df.rename(
            columns={
                "elo": "T1elo",
                "elo_delta": "T1elo_delta",
                "WTeamID": "T1TeamID",
                "WScore": "T1Score",
                "WLoc": "T1Loc",
            }
        ).drop(columns=["TeamID"])
        games_df["T1Wins"] = 1.0

        games_df["T1Home"] = games_df.T1Loc.apply(lambda x: 1 if x == "H" else 0)
        t2home = games_df.T1Loc.apply(lambda x: 1 if x == "A" else 0).values

        games_df = pd.merge(
            games_df,
            elo_df,
            how="inner",
            left_on=["Season", "LTeamID"],
            right_on=["Season", "TeamID"],
        )
        games_df["LTeamID"] = games_df[["Season", "LTeamID"]].apply(
            lambda x: "_".join([str(v) for v in x]), axis=1
        )
        games_df = games_df.rename(
            columns={
                "elo": "T2elo",
                "elo_delta": "T2elo_delta",
                "LTeamID": "T2TeamID",
                "LScore": "T2Score",
            }
        ).drop(columns=["TeamID"])

        renames = {
            c: "T2" + c[2:] for c in games_df.columns if c[:2] == "T1" and c != "T1Home"
        }
        renames.update({c: "T1" + c[2:] for c in games_df.columns if c[:2] == "T2"})
        inv_games_df = games_df.copy().rename(columns=renames)
        inv_games_df["T1Wins"] = 0.0
        inv_games_df["T1Home"] = t2home
        games_df = pd.concat([games_df, inv_games_df])
        games_df["ScoreDiff"] = games_df["T1Score"] - games_df["T2Score"]
        games_df = games_df.drop(columns=["T2Loc", "T2Score"]).reset_index(drop=True)

        games_df = games_df[games_df.NumOT < 2]
        games_df["T1Score"] = games_df[["T1Score", "NumOT"]].apply(
            lambda x: x[0] - OVERTIME_SCORE_BONUS * x[1], axis=1
        )
        return games_df, elo_df, teamnames

    def train_model_all_years(
        self,
        games_df: pd.DataFrame,
        pre_scaler: float,
        pre_base: float,
        seasons: T.Optional[T.List[int]] = None,
        n_samples: int = 500,
        n_tune: int = 2500,
        n_cores: T.Optional[int] = None,
    ) -> az.data.inference_data.InferenceData:
        if seasons is None:
            seasons = self.get_season_list()

        print(f"training models for: {seasons}")

        full_ratings_df = pd.DataFrame()
        for year in seasons:
            if year in games_df.Season.unique():
                print(f"training model for {year}")
                # process each year with past years to lower inter-year variance
                trace = train_model(
                    games_df[
                        games_df.Season.isin((year - 3, year - 2, year - 1, year))
                    ].copy(),
                    pre_scaler,
                    pre_base,
                    n_samples=n_samples,
                    n_tune=n_tune,
                    n_cores=n_cores,
                )
                ratings_df = get_ratings_df(trace)
                ratings_df = ratings_df[ratings_df.Season == year]
                full_ratings_df = pd.concat([full_ratings_df, ratings_df])
            else:
                print(
                    f"skipping training model for {year} because it is not in games_df"
                )

        return full_ratings_df

    def get_df_for_eff(
        self,
    ):
        data_prefix = f"{self.data_dir}/{self.prefix}"
        detailed_df = pd.read_csv(f"{data_prefix}RegularSeasonDetailedResults.csv")
        existing_feature_df = pd.read_csv(
            f"{data_prefix}_data_interim.csv",
        ).set_index(["Season", "TeamID"])
        df_for_eff = pd.merge(
            detailed_df,
            feature_rename(existing_feature_df, "W"),
            how="inner",
            left_on=["Season", "WTeamID"],
            right_index=True,
        )

        df_for_eff = pd.merge(
            df_for_eff,
            feature_rename(existing_feature_df, "L"),
            how="inner",
            left_on=["Season", "LTeamID"],
            right_index=True,
        )

        pos = df_rename(df_for_eff, "W", "L")
        neg = df_rename(df_for_eff, "L", "W")
        df_for_eff = pd.concat((pos, neg))

        df_for_eff["ApproxPoss"] = df_for_eff.apply(
            lambda x: (
                x["T1FGA"]
                - x["T1OR"]
                + x["T1TO"]
                + 0.475 * x["T1FTA"]
                + x["T2FGA"]
                - x["T2OR"]
                + x["T2TO"]
                + 0.475 * x["T2FTA"]
            )
            / 2,
            axis=1,
        )
        df_for_eff["PossAdjForOT"] = df_for_eff.apply(
            lambda x: x["ApproxPoss"] - 7.5 * x["NumOT"],
            axis=1,
        )

        df_for_eff["T1PtsPerPossEstimate"] = df_for_eff.apply(
            lambda x: x["T1Score"] / x["ApproxPoss"],
            axis=1,
        )
        return df_for_eff


def train_model(
    games_df: pd.DataFrame,
    pre_scaler: float,
    pre_base: float,
    n_samples: int = 500,
    n_tune: int = 2500,
    n_cores: T.Optional[int] = None,
) -> az.data.inference_data.InferenceData:
    mean_game_score = int(games_df[games_df.NumOT == 0].T1Score.mean())

    # factorize turns our team IDs into sequential ints like [0, 1, 2, ...]
    t1_idx, teams = pd.factorize(games_df["T1TeamID"], sort=True)
    t2_idx, _ = pd.factorize(games_df["T2TeamID"], sort=True)
    game_ids = games_df.index.values
    home = games_df.T1Home.values

    # shape of this is taken from Rugby Analytics example here:
    # https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html
    coords = {"team": teams, "game": game_ids}

    with pm.Model(coords=coords) as model:
        # constant data
        t1 = pm.Data(
            "t1",
            t1_idx,
            dims="game",
        )
        t2 = pm.Data(
            "t2",
            t2_idx,
            dims="game",
        )

        # keeping this static simplifies other equations, but if ever it
        # needs to be tuned because other variables have changed, uncomment
        # the Uniform and try another run to get a reasonable value. In past
        # runs, it has remained fairly tight
        if USE_PREDETERMINED:
            off_def_sigma = 10
        else:
            off_def_sigma = pm.Uniform("off_def_sigma", lower=7, upper=12)

        # team_score_sigma_mu = pm.Uniform("team_score_sigma_mu", lower=5, upper=15)
        # team_score_sigma_mu = 9.5
        # team_score_sigma_sigma = 2
        # team_score_sigma_sigma = pm.Uniform("team_score_sigma_sigma", lower=0.5, upper=10)
        off_def_rank_mu = 50
        # off_def_rank_mu = pm.Normal("off_def_rank_mu", mu=50, sigma=4)

        # team-specific model parameters
        offense = pm.Normal(
            "offense", mu=off_def_rank_mu, sigma=off_def_sigma, dims="team"
        )
        defense = pm.Normal(
            "defense", mu=off_def_rank_mu, sigma=off_def_sigma, dims="team"
        )
        # team_score_sigma = pm.Normal(
        #     "team_score_sigma",
        #     mu=team_score_sigma_mu,
        #     sigma=team_score_sigma_sigma,
        #     dims="team",
        # )
        team_score_sigma = pm.Uniform(
            "team_score_sigma", lower=5, upper=15, dims="team"
        )
        # home_court_adv = pm.Uniform("home_court_adv", lower=2.5, upper=4.5)
        home_court_adv = 3

        if not USE_PREDETERMINED:
            scaler = pm.Uniform("scaler", lower=0.4, upper=1)
            # base = pm.Uniform("base", lower=mean_game_score - 5, upper=mean_game_score)
            base = pre_base  # just use it, since it's so consistent
        else:
            scaler = pre_scaler
            base = pre_base

        # (offense[t1_idx] - defense[t2_idx]) * scaler + base + home_court_adv*home
        t1_score = pm.Deterministic(
            "score",
            (offense[t1_idx] - defense[t2_idx]) * scaler + base + home_court_adv * home,
        )

        # likelihood of observed data
        t1_pts = pm.Normal(
            "t1_points",
            mu=t1_score,
            sigma=team_score_sigma[t1_idx],
            observed=games_df.T1Score,
            dims=("game"),
        )
        #         t1_wins = pm.Bernoulli("t1_wins", p=t1_win_prob, observed=games_df.T1Wins)

        trace = pm.sample(
            n_samples,
            tune=n_tune,
            cores=n_cores if n_cores else os.cpu_count(),
            return_inferencedata=True,
            target_accept=0.9,
        )
    return trace


def get_ratings_df(trace: az.data.inference_data.InferenceData) -> pd.DataFrame:
    trace_hdi = az.hdi(trace, hdi_prob=0.9)
    for variable in (
        "home_court_adv",
        "off_def_sigma",
        "scaler",
        "base",
        "team_score_sigma",
        "off_def_rank_mu",
    ):
        try:
            print(f"{variable}: {trace_hdi[variable].values}")
        except:
            pass

    ratings_df = pd.DataFrame(
        list(
            zip(
                trace_hdi["offense"].team.values,
                trace_hdi["offense"].values[:, 0],
                trace_hdi["offense"].values[:, 1],
                trace_hdi["defense"].values[:, 0],
                trace_hdi["defense"].values[:, 1],
                trace_hdi["team_score_sigma"].values[:, 0],
                trace_hdi["team_score_sigma"].values[:, 1],
            )
        ),
        columns=[
            "FullTeamID",
            "OffensiveRatingLB",
            "OffensiveRatingUB",
            "DefensiveRatingLB",
            "DefensiveRatingUB",
            "ScoreVarianceLB",
            "ScoreVarianceUB",
        ],
    )
    ratings_df["OffensiveRating"] = (
        trace_hdi["offense"].values[:, 0] + trace_hdi["offense"].values[:, 1]
    ) / 2
    ratings_df["DefensiveRating"] = (
        trace_hdi["defense"].values[:, 0] + trace_hdi["defense"].values[:, 1]
    ) / 2
    ratings_df["ScoreVariance"] = (
        trace_hdi["team_score_sigma"].values[:, 0]
        + trace_hdi["team_score_sigma"].values[:, 1]
    ) / 2

    ratings_df["Season"] = ratings_df["FullTeamID"].apply(
        lambda x: int(x.split("_")[0])
    )
    ratings_df["TeamID"] = ratings_df["FullTeamID"].apply(
        lambda x: int(x.split("_")[1])
    )

    return ratings_df


def join_datasets(
    full_ratings_df: pd.DataFrame, elo_df: pd.DataFrame, teamnames: pd.DataFrame
) -> T.Tuple[pd.DataFrame, pd.DataFrame]:
    joined = pd.merge(
        full_ratings_df,
        elo_df,
        how="inner",
        left_on=["Season", "TeamID"],
        right_on=["Season", "TeamID"],
    )
    joined = pd.merge(
        joined, teamnames, how="left", left_on=["TeamID"], right_on=["TeamID"]
    )

    joined = joined.rename(
        columns={
            "elo_32_day0_True": "EloWithScore",
            "elo_32_day0_False": "EloWinLoss",
            "elo_64_day30_True": "EloDay30WithScore",
            "elo_64_day30_False": "EloDay30WinLoss",
            "elo_delta": "EloDelta21Days",
        }
    )
    joined["CombinedRating"] = joined["OffensiveRating"] + joined["DefensiveRating"]
    output_df = joined[
        [
            "Season",
            "TeamName",
            "TeamID",
            "CombinedRating",
            "OffensiveRating",
            "DefensiveRating",
            "ScoreVariance",
            "EloWithScore",
            "EloWinLoss",
            "EloDelta21Days",
            "EloDay30WithScore",
            "EloDay30WinLoss",
        ]
    ].copy()
    float_cols = output_df.select_dtypes(float).columns
    for fc in float_cols:
        output_df[fc] = output_df[fc].apply(np.round, args=[1])
    return output_df, joined


def create_pace_feature(df: pd.DataFrame, team_id_map: T.Dict[int, int]):
    data, rows, cols, y = [], [], [], []
    n = 0
    for season, t1, t2, poss_adj in df[
        ["Season", "T1TeamID", "T2TeamID", "PossAdjForOT"]
    ].values:
        ind1 = team_id_map[f"{int(season)}_{int(t1)}"]
        ind2 = team_id_map[f"{int(season)}_{int(t2)}"]
        rows.extend([n, n])
        cols.extend([ind1, ind2])
        data.extend([1, 1])
        y.append(poss_adj)
        n += 1
    x = csr_matrix((data, (rows, cols)), shape=(n, len(team_id_map)))
    lr = LinearRegression().fit(x, y)

    inv = {v: k for k, v in team_id_map.items()}
    pace_data = []
    for n, coef in enumerate(lr.coef_):
        season_team_id = inv[n]
        season, team_id = [int(i) for i in season_team_id.split("_")]
        pace_data.append([season, team_id, np.round(lr.intercept_ + coef, 1)])
    return pd.DataFrame(pace_data, columns=["Season", "TeamID", "TempoEstimate"])


def create_est_pts_per_poss_feature(df: pd.DataFrame, team_id_map: T.Dict[str, int]):
    data, rows, cols, y = [], [], [], []
    n_teams = len(team_id_map)
    n = 0
    for season, t1, t2def, pts_per_estimate in df[
        ["Season", "T1TeamID", "T2DefensiveRating", "T1PtsPerPossEstimate"]
    ].values:
        ind1 = team_id_map[f"{int(season)}_{int(t1)}"]
        rows.extend([n, n])
        cols.extend([ind1, n_teams])
        data.extend([1, t2def])
        y.append(pts_per_estimate)
        n += 1
    x = csr_matrix((data, (rows, cols)), shape=(n, n_teams + 1))
    lr = LinearRegression().fit(x, y)
    preds = lr.predict(x)
    r2 = linregress(y, preds).rvalue ** 2
    kt = kendalltau(y, preds)[0]
    print(lr.intercept_, lr.coef_[-1], r2, kt)

    inv = {v: k for k, v in team_id_map.items()}
    pace_data = []
    for n, coef in enumerate(lr.coef_[:-1]):
        season_team_id = inv[n]
        season, team_id = [int(i) for i in season_team_id.split("_")]
        pace_data.append([season, team_id, np.round(lr.intercept_ + coef, 2)])
    return pd.DataFrame(
        pace_data, columns=["Season", "TeamID", "PossessionEfficiencyFactor"]
    )


def create_win_prob_features(
    input_df: pd.DataFrame, prefix: str, n: int = 16
) -> pd.DataFrame:
    orig_df = input_df.reset_index()
    offensive_good_team_rating = (
        orig_df.groupby("Season")["OffensiveRating"]
        .agg(lambda x: sorted(x, reverse=True)[n])
        .reset_index()
        .rename(columns={"OffensiveRating": "T2OffensiveRating"})
    )
    defensive_good_team_rating = (
        orig_df.groupby("Season")["DefensiveRating"]
        .agg(lambda x: sorted(x, reverse=True)[n])
        .reset_index()
        .rename(columns={"DefensiveRating": "T2DefensiveRating"})
    )
    score_variance = orig_df.ScoreVariance.median()

    df_for_combined_rating = orig_df[
        [
            "TeamID",
            "TeamName",
            "Season",
            "CombinedRating",
            "OffensiveRating",
            "DefensiveRating",
            "ScoreVariance",
        ]
    ].rename(
        columns={
            "OffensiveRating": "T1OffensiveRating",
            "DefensiveRating": "T1DefensiveRating",
            "ScoreVariance": "T1ScoreVariance",
        }
    )
    df_for_combined_rating = pd.merge(
        df_for_combined_rating, offensive_good_team_rating, how="left", on="Season"
    )
    df_for_combined_rating = pd.merge(
        df_for_combined_rating, defensive_good_team_rating, how="left", on="Season"
    )
    df_for_combined_rating["T2ScoreVariance"] = score_variance
    if prefix == "W":
        df_for_combined_rating["WinProbAgainstGoodTeam"] = probabilistic_estimate_df(
            df_for_combined_rating, base=W_PRE_BASE, scaler=W_PRE_SCALER
        )
    else:
        df_for_combined_rating["WinProbAgainstGoodTeam"] = probabilistic_estimate_df(
            df_for_combined_rating, base=M_PRE_BASE, scaler=M_PRE_SCALER
        )
    return df_for_combined_rating[
        ["Season", "TeamID", "WinProbAgainstGoodTeam"]
    ].rename(columns={"WinProbAgainstGoodTeam": f"WP{n}"})


def get_full_features(
    df_for_eff: pd.DataFrame,
    existing_feature_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    team_map = {}
    for team in (
        df_for_eff[["Season", "T1TeamID"]]
        .apply(lambda x: f"{x[0]}_{x[1]}", axis=1)
        .unique()
    ):
        team_map[team] = len(team_map)

    pace_df = create_pace_feature(df_for_eff, team_map)
    pts_per_poss_df = create_est_pts_per_poss_feature(df_for_eff, team_map)
    wp_features = create_win_prob_features(existing_feature_df, prefix).set_index(
        ["Season", "TeamID"]
    )
    new_features = pd.merge(
        existing_feature_df,
        pts_per_poss_df,
        how="left",
        right_on=["Season", "TeamID"],
        left_index=True,
        suffixes=("_old", ""),
    ).set_index(["Season", "TeamID"])
    new_features = pd.merge(
        new_features,
        pace_df,
        how="left",
        right_on=["Season", "TeamID"],
        left_index=True,
        suffixes=("_old", ""),
    )
    new_features = pd.merge(
        new_features,
        wp_features,
        how="left",
        right_index=True,
        left_on=["Season", "TeamID"],
        suffixes=("_old", ""),
    )
    return new_features


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-year", type=int, default=datetime.date.today().year)
    ap.add_argument("--target-season", type=int, required=False)
    args = ap.parse_args()

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"feature-generation-{args.max_year}")
    mlflow.autolog(disable=True)

    max_season = args.max_year
    output_feature_names = [
        "Season",
        "TeamName",
        "TeamID",
        "WP16",
        "CombinedRating",
        "OffensiveRating",
        "DefensiveRating",
        "EloWithScore",
        "EloWinLoss",
        "EloDelta21Days",
        "PossessionEfficiencyFactor",
        "TempoEstimate",
        "ScoreVariance",
        "EloDay30WithScore",
        "EloDay30WinLoss",
    ]

    with open("output/build_data.json", "w") as f:
        json.dump(
            {
                "build_date": datetime.date.today().strftime("%Y-%m-%d"),
                "data_date": f" - {args.max_year} Season - Day 132",
            },
            f,
        )

    prefixes = ("M", "W")
    output_features = {}
    for prefix in prefixes:
        feature_generator = PMFeatureGenerator(
            prefix=prefix,
            max_year=max_season,
            base_data_dir="./data/",
        )
        # def generate_all_season_features(prefix: str, seasons: T.Optional[T.List[int]], starting_daynum: int=0):
        games_df, elo_df, teamnames = feature_generator.read_in_data()
        # print(f"Max DayNum {games_df[games_df.Season == games_df.Season.max()].DayNum.max()}")

        if prefix == "M":
            pre_scaler = M_PRE_SCALER
            pre_base = M_PRE_BASE
        else:
            pre_scaler = W_PRE_SCALER
            pre_base = W_PRE_BASE
        seasons = (
            [args.target_season]
            if args.target_season is not None
            else feature_generator.get_season_list()
        )
        ratings_df = feature_generator.train_model_all_years(
            games_df,
            pre_scaler=pre_scaler,
            pre_base=pre_base,
            seasons=seasons,
            n_cores=11,
        )
        output, joined = join_datasets(ratings_df, elo_df, teamnames)

        output.to_csv(
            f"{feature_generator.data_dir}/{prefix}_data_interim.csv", index=False
        )
        joined.to_csv(
            f"{feature_generator.data_dir}/{prefix}_features_interim.csv", index=False
        )

        df_for_eff = feature_generator.get_df_for_eff()
        new_features = get_full_features(
            df_for_eff, output.set_index(["Season", "TeamID"]), prefix
        )
        new_features = new_features[output_feature_names]
        new_features.to_csv(f"output/{prefix}_data_complete.csv", index=False)
        mlflow.log_artifact(f"output/{prefix}_data_complete.csv")
        output_features[prefix] = new_features

        orig_df = pd.read_csv(f"output/{prefix}_data_complete.csv")
        team_df = pd.read_csv(
            f"{feature_generator.data_dir}/{prefix}Teams.csv",
            usecols=["TeamName", "TeamID"],
        )
        joined = pd.merge(
            orig_df, team_df, how="inner", on=["TeamName"], suffixes=("", "_y")
        ).drop(columns=["TeamID_y"])
        print(joined.shape, orig_df.shape)
        joined.to_csv(f"output/{prefix}_data_complete.csv", index=False)
