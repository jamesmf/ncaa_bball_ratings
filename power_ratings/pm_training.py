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
from scipy.sparse import csr
from sklearn.linear_model import LinearRegression


from .featurizers import FeatureBase
from .tournament_dataset import df_rename, feature_rename


logging.basicConfig()
# some assumptions gleaned from prior data analysis

# an overtime adds about 7-9 points on average
OVERTIME_SCORE_BONUS = 7.4

# if we want to use predetermined values instead of distributions,
# we can set this flag
USE_PREDETERMINED = True


M_PRE_SCALER = 0.59
M_PRE_BASE = 69.5
W_PRE_SCALER = 0.79
W_PRE_BASE = 64.0

# params = {
#     "overtime_score_discount": OVERTIME_SCORE_BONUS,
#     "season": f"[{','.join([str(i) for i in SEASONS])}]",
#     "data_prefix": DATA_PREFIX,
#     "use_predetermined": USE_PREDETERMINED,
#     "pre_scaler": pre_scaler,
#     "pre_base": pre_base,
# }


def get_season_list():
    return sorted(list(range(2000, 2024)), reverse=True)


def read_in_data(
    prefix: str, seasons: T.Optional[T.List[int]] = None
) -> T.Tuple[pd.DataFrame, pd.DataFrame]:

    if seasons is None:
        seasons = get_season_list()

    feature_base = FeatureBase(prefix=prefix)
    games_df = pd.read_csv(f"./data/{prefix}RegularSeasonCompactResults.csv")
    teamnames = pd.read_csv(f"data/{prefix}Teams.csv")
    elo_df = feature_base.elo_features.reset_index()
    elo_df["elo"] = elo_df["elo_32_day0_True"]
    elo_df["elo_delta"] = elo_df["elo_32_day0_True_21d_diff"]

    games_df = games_df[games_df.Season.isin(seasons)]
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


def train_model(
    games_df: pd.DataFrame, pre_scaler: float, pre_base: float
) -> az.data.inference_data.InferenceData:
    mean_game_score = int(games_df[games_df.NumOT == 0].T1Score.mean())

    # factorize turns our team IDs into sequential ints like [0, 1, 2, ...]
    t1_idx, teams = pd.factorize(games_df["T1TeamID"], sort=True)
    t2_idx, _ = pd.factorize(games_df["T2TeamID"], sort=True)
    game_ids = games_df.index.values

    # shape of this is taken from Rugby Analytics example here:
    # https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html
    coords = {"team": teams, "game": game_ids}

    with pm.Model(coords=coords) as model:
        # constant data
        t1 = pm.Data("t1", t1_idx, dims="game", mutable=True)
        t2 = pm.Data("t2", t2_idx, dims="game", mutable=True)

        off_sigma, def_sigma = 10, 10

        # team-specific model parameters
        offense = pm.Normal("offense", mu=50, sigma=off_sigma, dims="team")
        defense = pm.Normal("defense", mu=50, sigma=def_sigma, dims="team")
        team_score_sigma = pm.Normal("team_score_sigma", mu=9.5, sigma=2, dims="team")

        if not USE_PREDETERMINED:
            scaler = pm.Normal("scaler", mu=0.5, sigma=0.3)
            base = pm.Normal("base", mu=mean_game_score - 5, sigma=2)
        else:
            scaler = pre_scaler
            base = pre_base

        t1_score = pm.Deterministic(
            "score", (offense[t1_idx] - defense[t2_idx]) * scaler + base
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
            750,
            tune=1000,
            cores=11,
            return_inferencedata=True,
            target_accept=0.9,
        )
    return trace


def get_ratings_df(trace: az.data.inference_data.InferenceData) -> pd.DataFrame:
    trace_hdi = az.hdi(trace, hdi_prob=0.9)
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


def train_model_all_years(
    games_df: pd.DataFrame,
    pre_scaler: float,
    pre_base: float,
    seasons: T.Optional[T.List[int]] = None,
) -> az.data.inference_data.InferenceData:
    if seasons is None:
        seasons = get_season_list()

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
            )
            ratings_df = get_ratings_df(trace)
            ratings_df = ratings_df[ratings_df.Season == year]
            full_ratings_df = pd.concat([full_ratings_df, ratings_df])
        else:
            print(f"skipping training model for {year} because it is not in games_df")

    return full_ratings_df


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
            "elo_32_day30_True": "EloWithScore",
            "elo_32_day0_False": "EloWinLoss",
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
        ]
    ].copy()
    float_cols = output_df.select_dtypes(float).columns
    for fc in float_cols:
        output_df[fc] = output_df[fc].apply(np.round, args=[1])
    return output_df, joined


def get_df_for_eff(prefix: str):

    detailed_df = pd.read_csv(f"./data/{prefix}RegularSeasonDetailedResults.csv")
    existing_feature_df = pd.read_csv(
        f"data/{prefix}_data_interim.csv",
    ).set_index(["Season", "TeamID"])
    print(detailed_df)
    print(existing_feature_df)
    df_for_eff = pd.merge(
        detailed_df,
        feature_rename(existing_feature_df, "W"),
        how="inner",
        left_on=["Season", "WTeamID"],
        right_index=True,
    )
    print(df_for_eff)
    df_for_eff = pd.merge(
        df_for_eff,
        feature_rename(existing_feature_df, "L"),
        how="inner",
        left_on=["Season", "LTeamID"],
        right_index=True,
    )
    print(df_for_eff)
    pos = df_rename(df_for_eff, "W", "L")
    neg = df_rename(df_for_eff, "L", "W")
    df_for_eff = pd.concat((pos, neg))
    print(df_for_eff)

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
    x = csr.csr_matrix((data, (rows, cols)), shape=(n, len(team_id_map)))
    lr = LinearRegression().fit(x, y)

    inv = {v: k for k, v in team_id_map.items()}
    pace_data = []
    for n, coef in enumerate(lr.coef_):
        season_team_id = inv[n]
        season, team_id = [int(i) for i in season_team_id.split("_")]
        pace_data.append([season, team_id, np.round(lr.intercept_ + coef, 1)])
    return pd.DataFrame(pace_data, columns=["Season", "TeamID", "TempoEstimate"])


def create_est_pts_per_poss_feature(df: pd.DataFrame, team_id_map: T.Dict[int, int]):
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
    x = csr.csr_matrix((data, (rows, cols)), shape=(n, n_teams + 1))
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


def get_full_features(
    df_for_eff: pd.DataFrame, existing_feature_df: pd.DataFrame
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
    return new_features
