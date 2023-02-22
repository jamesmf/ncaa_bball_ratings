import os
import typing as T

import numpy as np
import pandas as pd

from .basic_elo import Elo


class FeatureBase:
    # core_cols_w = [
    #     "WFGM",
    #     "WFGA",
    #     "WFGM3",
    #     "WFGA3",
    #     "WFTM",
    #     "WFTA",
    #     "WOR",
    #     "WDR",
    #     "WAst",
    #     "WTO",
    #     "WStl",
    #     "WBlk",
    #     "WPF",
    #     "WPt_diff",
    #     "WWin",
    # ]
    core_cols_w = [
        "WPt_diff",
        "WWin",
    ]
    core_cols_l = ["L" + i[1:] for i in core_cols_w]

    def __init__(
        self,
        prefix: str,
        base_path: str = "data/",
        start_year: int = 2000,
        elo_ks: T.List[int] = [32, 64],
        min_day_nums: T.List[int] = [30, 0],
        elo_verbose: bool = False,
    ):
        self.base_path = base_path
        self.prefix = prefix
        self.elo_ks = elo_ks
        self.min_day_nums = min_day_nums
        self.elo_verbose = elo_verbose
        self.season_df = pd.read_csv(
            os.path.join(self.base_path, f"{prefix}RegularSeasonCompactResults.csv")
        )
        self.season_df = self.season_df[self.season_df.Season >= start_year]
        self.season_df["WPt_diff"] = self.season_df["WScore"] - self.season_df["LScore"]
        self.season_df["LPt_diff"] = self.season_df["LScore"] - self.season_df["WScore"]
        self.season_df["WWin"] = 1
        self.season_df["LWin"] = 0
        wins = self.season_df[self.core_cols_w + ["WTeamID", "Season", "DayNum"]].copy()
        wins["OTeamID"] = self.season_df.LTeamID.values
        wins = wins.rename(columns={c: c[1:] for c in wins.columns if c[0] == "W"})
        losses = self.season_df[
            self.core_cols_l + ["LTeamID", "Season", "DayNum"]
        ].copy()
        losses["OTeamID"] = self.season_df.WTeamID.values
        losses = losses.rename(
            columns={c: c[1:] for c in losses.columns if c[0] == "L"}
        )
        self.stats = pd.concat([wins, losses])
        self.base = self.stats[["Season", "TeamID"]].drop_duplicates()

        self.elo_features = self.get_elo_feature_df()
        self.base = pd.merge(
            self.base,
            self.elo_features,
            how="left",
            left_on=["Season", "TeamID"],
            right_index=True,
        )

        self.adjusted = self.get_elo_adjusted_game_features()
        self.base = pd.merge(
            self.base,
            self.adjusted,
            how="left",
            left_on=["Season", "TeamID"],
            right_index=True,
        )

        self.unadjusted = self.get_unadjusted_stats()
        self.base = pd.merge(
            self.base,
            self.unadjusted,
            how="left",
            left_on=["Season", "TeamID"],
            right_index=True,
        )

        self.pom = self.get_pom()
        if self.pom is not None:
            self.base = pd.merge(
                self.base,
                self.pom,
                how="left",
                left_on=["Season", "TeamID"],
                right_index=True,
            )

        self.base = self.base.drop(
            columns=[
                "OTeamID",
                "Pt_diff",
            ]
        )

    def get_elo(self, k: int, min_day_num: int = 0, use_score: bool = True):
        w_teams = self.season_df[["Season", "WTeamID"]].drop_duplicates().values
        l_teams = self.season_df[["Season", "LTeamID"]].drop_duplicates().values
        teams = list(
            set(
                [
                    f"{season}_{team}"
                    for teams in (w_teams, l_teams)
                    for season, team in teams
                ]
            )
        )
        initial_ratings: T.Dict[str, int] = {}
        all_ratings: T.Dict[str, int] = {}
        three_week_ratings: T.Dict[str, int] = {}  # ratings 21 days prior to season end
        years = sorted(self.season_df.Season.unique())
        for year_ind, year in enumerate(years):
            reached_recent_cutoff = False
            elo = Elo(
                k=k,
                players=[t for t in teams if t[:4] == str(year)],
                use_score=use_score,
                verbose=self.elo_verbose,
                last_season=initial_ratings,
            )
            max_day = self.season_df[self.season_df.Season == year].DayNum.max()
            for row in (
                self.season_df[
                    (self.season_df.DayNum > min_day_num)
                    & (self.season_df.Season == year)
                ]
                .sort_values("DayNum")[
                    ["Season", "DayNum", "WTeamID", "LTeamID", "WPt_diff"]
                ]
                .values
            ):
                season, daynum, wteam, lteam, score = row
                if daynum > (max_day - 21) and not reached_recent_cutoff:
                    reached_recent_cutoff = True
                    three_week_ratings.update(elo.ratings.copy())
                winner = f"{season}_{wteam}"
                loser = f"{season}_{lteam}"
                elo.observe_game(winner, loser, score=score)

            all_ratings.update(elo.ratings)
            if year_ind + 1 < len(years):
                for key, value in elo.ratings.items():
                    new_id = f"{years[year_ind+1]}_{key[5:]}"
                    initial_ratings[new_id] = value
            # if self.elo_verbose:
            #     print(f"completed elo for season {year}")
            #     print(sorted(elo.ratings.items(), key=lambda x: x[1]))
        return all_ratings, three_week_ratings

    def get_elo_feature_df(self) -> pd.DataFrame:
        data = []
        for k in self.elo_ks:
            for min_day_num in self.min_day_nums:
                for use_score in (True, False):
                    ratings, three_week_ratings = self.get_elo(
                        k, min_day_num=min_day_num, use_score=use_score
                    )
                    for key, rating in ratings.items():
                        season, team_id = key.split("_")
                        row = [
                            int(season),
                            int(team_id),
                            f"elo_{k}_day{min_day_num}_{use_score}",
                            rating,
                        ]
                        data.append(row)
                        three_weeks_ago_elo = three_week_ratings.get(key, rating)
                        three_week_diff = rating - three_weeks_ago_elo
                        data.append(
                            [
                                int(season),
                                int(team_id),
                                f"elo_{k}_day{min_day_num}_{use_score}_21d_diff",
                                three_week_diff,
                            ]
                        )

        df = pd.DataFrame(data, columns=["Season", "TeamID", "col", "rating"])
        df = df.pivot(index=["Season", "TeamID"], columns=["col"])["rating"]
        print(df.columns)
        return df

    def get_elo_adjusted_game_features(self) -> pd.DataFrame:
        """create elo-adjusted features for a team's season. uses the
        first k and daynum values in each array

        Returns:
            pd.DataFrame:
        """
        elo_col = f"elo_{self.elo_ks[0]}_day{self.min_day_nums[0]}_True"
        elo_series = self.elo_features[elo_col]
        elo_max = elo_series.max()
        elo_min = elo_series.min()
        elo_div = elo_max - elo_min
        elo_series = (
            elo_series.apply(lambda x: np.max([x, elo_min])) - elo_min
        ) / elo_div
        stats_adj = pd.merge(
            self.stats,
            elo_series,
            how="left",
            left_on=["Season", "TeamID"],
            right_index=True,
        )
        stats_adj["Pt_adj"] = (
            stats_adj["Pt_diff"].apply(lambda x: np.max([x, 0])) * stats_adj[elo_col]
        )
        stats_adj["Win_adj"] = stats_adj["Win"] * stats_adj[elo_col]
        stats_adj = stats_adj.rename(columns={elo_col: "OppElo"})
        grouped = (
            stats_adj[["Season", "TeamID", "Pt_adj", "Win_adj", "OppElo"]]
            .groupby(["Season", "TeamID"])
            .mean()
        )

        wins_after_60 = (
            pd.merge(
                self.season_df[self.season_df["DayNum"] > 60],
                self.elo_features,
                how="left",
                left_on=["Season", "LTeamID"],
                right_index=True,
            )[["Season", "WTeamID", "elo_32_day0_True"]]
            .groupby(["Season", "WTeamID"])
            .max()
            .reset_index()
            .rename(
                columns={
                    "WTeamID": "TeamID",
                    "elo_32_day0_True": "best_team_beaten_elo",
                }
            )
            .set_index(["Season", "TeamID"])
        )
        wins_after_60_min = wins_after_60["best_team_beaten_elo"].min()
        wins_after_60 = wins_after_60.fillna(wins_after_60_min)
        grouped = pd.merge(
            grouped, wins_after_60, how="left", left_index=True, right_index=True
        )

        return grouped

    def get_unadjusted_stats(self, min_day_num: int = 30) -> pd.DataFrame:
        """Get the mean of the stats provided in the dataset

        Args:
            min_day_num (int, optional): exclude data from before this day. Defaults to 30.
        Returns:
            pd.DataFrame: raw means of game stats for all games after min_day_num
        """
        stats = (
            self.stats[self.stats.DayNum > min_day_num].copy().drop(columns=["DayNum"])
        )
        return stats.groupby(["Season", "TeamID"]).mean()

    def get_pom(self):
        """get the ken pom features if they exist"""
        pom_path = os.path.join(self.base_path, "kenpom.csv")
        if os.path.exists(pom_path):
            pom = pd.read_csv(pom_path).set_index(["Season", "TeamID"])
        else:
            pom = None
        return pom

    def persist(self, path: str):
        """save features to csv

        Args:
            path (str): output path
        """
        self.base.to_csv(path, index=False)
