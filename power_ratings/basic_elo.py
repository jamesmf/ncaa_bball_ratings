import numpy as np

import typing as T


# from https://github.com/ddm7018/Elo/blob/master/elosports/elo.py plus 538's posts
class Elo:
    def __init__(
        self,
        k,
        g=1,
        starting_score: int = 1500,
        players: T.List[str] = [],
        last_season: T.Dict[str, int] = {},
        use_score: bool = True,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.use_score = use_score
        self.ratings = {k: starting_score for k in players}
        for team, score in last_season.items():
            if team in self.ratings:
                value = int((2 / 3) * score + (1 / 3) * starting_score)
                self.ratings[team] = value
        self.k = k
        self.g = g

    def observe_game(self, winner: str, loser: str, score: T.Optional[int] = None):
        """Observe a game between two teams specifying the winner and loser

        Args:
            winner (str): winner id
            loser (str): loser id
        """
        result = self.get_expected_result(self.ratings[winner], self.ratings[loser])
        multiplier = 1.0
        if score is not None and self.use_score:
            multiplier = self.get_expected_score_factor(
                self.ratings[winner], self.ratings[loser], score
            )
        if self.verbose:
            print(
                f"{self.ratings[winner]} beat {self.ratings[loser]} by {score}: {self.k} * {result} * {multiplier}"
            )
        # result = result * multiplier

        self.ratings[winner] = (
            self.ratings[winner] + (self.k * self.g) * (1 - result) * multiplier
        )
        self.ratings[loser] = self.ratings[loser] + (self.k * self.g) * (
            0 - (1 - result) * multiplier
        )
        if self.verbose:
            print(f"  new ratings: {self.ratings[winner]} and {self.ratings[loser]}")

    def get_expected_result(self, p1, p2):
        exp = (p2 - p1) / 400.0
        return 1 / ((10.0 ** (exp)) + 1)

    def get_expected_score_factor(self, elow: int, elol: int, score: int) -> float:
        return np.log(abs(score) + 1) * (2.2 / ((elow - elol) * 0.001 + 2.2))
