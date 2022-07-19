## Definitions

This project calculates several features that might be of use for predicting the outcome of NCAA March Madness games. If you're starting
from the kaggle dataset, you can join on `["Season", "TeamName"]` and start modeling, but for an explanation of the fields, read on!


### Probabilistic Model

For the features generated in this section, we use `pymc` to generate a probabilistic model. We aim to understand the relationship between
points scored in a game as a function of a team's offense, the opponent's defense, and the variance in a team's natural game-to-game offensive performance.

To get our dataset, we take all regular season games, remove games with 3 or more overtimes, and subtract from each team's score a fixed number of points
for each overtime. This is a notable difference from [PossessionEfficiencyFactor](#possessionefficiencyfactor) which normalizes differently.

```python
t1_score = pm.Deterministic(
     "score", (offense[t1_idx] - defense[t2_idx]) * scaler + base
)
t1_pts = pm.Normal(
    "t1_points",
    mu=t1_score,
    sigma=team_score_sigma[t1_idx],
    observed=games_df.T1Score,
    dims=("game"),
)
```

To get more consistent results, all seasons share the same baseline score and scaling factor, but are calculated in batches.

For everything in this section with a suffix of `UB` or `LB`, like `OffensiveRatingUB`, these are the upper and lower bounds of the `0.9` HDI of the model.


#### OffensiveRating

The `OffensiveRating` is the first part of the probabilistic model outlined above. A higher `OffensiveRating` means the team's mean score against a team with a given [DefensiveRating](#defensiverating) is higher.

#### DefensiveRating

The `DefensiveRating` is the second part of the probabilistic model outlined above. A higher `DefensiveRating` means a team with a given [OffensiveRating](#offensiverating) tends to score fewer points against them.

#### CombinedRating

This is simply a team's `OffensiveRating + DefensiveRating`. Useful for sorting and the difference between two team's `CombinedRating` makes a strong simplistic baseline model.

#### ScoreVariance

This value estimates how much a team's performance varies when playing two teams with the same [DefensiveRating](#defensiverating)

_____________

### Elo Features

The two Elo features follow a number of the suggestions about calculating Elo scores [from fivethirtyeight](https://fivethirtyeight.com/features/how-our-2015-16-nba-predictions-work/). There are a number of hyperparameters you could choose, like whether Elo ratings roll over each season, whether the end of the season matters more, the value of `k`, etc. These are just two of the combinations that worked well in downstream models.

#### EloWithScore

This value is calculated starting from the first season and rolling over a team's Elo score year-to-year. The update function takes into account score as well, so beating a team by 30 is far better than beating them by 1.

#### EloWinLoss

This value is calculated starting from the first season and rolling over a team's Elo score year-to-year. The update function ignores score.

_____________

### Other Features

The rest of the features aren't grouped neatly, but are either features others have had success with or are simply interesting.

Note that the years for which each feature is provided is a function of which datasets Kaggle has provided per year (the `WNCAARegularSeasonDetailedResults` don't go back past 2010).

#### PossessionEfficiencyFactor

This feature requires you to first estimate the number of possessions in a game - several others have written up good guides on how to do this - here we do the following:

```python
df["ApproxPoss"] = df.apply(
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
```

The point is to get possessions that ended in a field goal without an offensive rebound, add possessions ending in turnovers, add an estimate of how many possessions ended in free throws, and then take the average between the two teams. Once we have that number, we simply divide the points a team scored by this approximate number of possessions to get `points_per_possession`. We then train a linear model on `(t1_id_one_hot, t2_DefensiveRating)` where [DefensiveRating](#defensiverating) is defined above. This gives us an estimate of a team's efficiency adjusted by the strength of the opponent's defense.


#### TempoEstimate

This feature is unlikely to provide value if you already have features that account for both pace and score, but it's interesting and perhaps useful for identifying interesting games. It uses the method outlined in [PossessionEfficiencyFactor](#possessionefficiencyfactor) to calculate approximate possessions, then uses one-hot encoding of teams to predict a game's pace. 