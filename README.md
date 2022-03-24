## NCAA Basketball Ratings using pymc3

This repo implements a simple offensive/defensive rating system to be used with the Kaggle NCAA Basketball competitions.

The approximate function we're estimating looks like this:

```python 
s = offense[t1_idx] - defense[t2_idx]) * scaler + base
score = Normal(mu=s, sigma=constant)
```

We can add in more terms or try other formulations, but this achieves a simple sanity check of sharing reasonably high
but not perfect rank-correlation with the kenpom.com features for 2022 and to Elo ratings (with and without factoring in score)

```
kendalltau(offensive_ratings, kenpom.adj_o) --> KendalltauResult(correlation=0.7286139425416902, pvalue=1.651697584060781e-93)
kendalltau(defensive_ratings, kenpom.adj_d): KendalltauResult(correlation=-0.7328130783894062, pvalue=1.7001932771500618e-94)
kendalltau(offensive_ratings+defensive_ratings, kenpom.rank): KendalltauResult(correlation=-0.9248418468510999, pvalue=5.588765768618526e-150)
kendalltau(offensive_ratings+defensive_ratings, elo_features): KendalltauResult(correlation=0.7084946338085796, pvalue=7.910211457815486e-89)
```

For future years, the analysis should be extended back as far as the NCAA data goes, and it should include a term for home/away games.


![Estimates of base and scale for 2022](img/base_and_scale_trace.png)
Above is an estimate for the base score and scaling factor. That means that the most likely score estimate is about `(T1Off - T2Def)*0.59 + 69.5`

![Estimates of offense and defense](img/off_def_trace.png)
Above is a plot of the estimated offense and defense ratings for every team.