# Corner Forecasting Report

## Final Model

The final model is:

```text
features = feature_1..feature_10 + home + kf_attack_filled
model    = PoissonRegressor(alpha=0.1)
```

`kf_attack` is a causal Kalman-filter estimate of a team's attacking corner
strength before the match. When it is missing because a team or opponent has
not reached the KF warm-up threshold, a small Ridge prior model predicts only
that missing team-strength value. Existing KF values are never overwritten.

## Kalman Logic In Plain Language

Raw corners mix three effects:

```text
own attack + opponent defensive weakness + home advantage
```

So before updating a team's attack state, the filter subtracts the opponent
defence estimate and, for the home team, the league home advantage. The
remaining value is treated as a cleaner observation of that team's attack.
Defence is updated with the same idea from the opponent's corners.

Matches on the same date are batched: all pre-match features are recorded
first, then that date's results update the states. This avoids same-day leakage.

## Validation Protocol

Model choices are made on:

```text
CORE: 2012-2017
TUNE: 2018-2019
```

The final check is one-shot:

```text
TRAIN: 2012-2019
VAL:   2020-2022
```

Final validation results:

```text
Baseline                         MAE=2.2717  RMSE=2.8574  PoissonDev=1.3406
Poisson + mean-imputed kf_attack MAE=2.1712  RMSE=2.7349  PoissonDev=1.2574
Poisson + cold-start KF prior    MAE=2.1588  RMSE=2.7196  PoissonDev=1.2437
```

The cold-start prior is trained only on rows where KF already produced a real
`kf_attack`. It uses `feature_1..10` and `home` to estimate a conservative
initial attack strength for rows that would otherwise receive one global mean
value. A regularised `HistGradientBoostingRegressor` gave a small but consistent
gain over the linear Ridge prior, while still only filling missing KF values.

## Removed Experiments

The following branches were tested but removed from the final code because they
did not improve the honest validation result robustly:

- Tweedie GLM tuning.
- Joint Residual Kalman with home/away attack/defence states.
- Rolling features.
- Static team target encoding.
- Team/opponent/season identity feature checks.
- Frozen home/away drift features.
- Home/away multiplicative calibration.
- Semi-linear spline/GAM-style Poisson checks.

The retained cold-start prior is deliberately narrow: it improves only missing
KF attack values and leaves the main Kalman/Poisson model simple.
