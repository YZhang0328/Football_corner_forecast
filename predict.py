"""Train the final model on all labelled data and write predictions.csv."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline

from kalman_features import KF_OBSERVATION_VAR, KF_PROCESS_VAR, build_kalman_features
from utility_functions import FEATURE_COLS, add_cold_start_attack_prior, load_data


DATA_DIR = Path.cwd()
MODEL_FEATURES = FEATURE_COLS + ["home", "kf_attack_filled"]
POISSON_ALPHA = 0.1


labelled_matches, holdout_matches = load_data(DATA_DIR)
print(
    f"Training on {len(labelled_matches) // 2:,} matches "
    f"(seasons {labelled_matches.season.min()}-{labelled_matches.season.max()})"
)
print(f"Predicting  {len(holdout_matches) // 2:,} holdout matches")

# Build the Kalman sequence as labelled history followed by unlabelled holdout.
# Holdout rows receive pre-match states, but their missing corners never update
# the hidden team strengths.
combined_rows = pd.concat([labelled_matches, holdout_matches], ignore_index=True)
combined_rows = build_kalman_features(
    combined_rows,
    process_var=KF_PROCESS_VAR,
    observation_var=KF_OBSERVATION_VAR,
)

train_rows = combined_rows[combined_rows["match_id"].isin(labelled_matches["match_id"])].copy()
holdout_rows = combined_rows[combined_rows["match_id"].isin(holdout_matches["match_id"])].copy()

# Fill only missing kf_attack values. Existing Kalman states are preserved; rows
# without enough team history receive a conservative prior predicted from the
# row features and home/away flag.
cold_start_prior = HistGradientBoostingRegressor(
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=5.0,
    learning_rate=0.03,
    random_state=42,
)
train_known_rows = train_rows.dropna(subset=["kf_attack"]).copy()
train_known_rows, fitted_prior = add_cold_start_attack_prior(
    train_known_rows,
    train_known_rows,
    cold_start_prior,
)
holdout_rows, _ = add_cold_start_attack_prior(
    train_known_rows,
    holdout_rows,
    cold_start_prior,
)

model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("poisson", PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000)),
    ]
)
model.fit(train_known_rows[MODEL_FEATURES], train_known_rows["corners"].to_numpy(float))

holdout_rows = holdout_rows.copy()
holdout_rows["predicted_corners"] = model.predict(holdout_rows[MODEL_FEATURES])

submission = holdout_rows[["match_id", "team", "predicted_corners"]]
submission.to_csv(DATA_DIR / "predictions.csv", index=False)

print(f"\nWritten predictions.csv ({len(submission):,} rows)")
print(f"  mean predicted corners : {submission['predicted_corners'].mean():.3f}")
print(
    "  min / max              : "
    f"{submission['predicted_corners'].min():.3f} / "
    f"{submission['predicted_corners'].max():.3f}"
)
