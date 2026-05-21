"""Train the final model on all labelled data and write holdout predictions."""

from pathlib import Path

import pandas as pd

from kalman_features import KF_OBSERVATION_VAR, KF_PROCESS_VAR, build_kalman_features
from utility_functions import (
    FINAL_MODEL_FEATURES,
    fill_missing_kf_attack_with_prior,
    load_data,
    make_cold_start_prior_model,
    make_corner_model,
)


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "inputs"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main() -> None:
    labelled_rows, holdout_rows = load_data(INPUT_DIR)
    print(
        f"Training on {len(labelled_rows) // 2:,} matches "
        f"(seasons {labelled_rows.season.min()}-{labelled_rows.season.max()})"
    )
    print(f"Predicting  {len(holdout_rows) // 2:,} holdout matches")

    # Labelled rows update Kalman states. Holdout rows have missing corners, so
    # they receive pre-match states.
    kalman_input_rows = pd.concat([labelled_rows, holdout_rows], ignore_index=True)
    kalman_rows = build_kalman_features(
        kalman_input_rows,
        process_var=KF_PROCESS_VAR,
        observation_var=KF_OBSERVATION_VAR,
    )

    train_rows = kalman_rows[
        kalman_rows["match_id"].isin(labelled_rows["match_id"])
    ].copy()
    prediction_rows = kalman_rows[
        kalman_rows["match_id"].isin(holdout_rows["match_id"])
    ].copy()

    # Train the corner model on rows with real KF history. For holdout cold-start
    # rows, fill only kf_attack using a small supervised prior.
    known_attack_train_rows = train_rows.dropna(subset=["kf_attack"]).copy()
    known_attack_train_rows, _ = fill_missing_kf_attack_with_prior(
        known_attack_train_rows,
        known_attack_train_rows,
        make_cold_start_prior_model(),
    )
    prediction_rows, _ = fill_missing_kf_attack_with_prior(
        known_attack_train_rows,
        prediction_rows,
        make_cold_start_prior_model(),
    )

    # Final Poisson model: predicts expected corners from base features plus filled KF attack.
    corner_model = make_corner_model()
    corner_model.fit(
        known_attack_train_rows[FINAL_MODEL_FEATURES],
        known_attack_train_rows["corners"].to_numpy(float),
    )

    prediction_rows = prediction_rows.copy()
    prediction_rows["predicted_corners"] = corner_model.predict(
        prediction_rows[FINAL_MODEL_FEATURES]
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission = prediction_rows[["match_id", "team", "predicted_corners"]]
    output_path = OUTPUT_DIR / "predictions.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nWritten {output_path} ({len(submission):,} rows)")
    print(f"  mean predicted corners : {submission['predicted_corners'].mean():.3f}")
    print(
        "  min / max              : "
        f"{submission['predicted_corners'].min():.3f} / "
        f"{submission['predicted_corners'].max():.3f}"
    )


if __name__ == "__main__":
    main()
