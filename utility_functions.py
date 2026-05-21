from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error


# Season protocol:
# CORE  : fit models while choosing hyperparameters.
# TUNE  : choose hyperparameters only. Never use VAL here.
# TRAIN : full labelled history used for the final validation fit.
# VAL   : final one-shot validation block.
CORE_SEASONS = list(range(2012, 2018))
TUNE_SEASONS = [2018, 2019]
TRAIN_SEASONS = list(range(2012, 2020))
VAL_SEASONS = [2020, 2021, 2022]

FEATURE_COLS = [f"feature_{i}" for i in range(1, 11)]
COLD_START_PRIOR_FEATURES = FEATURE_COLS + ["home"]


def load_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labelled match rows and unlabelled holdout rows."""
    root = data_dir or Path.cwd()
    labelled = pd.read_csv(root / "match_data.csv", parse_dates=["date"])
    holdout = pd.read_csv(root / "match_data_holdout_features.csv", parse_dates=["date"])
    return labelled, holdout


def add_cold_start_attack_prior(
    train_rows: pd.DataFrame,
    apply_rows: pd.DataFrame,
    model,
    *,
    target_col: str = "kf_attack",
    output_col: str = "kf_attack_filled",
    feature_cols: list[str] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[pd.DataFrame, object]:
    """Fill missing KF attack with a conservative supervised prior.

    The prior model learns from rows where the Kalman filter already has enough
    history to produce kf_attack. It does not predict corners; it predicts the
    missing team-strength feature only. Existing kf_attack values are preserved.

    The default inputs are feature_1..10 plus home. Opponent KF context was
    tested but added no meaningful validation gain, so the final prior stays
    independent of whether the opponent also has KF history.
    """
    feature_cols = feature_cols or COLD_START_PRIOR_FEATURES
    known_train_rows = train_rows.dropna(subset=[target_col]).copy()

    fitted_model = model.fit(known_train_rows[feature_cols], known_train_rows[target_col])
    filled_rows = apply_rows.copy()
    missing_mask = filled_rows[target_col].isna()

    filled_rows[output_col] = filled_rows[target_col]
    if missing_mask.any():
        prior_values = fitted_model.predict(filled_rows.loc[missing_mask, feature_cols])
        lower = known_train_rows[target_col].quantile(lower_quantile)
        upper = known_train_rows[target_col].quantile(upper_quantile)
        filled_rows.loc[missing_mask, output_col] = np.clip(prior_values, lower, upper)

    return filled_rows, fitted_model


def composite_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Selection loss used on TUNE only.

    Corners are count data, so Poisson deviance is the most natural error term.
    MAE and RMSE are still included so the chosen model also has sensible point
    accuracy and does not win only by matching the count distribution shape.
    """
    y_pred_pos = np.clip(y_pred, 1e-6, None)
    return (
        0.25 * mean_absolute_error(y_true, y_pred)
        + 0.25 * np.sqrt(mean_squared_error(y_true, y_pred))
        + 0.50 * mean_poisson_deviance(y_true, y_pred_pos)
    )


def score(name: str, y_true, y_pred, width: int = 32) -> None:
    """Print the three reported validation metrics."""
    y_pred_pos = np.clip(y_pred, 1e-6, None)
    print(
        f"{name:<{width}} "
        f"MAE={mean_absolute_error(y_true, y_pred):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}  "
        f"PoissonDev={mean_poisson_deviance(y_true, y_pred_pos):.4f}"
    )
