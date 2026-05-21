"""Shared helpers for the corner forecasting notebooks and final predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "inputs"

CORE_SEASONS = list(range(2012, 2018))
TUNE_SEASONS = [2018, 2019]
TRAIN_SEASONS = list(range(2012, 2020))
VAL_SEASONS = [2020, 2021, 2022]

FEATURE_COLS = [f"feature_{i}" for i in range(1, 11)]
COLD_START_PRIOR_FEATURES = FEATURE_COLS + ["home"]
FINAL_MODEL_FEATURES = FEATURE_COLS + ["home", "kf_attack_filled"]
POISSON_ALPHA = 0.1


def _resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    """Use either the project root or the actual inputs directory."""
    root = Path(data_dir).resolve() if data_dir is not None else INPUT_DIR
    if (root / "match_data.csv").exists():
        return root
    if (root / "inputs" / "match_data.csv").exists():
        return root / "inputs"
    raise FileNotFoundError(
        "Could not find match_data.csv. Pass the project root or inputs/ "
        f"to load_data(). Checked: {root} and {root / 'inputs'}."
    )


def load_data(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labelled training rows and unlabelled holdout feature rows.

    By default this reads from the repository's ``inputs/`` folder, not from the
    current notebook or terminal working directory.
    """
    root = _resolve_data_dir(data_dir)
    labelled_rows = pd.read_csv(root / "match_data.csv", parse_dates=["date"])
    holdout_rows = pd.read_csv(
        root / "match_data_holdout_features.csv",
        parse_dates=["date"],
    )
    return labelled_rows, holdout_rows


def composite_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Model loss used on TUNE only.

    Corners are counts, so Poisson deviance gets the largest weight. MAE and
    RMSE keep the selected model useful as a point forecast too. MAE measures
    the mean deviation. RMSE penalises large errors.
    """
    positive_pred = np.clip(y_pred, 1e-6, None)
    return (
        0.25 * mean_absolute_error(y_true, y_pred)
        + 0.25 * np.sqrt(mean_squared_error(y_true, y_pred))
        + 0.50 * mean_poisson_deviance(y_true, positive_pred)
    )


def score(name: str, y_true, y_pred, width: int = 32) -> None:
    """Print MAE, RMSE and Poisson deviance in one consistent format."""
    positive_pred = np.clip(y_pred, 1e-6, None)
    print(
        f"{name:<{width}} "
        f"MAE={mean_absolute_error(y_true, y_pred):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}  "
        f"PoissonDev={mean_poisson_deviance(y_true, positive_pred):.4f}"
    )


def make_cold_start_prior_model() -> HistGradientBoostingRegressor:
    """Create the model used only to fill missing ``kf_attack`` values.

    The tree is deliberately small and regularised. It is not the final corner
    model; it only estimates a plausible initial attack strength when Kalman has
    not seen enough history for a real team state.
    """
    return HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=5.0,
        learning_rate=0.03,
        random_state=42,
    )


def make_corner_model(alpha: float = POISSON_ALPHA) -> Pipeline:
    """Create the final Poisson model for expected corner counts."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )


def fill_missing_kf_attack_with_prior(
    prior_training_rows: pd.DataFrame,
    rows_to_fill: pd.DataFrame,
    prior_model,
    *,
    kf_attack_col: str = "kf_attack",
    filled_attack_col: str = "kf_attack_filled",
    prior_feature_cols: list[str] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[pd.DataFrame, object]:
    """Fill only missing Kalman attack values with a supervised prior.

    The prior learns:

    ``feature_1..10 + home -> kf_attack``

    Existing Kalman values are left untouched. Only rows where ``kf_attack`` is
    missing receive a prior prediction, clipped to the middle 90 percent of
    observed KF attack values to avoid extreme cold-start guesses.
    """
    prior_feature_cols = prior_feature_cols or COLD_START_PRIOR_FEATURES
    rows_with_known_attack = prior_training_rows.dropna(subset=[kf_attack_col]).copy()

    fitted_prior_model = prior_model.fit(
        rows_with_known_attack[prior_feature_cols],
        rows_with_known_attack[kf_attack_col],
    )

    rows_with_filled_attack = rows_to_fill.copy()
    needs_prior = rows_with_filled_attack[kf_attack_col].isna()
    rows_with_filled_attack[filled_attack_col] = rows_with_filled_attack[kf_attack_col]

    if needs_prior.any():
        prior_predictions = fitted_prior_model.predict(
            rows_with_filled_attack.loc[needs_prior, prior_feature_cols]
        )
        lower_bound = rows_with_known_attack[kf_attack_col].quantile(lower_quantile)
        upper_bound = rows_with_known_attack[kf_attack_col].quantile(upper_quantile)
        rows_with_filled_attack.loc[needs_prior, filled_attack_col] = np.clip(
            prior_predictions,
            lower_bound,
            upper_bound,
        )

    return rows_with_filled_attack, fitted_prior_model
