from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

# ── Season splits ─────────────────────────────────────────────────────────────
# CORE  → fit models during hyperparameter search
# TUNE  → select hyperparameters; val never touched during tuning
# TRAIN → full training set for final model fit
# VAL   → one-shot evaluation; never used for any selection decision
CORE_SEASONS  = list(range(2012, 2018))
TUNE_SEASONS  = [2018, 2019]
TRAIN_SEASONS = list(range(2012, 2020))
VAL_SEASONS   = [2020, 2021, 2022]

FEATURE_COLS = [f'feature_{i}' for i in range(1, 11)]


def load_data(data_dir: Path = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load match_data.csv (train) and match_data_holdout_features.csv (holdout)."""
    d = data_dir or Path.cwd()
    train   = pd.read_csv(d / 'match_data.csv',                  parse_dates=['date'])
    holdout = pd.read_csv(d / 'match_data_holdout_features.csv', parse_dates=['date'])
    return train, holdout


def composite_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tuning objective: 0.25·MAE + 0.25·RMSE + 0.50·PoissonDev.

    Poisson Deviance is weighted double because it penalises distributional
    misfit (not just point error) and is the natural loss for count data.
    Used only during hyperparameter search on TUNE; never on VAL.
    """
    y_pos = np.clip(y_pred, 1e-6, None)
    return (
        0.25 * mean_absolute_error(y_true, y_pred)
        + 0.25 * np.sqrt(mean_squared_error(y_true, y_pred))
        + 0.50 * mean_poisson_deviance(y_true, y_pos)
    )


def score(name: str, y_true, y_pred, width: int = 32) -> None:
    """Print MAE, RMSE and Poisson Deviance for a named model."""
    y_pos = np.clip(y_pred, 1e-6, None)
    print(
        f'{name:<{width}} '
        f'MAE={mean_absolute_error(y_true, y_pred):.4f}  '
        f'RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}  '
        f'PoissonDev={mean_poisson_deviance(y_true, y_pos):.4f}'
    )
