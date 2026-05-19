from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

FEATURE_COLS  = [f'feature_{i}' for i in range(1, 11)]
TRAIN_SEASONS = list(range(2012, 2020))
VAL_SEASONS   = [2020, 2021, 2022]


def load_data(data_dir: Path = None):
    d = data_dir or Path.cwd()
    train   = pd.read_csv(d / 'match_data.csv',                  parse_dates=['date'])
    holdout = pd.read_csv(d / 'match_data_holdout_features.csv', parse_dates=['date'])
    return train, holdout


def score(name: str, y_true, y_pred, width: int = 30) -> None:
    y_pos = np.clip(y_pred, 1e-6, None)
    print(f'{name:<{width}}  MAE={mean_absolute_error(y_true, y_pred):.4f}  '
          f'RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}  '
          f'PoissonDev={mean_poisson_deviance(y_true, y_pos):.4f}')
