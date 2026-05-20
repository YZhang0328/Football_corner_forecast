"""Per-team attack/defence strength tracking via scalar Kalman filter."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KFState:
    """Posterior mean and variance of a 1-D random-walk Kalman filter."""
    mean: float
    var: float


def _kf_predict(state: KFState, Q: float) -> KFState:
    """Time update: let uncertainty grow by process noise Q before a match."""
    return KFState(state.mean, state.var + Q)


def _kf_update(state: KFState, y: float, R: float) -> KFState:
    """Measurement update: fuse observed corners y with observation noise R."""
    K = state.var / (state.var + R)
    return KFState(state.mean + K * (y - state.mean), (1 - K) * state.var)


def build_kalman_features(
    df: pd.DataFrame,
    Q: float = 0.03,
    R: float = 30.0,
    init_mean: float = 6.0,
    init_var: float = 9.0,
    min_matches: int = 5,
) -> pd.DataFrame:
    """Add KF-based team strength columns to df with no lookahead.

    For every match, in chronological order:
      1. Predict step  — advance both teams' states by Q (pre-match).
      2. Record features — snapshot taken here, so no future data leaks in.
      3. Update step   — fuse observed corners with R (post-match).

    Q controls how fast team strength can drift between matches.
    R controls how much a single match shifts the belief.
    Tuned values Q=0.03, R=30.0 favour a slow-moving, conservative prior,
    selected via CORE(2012-2017) → TUNE(2018-2019) grid search and confirmed
    by 4-fold walk-forward CV (ranks 1/56 in both cases).

    Args:
        df: Must contain columns: match_id, date, home, team, corners (NaN
            allowed for holdout rows). Pass train+holdout concatenated so
            that holdout rows inherit the terminal training states.
        Q: Process noise variance. Smaller = slower drift, more inertia.
        R: Observation noise variance. Larger = more conservative updates.
        init_mean: Prior mean (≈ league-wide average corners).
        init_var: Prior variance (wide = fast early learning).
        min_matches: Teams with fewer matches get NaN features (cold-start);
            these rows are later imputed with the training mean.

    Returns:
        df with four new float columns:
          kf_attack      — team's filtered attacking strength
          kf_defense     — team's filtered defensive concession rate
          opp_kf_attack  — opponent's filtered attacking strength
          opp_kf_defense — opponent's filtered defensive concession rate
    """
    attack   = defaultdict(lambda: KFState(init_mean, init_var))
    defense  = defaultdict(lambda: KFState(init_mean, init_var))
    n_played = defaultdict(int)

    home_df = df[df['home'] == 1].sort_values('date').set_index('match_id')
    away_df = df[df['home'] == 0].set_index('match_id')

    records = []
    for mid, h in home_df.iterrows():
        a      = away_df.loc[mid]
        ht, at = h['team'], a['team']

        # Step 1: predict (advance uncertainty before match)
        for team in (ht, at):
            attack[team]  = _kf_predict(attack[team],  Q)
            defense[team] = _kf_predict(defense[team], Q)

        # Step 2: record pre-match features (no lookahead)
        def _feat(team, opp):
            if n_played[team] < min_matches or n_played[opp] < min_matches:
                return (np.nan, np.nan, np.nan, np.nan)
            return (
                attack[team].mean, defense[team].mean,
                attack[opp].mean,  defense[opp].mean,
            )

        records += [(mid, ht, *_feat(ht, at)), (mid, at, *_feat(at, ht))]

        # Step 3: update with observed result (post-match)
        hc = h.get('corners', np.nan)
        ac = a.get('corners', np.nan)
        if not (pd.isna(hc) or pd.isna(ac)):
            attack[ht]  = _kf_update(attack[ht],  hc, R)
            attack[at]  = _kf_update(attack[at],  ac, R)
            defense[ht] = _kf_update(defense[ht], ac, R)
            defense[at] = _kf_update(defense[at], hc, R)
            n_played[ht] += 1
            n_played[at] += 1

    feats = pd.DataFrame(
        records,
        columns=['match_id', 'team', 'kf_attack', 'kf_defense',
                 'opp_kf_attack', 'opp_kf_defense'],
    )
    return df.merge(feats, on=['match_id', 'team'], how='left')
