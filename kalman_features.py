"""Causal team-strength features from a scalar Kalman filter.

Each team has two hidden states:
  attack  = how many corners this team tends to win
  defense = how many corners this team tends to allow opponents to win

Raw corners are not a clean attack signal. If Arsenal win 8 corners at home,
some of that is Arsenal attack, some is opponent defensive weakness, and some
is home advantage. Before updating Arsenal's attack state, we subtract the
current opponent-defence estimate and the global home advantage. The remaining
number is a cleaner "attack observation".

Matches on the same date are batched. We first record all pre-match features,
then update states after the whole date. That prevents an early same-day match
from leaking into a later same-day pre-match row.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


KF_PROCESS_VAR = 0.03
KF_OBSERVATION_VAR = 30.0
KF_PRIOR_MEAN = 6.0
KF_PRIOR_VAR = 9.0
MIN_TEAM_MATCHES = 5


@dataclass
class KalmanState:
    """Mean and uncertainty for one hidden team-strength state."""

    mean: float
    variance: float


def _predict_state(state: KalmanState, process_var: float) -> KalmanState:
    """Let the team state drift a little before the next match."""
    return KalmanState(state.mean, state.variance + process_var)


def _update_state(state: KalmanState, observation: float, observation_var: float) -> KalmanState:
    """Blend old belief with one noisy match observation."""
    kalman_gain = state.variance / (state.variance + observation_var)
    updated_mean = state.mean + kalman_gain * (observation - state.mean)
    updated_var = (1.0 - kalman_gain) * state.variance
    return KalmanState(updated_mean, updated_var)


def build_kalman_features(
    rows: pd.DataFrame,
    process_var: float = KF_PROCESS_VAR,
    observation_var: float = KF_OBSERVATION_VAR,
    prior_mean: float = KF_PRIOR_MEAN,
    prior_var: float = KF_PRIOR_VAR,
    min_matches: int = MIN_TEAM_MATCHES,
) -> pd.DataFrame:
    """Add pre-match Kalman team-strength columns.

    Required columns are match_id, date, home, team, opponent and corners.
    Holdout rows may have missing corners. They receive features but never
    update states.

    For one home match:
      expected home corners = home_attack + (away_defense - league_mean) + home_adv
      expected away corners = away_attack + (home_defense - league_mean)

    So the update observations are:
      home attack observation = home corners - away_defense_adjustment - home_adv
      away attack observation = away corners - home_defense_adjustment

    Defense is updated with the same idea, just viewed from the opponent side.
    """
    labelled_rows = rows.dropna(subset=["corners"])
    home_advantage = (
        labelled_rows[labelled_rows["home"] == 1]["corners"].mean()
        - labelled_rows[labelled_rows["home"] == 0]["corners"].mean()
    )

    attack_state = defaultdict(lambda: KalmanState(prior_mean, prior_var))
    defense_state = defaultdict(lambda: KalmanState(prior_mean, prior_var))
    matches_seen = defaultdict(int)

    home_rows = rows[rows["home"] == 1].sort_values("date").set_index("match_id")
    away_rows = rows[rows["home"] == 0].set_index("match_id")

    feature_records = []

    for _, same_day_home_rows in home_rows.groupby("date", sort=True):
        teams_playing_today = set()
        for match_id, home_row in same_day_home_rows.iterrows():
            away_row = away_rows.loc[match_id]
            teams_playing_today.add(home_row["team"])
            teams_playing_today.add(away_row["team"])

        for team in teams_playing_today:
            attack_state[team] = _predict_state(attack_state[team], process_var)
            defense_state[team] = _predict_state(defense_state[team], process_var)

        pending_updates = []
        for match_id, home_row in same_day_home_rows.iterrows():
            away_row = away_rows.loc[match_id]
            home_team = home_row["team"]
            away_team = away_row["team"]

            def pre_match_features(team: str, opponent: str) -> tuple[float, float, float, float]:
                if matches_seen[team] < min_matches or matches_seen[opponent] < min_matches:
                    return (np.nan, np.nan, np.nan, np.nan)
                return (
                    attack_state[team].mean,
                    defense_state[team].mean,
                    attack_state[opponent].mean,
                    defense_state[opponent].mean,
                )

            feature_records += [
                (match_id, home_team, *pre_match_features(home_team, away_team)),
                (match_id, away_team, *pre_match_features(away_team, home_team)),
            ]

            home_corners = home_row.get("corners", np.nan)
            away_corners = away_row.get("corners", np.nan)
            if not (pd.isna(home_corners) or pd.isna(away_corners)):
                pending_updates.append(
                    (home_team, away_team, float(home_corners), float(away_corners))
                )

        for home_team, away_team, home_corners, away_corners in pending_updates:
            home_attack = attack_state[home_team].mean
            away_attack = attack_state[away_team].mean
            home_defense = defense_state[home_team].mean
            away_defense = defense_state[away_team].mean

            # Convert raw corners into cleaner state observations.
            # Example: home attack gets only the part of home corners not already
            # explained by away defensive weakness and league home advantage.
            home_attack_obs = home_corners - (away_defense - prior_mean) - home_advantage
            away_attack_obs = away_corners - (home_defense - prior_mean)
            away_defense_obs = home_corners - (home_attack - prior_mean) - home_advantage
            home_defense_obs = away_corners - (away_attack - prior_mean)

            attack_state[home_team] = _update_state(
                attack_state[home_team], home_attack_obs, observation_var
            )
            attack_state[away_team] = _update_state(
                attack_state[away_team], away_attack_obs, observation_var
            )
            defense_state[away_team] = _update_state(
                defense_state[away_team], away_defense_obs, observation_var
            )
            defense_state[home_team] = _update_state(
                defense_state[home_team], home_defense_obs, observation_var
            )

            matches_seen[home_team] += 1
            matches_seen[away_team] += 1

    features = pd.DataFrame(
        feature_records,
        columns=[
            "match_id",
            "team",
            "kf_attack",
            "kf_defense",
            "opp_kf_attack",
            "opp_kf_defense",
        ],
    )
    return rows.merge(features, on=["match_id", "team"], how="left")
