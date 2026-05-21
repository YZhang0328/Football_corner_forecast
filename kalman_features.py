"""Causal Kalman-filter team-strength features.

The filter keeps two hidden states for every team:

* attack: how many corners the team tends to win.
* defense: how many corners the team tends to allow its opponent to win.

Before a match, these hidden states are not directly observed. After the match,
the observed corners provide a noisy signal. For example, if a home team wins 8
corners, that number mixes its own attack strength, the opponent's defensive
weakness, and the general home advantage. The update removes the other known
pieces first, then uses the remainder as a noisy observation of attack.

Matches on the same date are batched. The code records every pre-match feature
for that date first, then updates states with that date's results. This avoids a
same-day leakage problem where an early kick-off could affect a later kick-off.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


KF_PROCESS_VAR = 0.03
KF_OBSERVATION_VAR = 30.0
KF_PRIOR_MEAN = 6.0
KF_PRIOR_VAR = 9.0
MIN_TEAM_MATCHES = 5
KALMAN_FEATURE_COLS = [
    "kf_attack",
    "kf_defense",
    "opp_kf_attack",
    "opp_kf_defense",
]


@dataclass
class KalmanState:
    """Mean and uncertainty for one hidden team-strength state."""

    mean: float
    variance: float


def _predict_state(state: KalmanState, process_var: float) -> KalmanState:
    """Let a team's hidden strength drift slightly before its next match."""
    return KalmanState(state.mean, state.variance + process_var)


def _update_state(
    state: KalmanState,
    observation: float,
    observation_var: float,
) -> KalmanState:
    """Blend the old hidden state with one noisy observed signal."""
    kalman_gain = state.variance / (state.variance + observation_var)
    updated_mean = state.mean + kalman_gain * (observation - state.mean)
    updated_variance = (1.0 - kalman_gain) * state.variance
    return KalmanState(updated_mean, updated_variance)


def _has_enough_history(
    team: str,
    opponent: str,
    matches_seen_by_team: dict[str, int],
    min_matches: int,
) -> bool:
    """Require both teams to have enough history before exposing KF features."""
    return (
        matches_seen_by_team[team] >= min_matches
        and matches_seen_by_team[opponent] >= min_matches
    )


def _pre_match_feature_values(
    team: str,
    opponent: str,
    attack_state_by_team: dict[str, KalmanState],
    defense_state_by_team: dict[str, KalmanState],
    matches_seen_by_team: dict[str, int],
    min_matches: int,
) -> tuple[float, float, float, float]:
    """Return this team's and the opponent's pre-match KF estimates."""
    if not _has_enough_history(team, opponent, matches_seen_by_team, min_matches):
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        attack_state_by_team[team].mean,
        defense_state_by_team[team].mean,
        attack_state_by_team[opponent].mean,
        defense_state_by_team[opponent].mean,
    )


def _teams_in_same_day_matches(
    same_day_home_rows: pd.DataFrame,
    away_rows: pd.DataFrame,
) -> set[str]:
    """Find all teams that must be drifted before a match date is processed."""
    teams = set()
    for match_id, home_row in same_day_home_rows.iterrows():
        away_row = away_rows.loc[match_id]
        teams.add(home_row["team"])
        teams.add(away_row["team"])
    return teams


def _state_observations_from_match(
    *,
    home_corners: float,
    away_corners: float,
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    prior_mean: float,
    home_advantage: float,
) -> tuple[float, float, float, float]:
    """Convert raw corners into noisy observations for the four hidden states.

    Raw home corners are read as:

        home attack + away defensive weakness + home advantage

    So the home attack observation subtracts the current away-defense estimate
    and the home edge. The defense observations use the same idea from the
    opponent's corner count.
    """
    home_attack_observation = (
        home_corners - (away_defense - prior_mean) - home_advantage
    )
    away_attack_observation = away_corners - (home_defense - prior_mean)
    away_defense_observation = (
        home_corners - (home_attack - prior_mean) - home_advantage
    )
    home_defense_observation = away_corners - (away_attack - prior_mean)
    return (
        home_attack_observation,
        away_attack_observation,
        away_defense_observation,
        home_defense_observation,
    )


def build_kalman_features(
    rows: pd.DataFrame,
    process_var: float = KF_PROCESS_VAR,
    observation_var: float = KF_OBSERVATION_VAR,
    prior_mean: float = KF_PRIOR_MEAN,
    prior_var: float = KF_PRIOR_VAR,
    min_matches: int = MIN_TEAM_MATCHES,
) -> pd.DataFrame:
    """Add pre-match Kalman team-strength columns.

    Rows with missing ``corners`` receive pre-match features but do not update
    any state. This is how validation and holdout rows stay honest.

    The returned columns are:

    * ``kf_attack``: this team's pre-match attack estimate.
    * ``kf_defense``: this team's pre-match defensive weakness estimate.
    * ``opp_kf_attack``: opponent's pre-match attack estimate.
    * ``opp_kf_defense``: opponent's pre-match defensive weakness estimate.

    The final submitted model uses only ``kf_attack`` after cold-start filling.
    The other three columns are still returned because the notebook uses them
    for ablation checks and to show why attack-only was selected.
    """
    labelled_rows = rows.dropna(subset=["corners"])
    home_advantage = (
        labelled_rows[labelled_rows["home"] == 1]["corners"].mean()
        - labelled_rows[labelled_rows["home"] == 0]["corners"].mean()
    )

    attack_state_by_team = defaultdict(lambda: KalmanState(prior_mean, prior_var))
    defense_state_by_team = defaultdict(lambda: KalmanState(prior_mean, prior_var))
    matches_seen_by_team = defaultdict(int)

    home_rows = rows[rows["home"] == 1].sort_values("date").set_index("match_id")
    away_rows = rows[rows["home"] == 0].set_index("match_id")

    feature_records = []

    for _, same_day_home_rows in home_rows.groupby("date", sort=True):
        teams_playing_today = _teams_in_same_day_matches(same_day_home_rows, away_rows)
        for team in teams_playing_today:
            attack_state_by_team[team] = _predict_state(
                attack_state_by_team[team], process_var
            )
            defense_state_by_team[team] = _predict_state(
                defense_state_by_team[team], process_var
            )

        pending_updates = []
        for match_id, home_row in same_day_home_rows.iterrows():
            away_row = away_rows.loc[match_id]
            home_team = home_row["team"]
            away_team = away_row["team"]

            feature_records += [
                (
                    match_id,
                    home_team,
                    *_pre_match_feature_values(
                        home_team,
                        away_team,
                        attack_state_by_team,
                        defense_state_by_team,
                        matches_seen_by_team,
                        min_matches,
                    ),
                ),
                (
                    match_id,
                    away_team,
                    *_pre_match_feature_values(
                        away_team,
                        home_team,
                        attack_state_by_team,
                        defense_state_by_team,
                        matches_seen_by_team,
                        min_matches,
                    ),
                ),
            ]

            home_corners = home_row.get("corners", np.nan)
            away_corners = away_row.get("corners", np.nan)
            if not (pd.isna(home_corners) or pd.isna(away_corners)):
                pending_updates.append(
                    (home_team, away_team, float(home_corners), float(away_corners))
                )

        for home_team, away_team, home_corners, away_corners in pending_updates:
            home_attack = attack_state_by_team[home_team].mean
            away_attack = attack_state_by_team[away_team].mean
            home_defense = defense_state_by_team[home_team].mean
            away_defense = defense_state_by_team[away_team].mean

            (
                home_attack_observation,
                away_attack_observation,
                away_defense_observation,
                home_defense_observation,
            ) = _state_observations_from_match(
                home_corners=home_corners,
                away_corners=away_corners,
                home_attack=home_attack,
                away_attack=away_attack,
                home_defense=home_defense,
                away_defense=away_defense,
                prior_mean=prior_mean,
                home_advantage=home_advantage,
            )

            attack_state_by_team[home_team] = _update_state(
                attack_state_by_team[home_team],
                home_attack_observation,
                observation_var,
            )
            attack_state_by_team[away_team] = _update_state(
                attack_state_by_team[away_team],
                away_attack_observation,
                observation_var,
            )
            defense_state_by_team[away_team] = _update_state(
                defense_state_by_team[away_team],
                away_defense_observation,
                observation_var,
            )
            defense_state_by_team[home_team] = _update_state(
                defense_state_by_team[home_team],
                home_defense_observation,
                observation_var,
            )

            matches_seen_by_team[home_team] += 1
            matches_seen_by_team[away_team] += 1

    features = pd.DataFrame(
        feature_records,
        columns=["match_id", "team", *KALMAN_FEATURE_COLS],
    )
    return rows.merge(features, on=["match_id", "team"], how="left")
