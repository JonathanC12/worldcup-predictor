import pandas as pd
import numpy as np
import logging
from pathlib import Path
from data_pipeline import load_raw_data, clean_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Elo configuration
BASE_RATING = 1500
K_FACTOR_DEFAULT = 20
K_FACTOR_TOURNAMENT = 40

def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score (win probability) for team A against team B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def get_match_result(home_score: int, away_score: int) -> tuple:
    """Return actual scores for Elo update. Win=1, Draw=0.5, Loss=0."""
    if home_score > away_score:
        return 1.0, 0.0
    elif home_score < away_score:
        return 0.0, 1.0
    else:
        return 0.5, 0.5

def get_k_factor(tournament: str) -> float:
    """Use higher K factor for competitive matches."""
    friendly_keywords = ["friendly", "Friendly"]
    if any(keyword in tournament for keyword in friendly_keywords):
        return K_FACTOR_DEFAULT
    return K_FACTOR_TOURNAMENT

def calculate_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop through every match chronologically and maintain
    a running Elo rating for every team.
    Returns the original dataframe with Elo columns added.
    """
    ratings = {}  # team -> current elo rating
    records = []

    logger.info("Calculating Elo ratings across all matches...")

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Assign base rating if team hasn't been seen yet
        if home not in ratings:
            ratings[home] = BASE_RATING
        if away not in ratings:
            ratings[away] = BASE_RATING

        home_elo = ratings[home]
        away_elo = ratings[away]

        # Calculate expected scores
        home_expected = expected_score(home_elo, away_elo)
        away_expected = expected_score(away_elo, home_elo)

        # Get actual result
        home_actual, away_actual = get_match_result(
            row["home_score"], row["away_score"]
        )

        # Get K factor based on tournament type
        k = get_k_factor(row["tournament"])

        # Update ratings
        ratings[home] = home_elo + k * (home_actual - home_expected)
        ratings[away] = away_elo + k * (away_actual - away_expected)

        # Store the pre-match ratings alongside match info
        records.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "tournament": row["tournament"],
            "neutral": row["neutral"],
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "result": home_actual  # 1=home win, 0.5=draw, 0=away win
        })

    logger.info(f"Elo ratings calculated for {len(ratings)} teams")
    return pd.DataFrame(records)


def add_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling form features — average result over last N matches
    for each team going into each match.
    """
    logger.info(f"Calculating rolling form over last {window} matches...")
    df = df.copy()

    # Build a per-team match history as we go
    team_results = {}
    home_form = []
    away_form = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Get recent form (average result, 0-1 scale)
        home_recent = team_results.get(home, [])
        away_recent = team_results.get(away, [])

        home_form.append(
            np.mean(home_recent[-window:]) if home_recent else 0.5
        )
        away_form.append(
            np.mean(away_recent[-window:]) if away_recent else 0.5
        )

        # Update history after recording pre-match form
        if home not in team_results:
            team_results[home] = []
        if away not in team_results:
            team_results[away] = []

        team_results[home].append(row["result"])
        team_results[away].append(1 - row["result"])

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["form_diff"] = df["home_form"] - df["away_form"]

    return df


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df = calculate_elo_ratings(df)
    df = add_form_features(df)
    print(df.head())
    print(df.shape)
    print(df.columns.tolist())
