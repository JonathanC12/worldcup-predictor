import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TRACKER_COLUMNS = [
    "date", "home_team", "away_team",
    "home_win_prob", "draw_prob", "away_win_prob",
    "predicted_outcome", "actual_outcome", "correct",
]


def record_actual_result(df: pd.DataFrame, home_team: str, away_team: str, home_score: int, away_score: int) -> pd.DataFrame:
    """
    Record (or correct) the actual result of a match by team names.
    This is how newly-completed matches get fed into the tracker: call this
    once per finished fixture with its final score before rerunning the script.
    """
    mask = (df["home_team"] == home_team) & (df["away_team"] == away_team)
    if not mask.any():
        raise ValueError(f"No fixture found for {home_team} vs {away_team}")

    if home_score > away_score:
        outcome = f"{home_team} win"
    elif home_score < away_score:
        outcome = f"{away_team} win"
    else:
        outcome = "Draw"

    df.loc[mask, "home_score"] = home_score
    df.loc[mask, "away_score"] = away_score
    df.loc[mask, "actual_outcome"] = outcome
    return df


def build_tracker(predictions_csv: str) -> pd.DataFrame:
    """Merge predicted probabilities with actual results into the tracker schema."""
    df = pd.read_csv(predictions_csv)

    # Only matches with a known actual outcome can be tracked so far.
    df = df.dropna(subset=["actual_outcome"]).copy()

    tracker = pd.DataFrame({
        "date": df["date"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_win_prob": df["home_win%"].str.rstrip("%").astype(float) / 100,
        "draw_prob": df["draw%"].str.rstrip("%").astype(float) / 100,
        "away_win_prob": df["away_win%"].str.rstrip("%").astype(float) / 100,
        "predicted_outcome": df["prediction"],
        "actual_outcome": df["actual_outcome"],
    })
    tracker["correct"] = tracker["predicted_outcome"] == tracker["actual_outcome"]
    return tracker[TRACKER_COLUMNS]


def print_accuracy(tracker: pd.DataFrame) -> None:
    total = len(tracker)
    correct = int(tracker["correct"].sum())
    accuracy = correct / total * 100 if total else 0.0
    print(f"\nTracked matches: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall prediction accuracy so far: {accuracy:.1f}%")


if __name__ == "__main__":
    logger.info("Loading predictions from data/group_stage_predictions.csv...")
    tracker_df = build_tracker("data/group_stage_predictions.csv")

    print(f"\n{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Predicted':<20} {'Actual':<20} {'Correct'}")
    print("-" * 115)
    for _, row in tracker_df.iterrows():
        print(f"{row['date']:<12} {row['home_team']:<25} {row['away_team']:<25} {row['predicted_outcome']:<20} {row['actual_outcome']:<20} {row['correct']}")

    print_accuracy(tracker_df)

    tracker_df.to_csv("data/prediction_tracker.csv", index=False)
    logger.info("Tracker saved to data/prediction_tracker.csv")
