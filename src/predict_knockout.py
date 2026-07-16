import argparse
import logging
import sys

import pandas as pd

from data_pipeline import load_raw_data, clean_data
from features import calculate_elo_ratings, add_form_features, expected_score, get_match_result, K_FACTOR_TOURNAMENT
from predict_matches import FIXTURES as GROUP_STAGE_RESULTS, get_latest_elos, get_latest_form
from app import predict_knockout as api_predict_knockout, MatchRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

FORM_WINDOW = 5

# --- Round of 32 (actual bracket) ---
ROUND_OF_32 = [
    {"date": "2026-06-28", "home_team": "South Africa",   "away_team": "Canada"},
    {"date": "2026-06-29", "home_team": "Brazil",         "away_team": "Japan"},
    {"date": "2026-06-29", "home_team": "Germany",        "away_team": "Paraguay"},
    {"date": "2026-06-29", "home_team": "Netherlands",    "away_team": "Morocco"},
    {"date": "2026-06-30", "home_team": "Ivory Coast",    "away_team": "Norway"},
    {"date": "2026-06-30", "home_team": "France",         "away_team": "Sweden"},
    {"date": "2026-06-30", "home_team": "Mexico",         "away_team": "Ecuador"},
    {"date": "2026-07-01", "home_team": "England",        "away_team": "DR Congo"},
    {"date": "2026-07-01", "home_team": "Belgium",        "away_team": "Senegal"},
    {"date": "2026-07-01", "home_team": "United States",  "away_team": "Bosnia and Herzegovina"},
    {"date": "2026-07-02", "home_team": "Spain",          "away_team": "Austria"},
    {"date": "2026-07-02", "home_team": "Portugal",       "away_team": "Croatia"},
    {"date": "2026-07-02", "home_team": "Switzerland",    "away_team": "Algeria"},
    {"date": "2026-07-03", "home_team": "Australia",      "away_team": "Egypt"},
    {"date": "2026-07-03", "home_team": "Argentina",      "away_team": "Cape Verde"},
    {"date": "2026-07-03", "home_team": "Colombia",       "away_team": "Ghana"},
]

# Actual Round of 32 results. Extra-time/penalty matches are recorded with
# their final (AET) score, so a shootout counts as a draw for Elo purposes.
ROUND_OF_32_RESULTS = [
    {"home_team": "South Africa",  "away_team": "Canada",       "home_score": 0, "away_score": 1},
    {"home_team": "Brazil",        "away_team": "Japan",        "home_score": 2, "away_score": 1},
    {"home_team": "Germany",       "away_team": "Paraguay",     "home_score": 1, "away_score": 1},
    {"home_team": "Netherlands",   "away_team": "Morocco",      "home_score": 1, "away_score": 1},
    {"home_team": "Ivory Coast",   "away_team": "Norway",       "home_score": 1, "away_score": 2},
    {"home_team": "France",        "away_team": "Sweden",       "home_score": 3, "away_score": 0},
    {"home_team": "Mexico",        "away_team": "Ecuador",      "home_score": 2, "away_score": 0},
    {"home_team": "England",       "away_team": "DR Congo",     "home_score": 2, "away_score": 1},
    {"home_team": "Belgium",       "away_team": "Senegal",      "home_score": 3, "away_score": 2},
    {"home_team": "United States", "away_team": "Bosnia and Herzegovina", "home_score": 2, "away_score": 0},
    {"home_team": "Spain",         "away_team": "Austria",      "home_score": 3, "away_score": 0},
    {"home_team": "Portugal",      "away_team": "Croatia",      "home_score": 2, "away_score": 1},
    {"home_team": "Switzerland",   "away_team": "Algeria",      "home_score": 2, "away_score": 0},
    {"home_team": "Australia",     "away_team": "Egypt",        "home_score": 1, "away_score": 1},
    {"home_team": "Argentina",     "away_team": "Cape Verde",   "home_score": 3, "away_score": 2},
    {"home_team": "Colombia",      "away_team": "Ghana",        "home_score": 1, "away_score": 0},
]

# --- Round of 16 (actual bracket) ---
ROUND_OF_16 = [
    {"date": "2026-07-04", "home_team": "Canada",   "away_team": "Morocco"},
    {"date": "2026-07-04", "home_team": "Paraguay", "away_team": "France"},
    {"date": "2026-07-05", "home_team": "Brazil",   "away_team": "Norway"},
    {"date": "2026-07-05", "home_team": "Mexico",   "away_team": "England"},
    {"date": "2026-07-06", "home_team": "Portugal", "away_team": "Spain"},
    {"date": "2026-07-06", "home_team": "United States", "away_team": "Belgium"},
    {"date": "2026-07-07", "home_team": "Argentina", "away_team": "Egypt"},
    {"date": "2026-07-07", "home_team": "Switzerland", "away_team": "Colombia"},
]

ROUND_OF_16_RESULTS = [
    {"home_team": "Canada",   "away_team": "Morocco",      "home_score": 0, "away_score": 3},
    {"home_team": "Paraguay", "away_team": "France",       "home_score": 0, "away_score": 1},
    {"home_team": "Brazil",   "away_team": "Norway",       "home_score": 1, "away_score": 2},
    {"home_team": "Mexico",   "away_team": "England",      "home_score": 2, "away_score": 3},
    {"home_team": "Portugal", "away_team": "Spain",        "home_score": 0, "away_score": 1},
    {"home_team": "United States", "away_team": "Belgium", "home_score": 1, "away_score": 4},
    {"home_team": "Argentina", "away_team": "Egypt",       "home_score": 3, "away_score": 2},
    {"home_team": "Switzerland", "away_team": "Colombia",  "home_score": 0, "away_score": 0},
]

# --- Quarterfinals (actual bracket) ---
QUARTERFINALS = [
    {"date": "2026-07-09", "home_team": "France",   "away_team": "Morocco"},
    {"date": "2026-07-10", "home_team": "Spain",    "away_team": "Belgium"},
    {"date": "2026-07-11", "home_team": "Norway",   "away_team": "England"},
    {"date": "2026-07-11", "home_team": "Argentina", "away_team": "Switzerland"},
]

QUARTERFINAL_RESULTS = [
    {"home_team": "France",   "away_team": "Morocco",   "home_score": 2, "away_score": 0},
    {"home_team": "Spain",    "away_team": "Belgium",   "home_score": 2, "away_score": 1},
    {"home_team": "Norway",   "away_team": "England",   "home_score": 1, "away_score": 2},
    {"home_team": "Argentina", "away_team": "Switzerland", "home_score": 3, "away_score": 1},
]

# --- Semifinals (actual bracket) ---
SEMIFINALS = [
    {"date": "2026-07-14", "home_team": "Spain",   "away_team": "France"},
    {"date": "2026-07-15", "home_team": "England", "away_team": "Argentina"},
]

SEMIFINAL_RESULTS = [
    {"home_team": "Spain",   "away_team": "France",   "home_score": 2, "away_score": 0},
    {"home_team": "England", "away_team": "Argentina", "home_score": 1, "away_score": 2},
]

# --- Final ---
FINAL = [
    {"date": "2026-07-19", "home_team": "Spain", "away_team": "Argentina"},
]

# Each round is predicted using Elo/form built up from every match completed
# *before* it. Rerun with --round <name> once the previous round's fixtures
# are updated with real results, or just add the next round here.
ROUNDS = {
    "round_of_32": {
        "fixtures": ROUND_OF_32,
        "prior_results": GROUP_STAGE_RESULTS,
    },
    "round_of_16": {
        "fixtures": ROUND_OF_16,
        "prior_results": GROUP_STAGE_RESULTS + ROUND_OF_32_RESULTS,
    },
    "quarterfinals": {
        "fixtures": QUARTERFINALS,
        "prior_results": GROUP_STAGE_RESULTS + ROUND_OF_32_RESULTS + ROUND_OF_16_RESULTS,
    },
    "semifinals": {
        "fixtures": SEMIFINALS,
        "prior_results": GROUP_STAGE_RESULTS + ROUND_OF_32_RESULTS + ROUND_OF_16_RESULTS + QUARTERFINAL_RESULTS,
    },
    "final": {
        "fixtures": FINAL,
        "prior_results": GROUP_STAGE_RESULTS + ROUND_OF_32_RESULTS + ROUND_OF_16_RESULTS + QUARTERFINAL_RESULTS + SEMIFINAL_RESULTS,
    },
}


def update_ratings_with_results(elo_ratings: dict, form_ratings: dict, results: list) -> tuple:
    """
    Roll Elo (tournament K-factor) and rolling form forward through a list of
    completed matches, in the same way features.calculate_elo_ratings does
    for the historical dataset. Shootout matches are passed in with their
    AET score, so they count as a draw for both Elo and form.
    """
    elo_ratings = dict(elo_ratings)
    form_ratings = dict(form_ratings)
    team_history = {team: [] for team in elo_ratings}

    for m in results:
        home, away = m["home_team"], m["away_team"]
        home_elo = elo_ratings.get(home, 1500.0)
        away_elo = elo_ratings.get(away, 1500.0)
        home_exp = expected_score(home_elo, away_elo)
        away_exp = expected_score(away_elo, home_elo)
        home_actual, away_actual = get_match_result(m["home_score"], m["away_score"])

        elo_ratings[home] = home_elo + K_FACTOR_TOURNAMENT * (home_actual - home_exp)
        elo_ratings[away] = away_elo + K_FACTOR_TOURNAMENT * (away_actual - away_exp)

        team_history.setdefault(home, []).append(home_actual)
        team_history.setdefault(away, []).append(away_actual)
        form_ratings[home] = sum(team_history[home][-FORM_WINDOW:]) / len(team_history[home][-FORM_WINDOW:])
        form_ratings[away] = sum(team_history[away][-FORM_WINDOW:]) / len(team_history[away][-FORM_WINDOW:])

    return elo_ratings, form_ratings


def predict_knockout_fixture(fixture: dict, elo_ratings: dict, form_ratings: dict) -> dict:
    """Predict a single knockout fixture via the /predict_knockout endpoint logic."""
    home, away = fixture["home_team"], fixture["away_team"]
    request = MatchRequest(
        home_team=home,
        away_team=away,
        home_elo=elo_ratings.get(home, 1500.0),
        away_elo=elo_ratings.get(away, 1500.0),
        home_form=form_ratings.get(home, 0.5),
        away_form=form_ratings.get(away, 0.5),
        neutral=True,
    )
    response = api_predict_knockout(request)

    return {
        "date": fixture["date"],
        "home_team": home,
        "away_team": away,
        "home_advance%": f"{response.home_advance_probability*100:.1f}%",
        "away_advance%": f"{response.away_advance_probability*100:.1f}%",
        "extra_time%": f"{response.extra_time_probability*100:.1f}%",
        "predicted_outcome": response.predicted_outcome,
        "confidence": response.confidence,
    }


def run_round(round_name: str) -> pd.DataFrame:
    logger.info("Building baseline Elo and form ratings from historical data...")
    df = load_raw_data()
    df = clean_data(df)
    df = calculate_elo_ratings(df)
    df = add_form_features(df)

    elo_ratings = get_latest_elos(df)
    form_ratings = get_latest_form(df)

    round_config = ROUNDS[round_name]
    logger.info(f"Rolling Elo/form forward through {len(round_config['prior_results'])} completed matches...")
    elo_ratings, form_ratings = update_ratings_with_results(elo_ratings, form_ratings, round_config["prior_results"])

    logger.info(f"Predicting {len(round_config['fixtures'])} {round_name} fixtures...")
    results = [predict_knockout_fixture(f, elo_ratings, form_ratings) for f in round_config["fixtures"]]
    results_df = pd.DataFrame(results)
    results_df.insert(0, "round", round_name)
    return results_df


def print_table(round_name: str, results_df: pd.DataFrame) -> None:
    print(f"\n{round_name.upper().replace('_', ' ')}")
    print(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Home Adv%':<10} {'Away Adv%':<10} {'ExtraTime%':<11} {'Predicted':<25} {'Confidence'}")
    print("-" * 135)
    for _, row in results_df.iterrows():
        print(f"{row['date']:<12} {row['home_team']:<25} {row['away_team']:<25} {row['home_advance%']:<10} {row['away_advance%']:<10} {row['extra_time%']:<11} {row['predicted_outcome']:<25} {row['confidence']}")


def save_results(results_df: pd.DataFrame, output_path: str = "data/knockout_predictions.csv") -> None:
    try:
        existing = pd.read_csv(output_path)
        existing = existing[existing["round"] != results_df["round"].iloc[0]]
        combined = pd.concat([existing, results_df], ignore_index=True)
    except FileNotFoundError:
        combined = results_df
    combined.to_csv(output_path, index=False)
    logger.info(f"Knockout predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a round of the 2026 World Cup knockout stage.")
    parser.add_argument("--round", choices=list(ROUNDS.keys()), default="round_of_32",
                         help="Which round to predict. Update the fixtures list for each new round as it's confirmed.")
    args = parser.parse_args()

    results_df = run_round(args.round)
    print_table(args.round, results_df)
    save_results(results_df)
