import pandas as pd
import logging
from data_pipeline import load_raw_data, clean_data
from features import calculate_elo_ratings, add_form_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All 2026 World Cup Group Stage Fixtures, with actual final scores now that
# the group stage is complete. Team names and home/away designations match
# the official schedule in data/results.csv.
FIXTURES = [
    # Group A
    {"date": "2026-06-11", "home_team": "Mexico",        "away_team": "South Africa",   "group": "A", "neutral": False, "home_score": 2, "away_score": 0},
    {"date": "2026-06-11", "home_team": "South Korea",   "away_team": "Czech Republic", "group": "A", "neutral": True,  "home_score": 2, "away_score": 1},
    {"date": "2026-06-18", "home_team": "Czech Republic", "away_team": "South Africa",  "group": "A", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-18", "home_team": "Mexico",        "away_team": "South Korea",    "group": "A", "neutral": False, "home_score": 1, "away_score": 0},
    {"date": "2026-06-24", "home_team": "South Africa",  "away_team": "South Korea",    "group": "A", "neutral": True,  "home_score": 1, "away_score": 0},
    {"date": "2026-06-24", "home_team": "Mexico",        "away_team": "Czech Republic", "group": "A", "neutral": False, "home_score": 3, "away_score": 0},
    # Group B
    {"date": "2026-06-12", "home_team": "Canada",        "away_team": "Bosnia and Herzegovina", "group": "B", "neutral": False, "home_score": 1, "away_score": 1},
    {"date": "2026-06-13", "home_team": "Qatar",         "away_team": "Switzerland",    "group": "B", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-18", "home_team": "Switzerland",   "away_team": "Bosnia and Herzegovina", "group": "B", "neutral": True,  "home_score": 4, "away_score": 1},
    {"date": "2026-06-18", "home_team": "Canada",        "away_team": "Qatar",          "group": "B", "neutral": False, "home_score": 6, "away_score": 0},
    {"date": "2026-06-24", "home_team": "Bosnia and Herzegovina", "away_team": "Qatar",  "group": "B", "neutral": True,  "home_score": 3, "away_score": 1},
    {"date": "2026-06-24", "home_team": "Canada",        "away_team": "Switzerland",    "group": "B", "neutral": False, "home_score": 1, "away_score": 2},
    # Group C
    {"date": "2026-06-13", "home_team": "Brazil",        "away_team": "Morocco",        "group": "C", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-13", "home_team": "Haiti",         "away_team": "Scotland",       "group": "C", "neutral": True,  "home_score": 0, "away_score": 1},
    {"date": "2026-06-19", "home_team": "Scotland",      "away_team": "Morocco",        "group": "C", "neutral": True,  "home_score": 0, "away_score": 1},
    {"date": "2026-06-19", "home_team": "Brazil",        "away_team": "Haiti",          "group": "C", "neutral": True,  "home_score": 3, "away_score": 0},
    {"date": "2026-06-24", "home_team": "Morocco",       "away_team": "Haiti",          "group": "C", "neutral": True,  "home_score": 4, "away_score": 2},
    {"date": "2026-06-24", "home_team": "Scotland",      "away_team": "Brazil",         "group": "C", "neutral": True,  "home_score": 0, "away_score": 3},
    # Group D
    {"date": "2026-06-12", "home_team": "United States", "away_team": "Paraguay",       "group": "D", "neutral": False, "home_score": 4, "away_score": 1},
    {"date": "2026-06-13", "home_team": "Australia",     "away_team": "Turkey",         "group": "D", "neutral": True,  "home_score": 2, "away_score": 0},
    {"date": "2026-06-19", "home_team": "Turkey",        "away_team": "Paraguay",       "group": "D", "neutral": True,  "home_score": 0, "away_score": 1},
    {"date": "2026-06-19", "home_team": "United States", "away_team": "Australia",      "group": "D", "neutral": False, "home_score": 2, "away_score": 0},
    {"date": "2026-06-25", "home_team": "United States", "away_team": "Turkey",         "group": "D", "neutral": False, "home_score": 2, "away_score": 3},
    {"date": "2026-06-25", "home_team": "Paraguay",      "away_team": "Australia",      "group": "D", "neutral": True,  "home_score": 0, "away_score": 0},
    # Group E
    {"date": "2026-06-14", "home_team": "Germany",       "away_team": "Curaçao",        "group": "E", "neutral": True,  "home_score": 7, "away_score": 1},
    {"date": "2026-06-14", "home_team": "Ivory Coast",   "away_team": "Ecuador",        "group": "E", "neutral": True,  "home_score": 1, "away_score": 0},
    {"date": "2026-06-20", "home_team": "Germany",       "away_team": "Ivory Coast",    "group": "E", "neutral": True,  "home_score": 2, "away_score": 1},
    {"date": "2026-06-20", "home_team": "Ecuador",       "away_team": "Curaçao",        "group": "E", "neutral": True,  "home_score": 0, "away_score": 0},
    {"date": "2026-06-25", "home_team": "Curaçao",       "away_team": "Ivory Coast",    "group": "E", "neutral": True,  "home_score": 0, "away_score": 2},
    {"date": "2026-06-25", "home_team": "Ecuador",       "away_team": "Germany",        "group": "E", "neutral": True,  "home_score": 2, "away_score": 1},
    # Group F
    {"date": "2026-06-14", "home_team": "Sweden",        "away_team": "Tunisia",        "group": "F", "neutral": True,  "home_score": 5, "away_score": 1},
    {"date": "2026-06-14", "home_team": "Netherlands",   "away_team": "Japan",          "group": "F", "neutral": True,  "home_score": 2, "away_score": 2},
    {"date": "2026-06-20", "home_team": "Netherlands",   "away_team": "Sweden",         "group": "F", "neutral": True,  "home_score": 5, "away_score": 1},
    {"date": "2026-06-20", "home_team": "Tunisia",       "away_team": "Japan",          "group": "F", "neutral": True,  "home_score": 0, "away_score": 4},
    {"date": "2026-06-25", "home_team": "Japan",         "away_team": "Sweden",         "group": "F", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-25", "home_team": "Tunisia",       "away_team": "Netherlands",    "group": "F", "neutral": True,  "home_score": 1, "away_score": 3},
    # Group G
    {"date": "2026-06-15", "home_team": "Belgium",       "away_team": "Egypt",          "group": "G", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-15", "home_team": "Iran",          "away_team": "New Zealand",    "group": "G", "neutral": True,  "home_score": 2, "away_score": 2},
    {"date": "2026-06-21", "home_team": "Belgium",       "away_team": "Iran",           "group": "G", "neutral": True,  "home_score": 0, "away_score": 0},
    {"date": "2026-06-21", "home_team": "New Zealand",   "away_team": "Egypt",          "group": "G", "neutral": True,  "home_score": 1, "away_score": 3},
    {"date": "2026-06-26", "home_team": "New Zealand",   "away_team": "Belgium",        "group": "G", "neutral": True,  "home_score": 1, "away_score": 5},
    {"date": "2026-06-26", "home_team": "Egypt",         "away_team": "Iran",           "group": "G", "neutral": True,  "home_score": 1, "away_score": 1},
    # Group H
    {"date": "2026-06-15", "home_team": "Spain",         "away_team": "Cape Verde",     "group": "H", "neutral": True,  "home_score": 0, "away_score": 0},
    {"date": "2026-06-15", "home_team": "Saudi Arabia",  "away_team": "Uruguay",        "group": "H", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-21", "home_team": "Spain",         "away_team": "Saudi Arabia",   "group": "H", "neutral": True,  "home_score": 4, "away_score": 0},
    {"date": "2026-06-21", "home_team": "Uruguay",       "away_team": "Cape Verde",     "group": "H", "neutral": True,  "home_score": 2, "away_score": 2},
    {"date": "2026-06-26", "home_team": "Uruguay",       "away_team": "Spain",          "group": "H", "neutral": True,  "home_score": 0, "away_score": 1},
    {"date": "2026-06-26", "home_team": "Cape Verde",    "away_team": "Saudi Arabia",   "group": "H", "neutral": True,  "home_score": 0, "away_score": 0},
    # Group I
    {"date": "2026-06-16", "home_team": "France",        "away_team": "Senegal",        "group": "I", "neutral": True,  "home_score": 3, "away_score": 1},
    {"date": "2026-06-16", "home_team": "Iraq",          "away_team": "Norway",         "group": "I", "neutral": True,  "home_score": 1, "away_score": 4},
    {"date": "2026-06-22", "home_team": "France",        "away_team": "Iraq",           "group": "I", "neutral": True,  "home_score": 4, "away_score": 0},
    {"date": "2026-06-22", "home_team": "Norway",        "away_team": "Senegal",        "group": "I", "neutral": True,  "home_score": 3, "away_score": 2},
    {"date": "2026-06-26", "home_team": "Senegal",       "away_team": "Iraq",           "group": "I", "neutral": True,  "home_score": 5, "away_score": 0},
    {"date": "2026-06-26", "home_team": "Norway",        "away_team": "France",         "group": "I", "neutral": True,  "home_score": 1, "away_score": 4},
    # Group J
    {"date": "2026-06-16", "home_team": "Austria",       "away_team": "Jordan",         "group": "J", "neutral": True,  "home_score": 3, "away_score": 1},
    {"date": "2026-06-16", "home_team": "Argentina",     "away_team": "Algeria",        "group": "J", "neutral": True,  "home_score": 3, "away_score": 0},
    {"date": "2026-06-22", "home_team": "Argentina",     "away_team": "Austria",        "group": "J", "neutral": True,  "home_score": 2, "away_score": 0},
    {"date": "2026-06-22", "home_team": "Jordan",        "away_team": "Algeria",        "group": "J", "neutral": True,  "home_score": 1, "away_score": 2},
    {"date": "2026-06-27", "home_team": "Algeria",       "away_team": "Austria",        "group": "J", "neutral": True,  "home_score": 3, "away_score": 3},
    {"date": "2026-06-27", "home_team": "Jordan",        "away_team": "Argentina",      "group": "J", "neutral": True,  "home_score": 1, "away_score": 3},
    # Group K
    {"date": "2026-06-17", "home_team": "Portugal",      "away_team": "DR Congo",       "group": "K", "neutral": True,  "home_score": 1, "away_score": 1},
    {"date": "2026-06-17", "home_team": "Uzbekistan",    "away_team": "Colombia",       "group": "K", "neutral": True,  "home_score": 1, "away_score": 3},
    {"date": "2026-06-23", "home_team": "Portugal",      "away_team": "Uzbekistan",     "group": "K", "neutral": True,  "home_score": 5, "away_score": 0},
    {"date": "2026-06-23", "home_team": "Colombia",      "away_team": "DR Congo",       "group": "K", "neutral": True,  "home_score": 1, "away_score": 0},
    {"date": "2026-06-27", "home_team": "Colombia",      "away_team": "Portugal",       "group": "K", "neutral": True,  "home_score": 0, "away_score": 0},
    {"date": "2026-06-27", "home_team": "DR Congo",      "away_team": "Uzbekistan",     "group": "K", "neutral": True,  "home_score": 3, "away_score": 1},
    # Group L
    {"date": "2026-06-17", "home_team": "England",       "away_team": "Croatia",        "group": "L", "neutral": True,  "home_score": 4, "away_score": 2},
    {"date": "2026-06-17", "home_team": "Ghana",         "away_team": "Panama",         "group": "L", "neutral": True,  "home_score": 1, "away_score": 0},
    {"date": "2026-06-23", "home_team": "England",       "away_team": "Ghana",          "group": "L", "neutral": True,  "home_score": 0, "away_score": 0},
    {"date": "2026-06-23", "home_team": "Panama",        "away_team": "Croatia",        "group": "L", "neutral": True,  "home_score": 0, "away_score": 1},
    {"date": "2026-06-27", "home_team": "Panama",        "away_team": "England",        "group": "L", "neutral": True,  "home_score": 0, "away_score": 2},
    {"date": "2026-06-27", "home_team": "Croatia",       "away_team": "Ghana",          "group": "L", "neutral": True,  "home_score": 2, "away_score": 1},
]


def get_latest_elos(df: pd.DataFrame) -> dict:
    """Get the most recent Elo rating for every team."""
    latest_home = df.drop_duplicates(subset="home_team", keep="last")[["home_team", "home_elo"]]
    latest_away = df.drop_duplicates(subset="away_team", keep="last")[["away_team", "away_elo"]]
    latest_home.columns = ["team", "elo"]
    latest_away.columns = ["team", "elo"]
    combined = pd.concat([latest_home, latest_away])
    combined = combined.drop_duplicates(subset="team", keep="last")
    return dict(zip(combined["team"], combined["elo"]))

def get_latest_form(df: pd.DataFrame) -> dict:
    """Get the most recent form rating for every team."""
    latest = df.drop_duplicates(subset="home_team", keep="last")[["home_team", "home_form"]]
    latest.columns = ["team", "form"]
    return dict(zip(latest["team"], latest["form"]))

def actual_outcome(fixture: dict) -> str:
    """Derive the actual match outcome from the recorded final score."""
    if fixture["home_score"] > fixture["away_score"]:
        return f"{fixture['home_team']} win"
    elif fixture["home_score"] < fixture["away_score"]:
        return f"{fixture['away_team']} win"
    return "Draw"

def predict_fixture(model, fixture: dict, elo_ratings: dict, form_ratings: dict) -> dict:
    """Predict a single fixture using the pre-match Elo/form ratings."""
    import pandas as pd

    home = fixture["home_team"]
    away = fixture["away_team"]
    default_elo = 1500.0
    default_form = 0.5

    home_elo = elo_ratings.get(home, default_elo)
    away_elo = elo_ratings.get(away, default_elo)
    home_form = form_ratings.get(home, default_form)
    away_form = form_ratings.get(away, default_form)

    features = pd.DataFrame([{
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_form": home_form,
        "away_form": away_form,
        "form_diff": home_form - away_form,
        "neutral": int(fixture["neutral"])
    }])

    proba = model.predict_proba(features)[0]
    away_win = round(float(proba[0]), 3)
    draw = round(float(proba[1]), 3)
    home_win = round(float(proba[2]), 3)

    max_prob = max(away_win, draw, home_win)
    if max_prob == home_win:
        prediction = f"{home} win"
    elif max_prob == draw:
        prediction = "Draw"
    else:
        prediction = f"{away} win"

    actual = actual_outcome(fixture)

    return {
        "group": fixture["group"],
        "date": fixture["date"],
        "home_team": home,
        "away_team": away,
        "home_win%": f"{home_win*100:.1f}%",
        "draw%": f"{draw*100:.1f}%",
        "away_win%": f"{away_win*100:.1f}%",
        "prediction": prediction,
        "home_score": fixture["home_score"],
        "away_score": fixture["away_score"],
        "actual_outcome": actual,
        "correct": prediction == actual,
    }

if __name__ == "__main__":
    import mlflow
    import mlflow.sklearn

    # Load model
    logger.info("Loading model from MLflow...")
    model = mlflow.sklearn.load_model("models:/worldcup-predictor-xgboost/1")

    # Build feature data
    logger.info("Building Elo and form ratings...")
    df = load_raw_data()
    df = clean_data(df)
    df = calculate_elo_ratings(df)
    df = add_form_features(df)

    elo_ratings = get_latest_elos(df)
    form_ratings = get_latest_form(df)

    # Run predictions for all fixtures
    logger.info("Predicting all group stage fixtures...")
    results = []
    for fixture in FIXTURES:
        result = predict_fixture(model, fixture, elo_ratings, form_ratings)
        results.append(result)

    # Display results by group
    results_df = pd.DataFrame(results)
    correct_count = int(results_df["correct"].sum())
    # Pretty table display
    for group in sorted(results_df["group"].unique()):
        print(f"\nGROUP {group}")
        print(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Home%':<8} {'Draw%':<8} {'Away%':<8} {'Prediction':<20} {'Actual':<20} {'Score'}")
        print("-" * 140)
        group_df = results_df[results_df["group"] == group]
        for _, row in group_df.iterrows():
            score = f"{row['home_score']}-{row['away_score']}"
            print(f"{row['date']:<12} {row['home_team']:<25} {row['away_team']:<25} {row['home_win%']:<8} {row['draw%']:<8} {row['away_win%']:<8} {row['prediction']:<20} {row['actual_outcome']:<20} {score}")

    print(f"\nOverall group stage prediction accuracy: {correct_count}/{len(results_df)} ({correct_count/len(results_df)*100:.1f}%)")

    # Save to CSV
    results_df.to_csv("data/group_stage_predictions.csv", index=False)
    logger.info("Predictions saved to data/group_stage_predictions.csv")
