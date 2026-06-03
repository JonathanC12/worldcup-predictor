import pandas as pd
from predict_matches import FIXTURES

def calculate_group_standings(predictions_csv: str) -> None:
    """
    Calculate and display group standings based on predicted results.
    Uses 3 points for win, 1 point for draw, 0 for loss.
    """
    df = pd.read_csv(predictions_csv)

    for group in sorted(df["group"].unique()):
        group_df = df[df["group"] == group].copy()

        # Collect all teams in this group
        teams = set(group_df["home_team"].tolist() + group_df["away_team"].tolist())
        standings = {team: {"P": 0, "W": 0, "D": 0, "L": 0, "Pts": 0} for team in teams}

        # Loop through each fixture and award points
        for _, row in group_df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            prediction = row["prediction"]

            standings[home]["P"] += 1
            standings[away]["P"] += 1

            if prediction == f"{home} win":
                standings[home]["W"] += 1
                standings[home]["Pts"] += 3
                standings[away]["L"] += 1

            elif prediction == "Draw":
                standings[home]["D"] += 1
                standings[home]["Pts"] += 1
                standings[away]["D"] += 1
                standings[away]["Pts"] += 1

            else:  # away win
                standings[away]["W"] += 1
                standings[away]["Pts"] += 3
                standings[home]["L"] += 1

        # Sort by points descending
        sorted_standings = sorted(
            standings.items(),
            key=lambda x: (x[1]["Pts"], x[1]["W"]),
            reverse=True
        )

        # Display
        print(f"\nGROUP {group} STANDINGS")
        print(f"{'Team':<25} {'P':<5} {'W':<5} {'D':<5} {'L':<5} {'Pts':<5}")
        print("-" * 50)
        for i, (team, stats) in enumerate(sorted_standings):
            # Mark top 2 who advance
            marker = "✓" if i < 2 else " "
            print(f"{marker} {team:<23} {stats['P']:<5} {stats['W']:<5} {stats['D']:<5} {stats['L']:<5} {stats['Pts']:<5}")

        print(f"  ✓ = Predicted to advance")


if __name__ == "__main__":
    calculate_group_standings("data/group_stage_predictions.csv")