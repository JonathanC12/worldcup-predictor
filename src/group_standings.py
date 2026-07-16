import sys
from predict_matches import FIXTURES

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def calculate_standings(fixtures: list) -> dict:
    """
    Calculate final group standings from actual match results.
    Uses 3 points for a win, 1 for a draw, 0 for a loss, with FIFA's
    standard tiebreakers: points, then goal difference, then goals for.
    """
    standings = {}
    for group in sorted(set(f["group"] for f in fixtures)):
        group_fixtures = [f for f in fixtures if f["group"] == group]
        teams = set(f["home_team"] for f in group_fixtures) | set(f["away_team"] for f in group_fixtures)
        stats = {team: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0} for team in teams}

        for f in group_fixtures:
            home, away = f["home_team"], f["away_team"]
            hs, as_ = f["home_score"], f["away_score"]

            stats[home]["P"] += 1
            stats[away]["P"] += 1
            stats[home]["GF"] += hs
            stats[home]["GA"] += as_
            stats[away]["GF"] += as_
            stats[away]["GA"] += hs

            if hs > as_:
                stats[home]["W"] += 1
                stats[home]["Pts"] += 3
                stats[away]["L"] += 1
            elif hs < as_:
                stats[away]["W"] += 1
                stats[away]["Pts"] += 3
                stats[home]["L"] += 1
            else:
                stats[home]["D"] += 1
                stats[away]["D"] += 1
                stats[home]["Pts"] += 1
                stats[away]["Pts"] += 1

        for team in stats:
            stats[team]["GD"] = stats[team]["GF"] - stats[team]["GA"]

        sorted_teams = sorted(
            stats.items(),
            key=lambda x: (x[1]["Pts"], x[1]["GD"], x[1]["GF"]),
            reverse=True
        )
        standings[group] = sorted_teams

    return standings


def best_third_placed_teams(standings: dict, n: int = 8) -> list:
    """Rank every group's 3rd-place team by points/GD/GF and return the top n."""
    thirds = [(group, teams[2]) for group, teams in standings.items()]
    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x[1][1]["Pts"], x[1][1]["GD"], x[1][1]["GF"]),
        reverse=True
    )
    return thirds_sorted[:n]


def calculate_group_standings() -> None:
    """Display final group standings and advancement based on actual results."""
    standings = calculate_standings(FIXTURES)
    qualifying_thirds = {group for group, _ in best_third_placed_teams(standings)}

    for group in sorted(standings.keys()):
        print(f"\nGROUP {group} STANDINGS")
        print(f"{'Team':<25} {'P':<5} {'W':<5} {'D':<5} {'L':<5} {'GF':<5} {'GA':<5} {'GD':<5} {'Pts':<5}")
        print("-" * 70)
        for i, (team, s) in enumerate(standings[group]):
            if i < 2:
                marker = "✓"
            elif i == 2 and group in qualifying_thirds:
                marker = "✓"
            else:
                marker = " "
            print(f"{marker} {team:<23} {s['P']:<5} {s['W']:<5} {s['D']:<5} {s['L']:<5} {s['GF']:<5} {s['GA']:<5} {s['GD']:<5} {s['Pts']:<5}")

    print(f"\n  ✓ = Advanced to the Round of 32 (top 2 per group, plus the 8 best third-placed teams)")

    print("\nBEST THIRD-PLACED TEAMS")
    print(f"{'Group':<8} {'Team':<25} {'P':<5} {'W':<5} {'D':<5} {'L':<5} {'GD':<5} {'Pts':<5} {'Advanced'}")
    print("-" * 80)
    for group, (team, s) in sorted(best_third_placed_teams(standings, n=12), key=lambda x: (-x[1][1]["Pts"], -x[1][1]["GD"], -x[1][1]["GF"])):
        advanced = "✓" if group in qualifying_thirds else ""
        print(f"{group:<8} {team:<25} {s['P']:<5} {s['W']:<5} {s['D']:<5} {s['L']:<5} {s['GD']:<5} {s['Pts']:<5} {advanced}")


if __name__ == "__main__":
    calculate_group_standings()
