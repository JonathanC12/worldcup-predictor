import pandas as pd
from predict_matches import FIXTURES

KNOCKOUT_ROUND_ORDER = ["round_of_32", "round_of_16", "quarterfinals", "semifinals"]
KNOCKOUT_ROUND_LABELS = {
    "round_of_32": "ROUND OF 32",
    "round_of_16": "ROUND OF 16",
    "quarterfinals": "QUARTERFINALS",
    "semifinals": "SEMIFINALS",
    "final": "FINAL",
}

def calculate_standings(df):
    standings = {}
    for group in df["group"].unique():
        group_df = df[df["group"] == group]
        teams = set(group_df["home_team"].tolist() + group_df["away_team"].tolist())
        team_stats = {team: {"P": 0, "W": 0, "D": 0, "L": 0, "Pts": 0} for team in teams}

        for _, row in group_df.iterrows():
            home, away = row["home_team"], row["away_team"]
            pred = row["prediction"]
            team_stats[home]["P"] += 1
            team_stats[away]["P"] += 1
            if pred == f"{home} win":
                team_stats[home]["W"] += 1; team_stats[home]["Pts"] += 3
                team_stats[away]["L"] += 1
            elif pred == "Draw":
                team_stats[home]["D"] += 1; team_stats[home]["Pts"] += 1
                team_stats[away]["D"] += 1; team_stats[away]["Pts"] += 1
            else:
                team_stats[away]["W"] += 1; team_stats[away]["Pts"] += 3
                team_stats[home]["L"] += 1

        standings[group] = sorted(team_stats.items(), key=lambda x: (x[1]["Pts"], x[1]["W"]), reverse=True)
    return standings

def build_matchup_card(row) -> str:
    """Render one knockout fixture as a matchup card with advance-probability bars."""
    home, away = row["home_team"], row["away_team"]
    home_pct = float(str(row["home_advance%"]).replace("%", ""))
    away_pct = float(str(row["away_advance%"]).replace("%", ""))
    home_wins = row["predicted_outcome"] == f"{home} advance"

    return f"""
    <div class="matchup-card">
        <div class="matchup-date">{row['date']}</div>
        <div class="matchup-row">
            <span class="matchup-team {'winner' if home_wins else ''}">{home}{' <span class="win-badge">WIN</span>' if home_wins else ''}</span>
            <div class="adv-bar-track"><div class="adv-bar home-adv" style="width:{home_pct}%"></div></div>
            <span class="matchup-pct">{row['home_advance%']}</span>
        </div>
        <div class="matchup-row">
            <span class="matchup-team {'winner' if not home_wins else ''}">{away}{' <span class="win-badge">WIN</span>' if not home_wins else ''}</span>
            <div class="adv-bar-track"><div class="adv-bar away-adv" style="width:{away_pct}%"></div></div>
            <span class="matchup-pct">{row['away_advance%']}</span>
        </div>
        <div class="matchup-confidence conf-{row['confidence'].lower()}">{row['confidence']} confidence</div>
    </div>"""


def build_knockout_html(knockout_df: pd.DataFrame) -> str:
    """Build the knockout bracket section (Round of 32 through Semifinals)."""
    sections_html = ""
    for round_key in KNOCKOUT_ROUND_ORDER:
        round_df = knockout_df[knockout_df["round"] == round_key]
        if round_df.empty:
            continue
        cards = "".join(build_matchup_card(row) for _, row in round_df.iterrows())
        sections_html += f"""
        <div class="bracket-round">
            <div class="round-header">{KNOCKOUT_ROUND_LABELS[round_key]}</div>
            <div class="matchup-grid">{cards}</div>
        </div>"""

    return f"""
    <div class="section-title">KNOCKOUT STAGE PREDICTIONS</div>
    {sections_html}"""


def build_final_html(knockout_df: pd.DataFrame) -> str:
    """Build the spotlight section for the final."""
    final_row = knockout_df[knockout_df["round"] == "final"]
    if final_row.empty:
        return ""
    row = final_row.iloc[0]
    home, away = row["home_team"], row["away_team"]
    home_pct = float(str(row["home_advance%"]).replace("%", ""))
    away_pct = float(str(row["away_advance%"]).replace("%", ""))
    home_wins = row["predicted_outcome"] == f"{home} advance"

    return f"""
    <div class="final-spotlight">
        <div class="final-eyebrow">2026 FIFA WORLD CUP FINAL &middot; {row['date']}</div>
        <div class="final-teams">
            <div class="final-team {'winner' if home_wins else ''}">
                <div class="final-team-name">{home}</div>
                <div class="final-team-pct">{row['home_advance%']}</div>
            </div>
            <div class="final-vs">VS</div>
            <div class="final-team {'winner' if not home_wins else ''}">
                <div class="final-team-name">{away}</div>
                <div class="final-team-pct">{row['away_advance%']}</div>
            </div>
        </div>
        <div class="final-bar-track">
            <div class="final-bar-home" style="width:{home_pct}%"></div>
            <div class="final-bar-away" style="width:{away_pct}%"></div>
        </div>
        <div class="final-prediction">🏆 Predicted champion: <strong>{row['predicted_outcome'].replace(' advance', '')}</strong> &middot; {row['confidence']} confidence &middot; {row['extra_time%']} extra-time probability</div>
    </div>"""


def generate_dashboard(predictions_csv: str, output_path: str, knockout_csv: str = "data/knockout_predictions.csv"):
    df = pd.read_csv(predictions_csv)
    standings = calculate_standings(df)
    accuracy = df["correct"].mean() * 100 if "correct" in df.columns else None

    try:
        knockout_df = pd.read_csv(knockout_csv)
    except FileNotFoundError:
        knockout_df = pd.DataFrame(columns=["round"])

    knockout_html = build_knockout_html(knockout_df)
    final_html = build_final_html(knockout_df)

    fixtures_by_group = {}
    for _, row in df.iterrows():
        g = row["group"]
        if g not in fixtures_by_group:
            fixtures_by_group[g] = []
        fixtures_by_group[g].append(row)

    groups_html = ""
    for group in sorted(standings.keys()):
        # Standings rows
        standing_rows = ""
        for i, (team, s) in enumerate(standings[group]):
            advance = "advancing" if i < 2 else ""
            badge = '<span class="badge">ADVANCE</span>' if i < 2 else ""
            standing_rows += f"""
            <tr class="{advance}">
                <td class="pos">{i+1}</td>
                <td class="team-name">{team}{badge}</td>
                <td>{s['P']}</td><td>{s['W']}</td><td>{s['D']}</td><td>{s['L']}</td>
                <td class="pts">{s['Pts']}</td>
            </tr>"""

        # Fixture rows
        fixture_rows = ""
        for row in fixtures_by_group.get(group, []):
            home_pct = float(row["home_win%"].replace("%",""))
            draw_pct = float(row["draw%"].replace("%",""))
            away_pct = float(row["away_win%"].replace("%",""))
            pred = row["prediction"]

            home_bold = "bold" if pred == f"{row['home_team']} win" else ""
            draw_bold = "bold" if pred == "Draw" else ""
            away_bold = "bold" if pred == f"{row['away_team']} win" else ""

            fixture_rows += f"""
            <div class="fixture">
                <div class="fixture-date">{row['date']}</div>
                <div class="fixture-teams">
                    <span class="team {home_bold}">{row['home_team']}</span>
                    <span class="vs">vs</span>
                    <span class="team {away_bold}">{row['away_team']}</span>
                </div>
                <div class="prob-bars">
                    <div class="bar-group">
                        <div class="bar-label {home_bold}">{row['home_win%']}</div>
                        <div class="bar-track"><div class="bar home-bar" style="width:{home_pct}%"></div></div>
                    </div>
                    <div class="bar-group draw-group">
                        <div class="bar-label {draw_bold}">{row['draw%']}</div>
                        <div class="bar-track"><div class="bar draw-bar" style="width:{draw_pct}%"></div></div>
                    </div>
                    <div class="bar-group">
                        <div class="bar-label {away_bold}">{row['away_win%']}</div>
                        <div class="bar-track"><div class="bar away-bar" style="width:{away_pct}%"></div></div>
                    </div>
                </div>
                <div class="prediction-tag">⚽ {pred}</div>
            </div>"""

        groups_html += f"""
        <div class="group-card">
            <div class="group-header">GROUP {group}</div>
            <div class="group-body">
                <div class="standings-section">
                    <div class="section-label">STANDINGS</div>
                    <table class="standings-table">
                        <thead><tr>
                            <th>#</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>Pts</th>
                        </tr></thead>
                        <tbody>{standing_rows}</tbody>
                    </table>
                </div>
                <div class="fixtures-section">
                    <div class="section-label">FIXTURES & PREDICTIONS</div>
                    {fixture_rows}
                </div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2026 World Cup Predictions</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --gold: #f0b429;
    --gold2: #ffd700;
    --green: #10b981;
    --blue: #3b82f6;
    --draw: #8b5cf6;
    --text: #e2e8f0;
    --muted: #64748b;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    background-image: radial-gradient(ellipse at 20% 50%, rgba(240,180,41,0.04) 0%, transparent 60%),
                      radial-gradient(ellipse at 80% 20%, rgba(59,130,246,0.05) 0%, transparent 50%);
  }}

  header {{
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }}

  header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      45deg, transparent, transparent 40px,
      rgba(240,180,41,0.015) 40px, rgba(240,180,41,0.015) 80px
    );
  }}

  .header-eyebrow {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.4em;
    color: var(--gold);
    margin-bottom: 0.5rem;
  }}

  h1 {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.5rem, 6vw, 5rem);
    letter-spacing: 0.05em;
    line-height: 1;
    background: linear-gradient(135deg, #fff 0%, var(--gold) 50%, #fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  .header-sub {{
    margin-top: 0.75rem;
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.05em;
  }}

  .model-tag {{
    display: inline-block;
    margin-top: 1rem;
    padding: 0.3rem 1rem;
    border: 1px solid var(--gold);
    border-radius: 999px;
    font-size: 0.75rem;
    color: var(--gold);
    letter-spacing: 0.1em;
    font-weight: 500;
  }}

  .container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }}

  .groups-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
    gap: 1.5rem;
  }}

  .group-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeUp 0.5s ease both;
  }}

  .group-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
  }}

  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  .group-card:nth-child(1) {{ animation-delay: 0.05s; }}
  .group-card:nth-child(2) {{ animation-delay: 0.1s; }}
  .group-card:nth-child(3) {{ animation-delay: 0.15s; }}
  .group-card:nth-child(4) {{ animation-delay: 0.2s; }}
  .group-card:nth-child(5) {{ animation-delay: 0.25s; }}
  .group-card:nth-child(6) {{ animation-delay: 0.3s; }}

  .group-header {{
    background: linear-gradient(135deg, var(--surface2), var(--border));
    padding: 0.9rem 1.25rem;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.15em;
    color: var(--gold);
    border-bottom: 1px solid var(--border);
  }}

  .group-body {{
    display: grid;
    grid-template-columns: 1fr 1.4fr;
    gap: 0;
  }}

  .standings-section {{
    padding: 1rem;
    border-right: 1px solid var(--border);
  }}

  .fixtures-section {{
    padding: 1rem;
  }}

  .section-label {{
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
  }}

  .standings-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }}

  .standings-table thead tr {{
    border-bottom: 1px solid var(--border);
  }}

  .standings-table th {{
    text-align: center;
    padding: 0.35rem 0.25rem;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-weight: 600;
  }}

  .standings-table th:nth-child(2) {{ text-align: left; }}

  .standings-table td {{
    padding: 0.5rem 0.25rem;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}

  .standings-table td.team-name {{ text-align: left; font-weight: 500; font-size: 0.8rem; }}
  .standings-table td.pos {{ color: var(--muted); font-size: 0.75rem; }}
  .standings-table td.pts {{ font-weight: 700; color: var(--gold); }}

  .standings-table tr.advancing {{
    background: rgba(16, 185, 129, 0.06);
  }}

  .standings-table tr.advancing td.team-name {{
    color: var(--green);
  }}

  .badge {{
    display: inline-block;
    margin-left: 0.4rem;
    padding: 0.1rem 0.4rem;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 4px;
    font-size: 0.55rem;
    color: var(--green);
    letter-spacing: 0.08em;
    vertical-align: middle;
    font-weight: 700;
  }}

  .fixture {{
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}

  .fixture:last-child {{ border-bottom: none; }}

  .fixture-date {{
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
  }}

  .fixture-teams {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
  }}

  .team {{ font-size: 0.82rem; font-weight: 400; color: var(--text); flex: 1; }}
  .team:last-child {{ text-align: right; }}
  .team.bold {{ font-weight: 700; color: #fff; }}
  .vs {{ font-size: 0.65rem; color: var(--muted); flex-shrink: 0; }}

  .prob-bars {{
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0.3rem;
    align-items: center;
  }}

  .bar-group {{
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }}

  .draw-group {{ align-items: center; padding: 0 0.25rem; }}

  .bar-label {{
    font-size: 0.65rem;
    color: var(--muted);
    font-weight: 500;
  }}
  .bar-label.bold {{ color: #fff; font-weight: 700; }}

  .bar-track {{
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }}

  .bar {{ height: 100%; border-radius: 2px; transition: width 1s ease; }}
  .home-bar {{ background: var(--blue); }}
  .draw-bar {{ background: var(--draw); }}
  .away-bar {{ background: var(--gold); margin-left: auto; }}

  .draw-group .bar-track {{ width: 100%; }}

  .prediction-tag {{
    margin-top: 0.35rem;
    font-size: 0.7rem;
    color: var(--gold);
    font-weight: 600;
    letter-spacing: 0.03em;
  }}

  footer {{
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
  }}

  @media (max-width: 700px) {{
    .groups-grid {{ grid-template-columns: 1fr; }}
    .group-body {{ grid-template-columns: 1fr; }}
    .standings-section {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}

  .section-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.1em;
    color: var(--gold);
    text-align: center;
    margin: 3rem 0 1.5rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
  }}

  .bracket-round {{
    margin-bottom: 2rem;
  }}

  .round-header {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 0.15em;
    color: var(--text);
    margin-bottom: 0.9rem;
    text-align: center;
  }}

  .matchup-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1rem;
  }}

  .matchup-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
  }}

  .matchup-date {{
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }}

  .matchup-row {{
    display: grid;
    grid-template-columns: 1fr 3fr auto;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
  }}

  .matchup-team {{
    font-size: 0.85rem;
    color: var(--text);
  }}

  .matchup-team.winner {{
    font-weight: 700;
    color: var(--green);
  }}

  .win-badge {{
    display: inline-block;
    margin-left: 0.3rem;
    padding: 0.05rem 0.35rem;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 4px;
    font-size: 0.55rem;
    color: var(--green);
    letter-spacing: 0.06em;
    font-weight: 700;
    vertical-align: middle;
  }}

  .adv-bar-track {{
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }}

  .adv-bar {{ height: 100%; border-radius: 3px; transition: width 1s ease; }}
  .home-adv {{ background: var(--blue); }}
  .away-adv {{ background: var(--gold); }}

  .matchup-pct {{
    font-size: 0.75rem;
    color: var(--muted);
    text-align: right;
    min-width: 3.2rem;
  }}

  .matchup-confidence {{
    margin-top: 0.5rem;
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 600;
  }}

  .conf-high {{ color: var(--green); }}
  .conf-medium {{ color: var(--gold); }}
  .conf-low {{ color: var(--muted); }}

  .final-spotlight {{
    margin-top: 3rem;
    padding: 2.5rem 1.5rem;
    background: linear-gradient(160deg, var(--surface2), var(--surface));
    border: 1px solid var(--gold);
    border-radius: 16px;
    text-align: center;
  }}

  .final-eyebrow {{
    font-size: 0.8rem;
    letter-spacing: 0.25em;
    color: var(--gold);
    margin-bottom: 1.5rem;
    font-weight: 600;
  }}

  .final-teams {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2.5rem;
    margin-bottom: 1.5rem;
  }}

  .final-team-name {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(1.8rem, 5vw, 3rem);
    letter-spacing: 0.05em;
    color: var(--text);
  }}

  .final-team.winner .final-team-name {{ color: var(--gold2); }}

  .final-team-pct {{
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: 0.25rem;
  }}

  .final-team.winner .final-team-pct {{ color: var(--gold); font-weight: 700; }}

  .final-vs {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    color: var(--muted);
    letter-spacing: 0.1em;
  }}

  .final-bar-track {{
    display: flex;
    height: 10px;
    max-width: 500px;
    margin: 0 auto 1.25rem;
    border-radius: 5px;
    overflow: hidden;
  }}

  .final-bar-home {{ background: var(--blue); }}
  .final-bar-away {{ background: var(--gold); }}

  .final-prediction {{
    font-size: 0.95rem;
    color: var(--text);
  }}

  @media (max-width: 700px) {{
    .final-teams {{ gap: 1.2rem; }}
  }}
</style>
</head>
<body>
<header>
  <div class="header-eyebrow">ML-Powered Tournament Analysis</div>
  <h1>2026 World Cup Predictions</h1>
  <div class="header-sub">Group Stage &amp; Knockout Stage &middot; XGBoost Model &middot; Elo + Form Features</div>
  <div class="model-tag">{f"{accuracy:.1f}% GROUP STAGE ACCURACY" if accuracy is not None else "58.3% TEST ACCURACY"} &middot; 72 FIXTURES TRACKED</div>
</header>
<div class="container">
  <div class="groups-grid">
    {groups_html}
  </div>
  {knockout_html}
  {final_html}
</div>
<footer>
  Built with Python · XGBoost · MLflow · FastAPI &nbsp;|&nbsp; Predictions based on historical Elo ratings and team form
</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {output_path}")
    print("Open it in your browser to view!")

if __name__ == "__main__":
    generate_dashboard("data/group_stage_predictions.csv", "dashboard.html")