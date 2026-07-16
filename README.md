# World Cup 2026 Match Predictor

A full end-to-end machine learning project that predicts match outcomes for the 2026 FIFA World Cup. The project covers data ingestion, feature engineering, model training with experiment tracking, a REST API for live predictions, group stage and knockout stage prediction pipelines, a live results tracker, and an interactive HTML dashboard to visualize results — all running entirely on your local machine.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data](#data)
- [How to Run](#how-to-run)
- [Model and Features](#model-and-features)
- [Results](#results)
- [Next Steps](#next-steps)

---

## Project Overview

This project was built to demonstrate production-oriented data science skills, including writing modular Python scripts, tracking machine learning experiments, and serving model predictions via a REST API — all designed to run locally with no cloud dependencies.

The model predicts one of three outcomes for any international football match: home win, draw, or away win. It is trained on over 45,000 historical international match results and uses Elo ratings and rolling team form as its core features.

The group stage is now complete. All 72 group stage fixtures have been scored against their actual results, and `src/track_results.py` maintains a running accuracy log. The knockout stage predictor (`src/predict_knockout.py`) carries Elo and form ratings forward through each completed round — Round of 32, Round of 16, quarterfinals, and semifinals — and predicts advancement probabilities for the final, culminating in a Spain vs. Argentina final prediction. The HTML dashboard visualizes the full tournament: group standings, knockout bracket predictions round by round, and a spotlight section for the final.

---

## Technologies Used

- **Python 3.11** — core language for all scripts and pipelines
- **pandas, numpy** — data manipulation and feature engineering
- **XGBoost, LightGBM, scikit-learn** — model training and evaluation
- **MLflow** — experiment tracking, model registry, and artifact logging
- **FastAPI, uvicorn** — REST API for serving predictions
- **pytest** — unit testing
- **python-dotenv** — environment variable management

---

## Project Structure

```
worldcup-predictor/
├── data/
│   ├── results.csv                   # Raw match data (download separately, see Data section)
│   ├── shootouts.csv                 # Penalty shootout results (download separately)
│   ├── goalscorers.csv               # Goalscorer data (download separately)
│   ├── group_stage_predictions.csv   # Predictions + actual results for all 72 group stage fixtures
│   ├── prediction_tracker.csv        # Predicted vs. actual outcome log with running accuracy
│   └── knockout_predictions.csv      # Advancement probabilities for every knockout round
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Data loading and cleaning
│   ├── features.py                   # Elo rating system and form feature engineering
│   ├── train.py                      # Model training with MLflow experiment tracking
│   ├── app.py                        # FastAPI prediction endpoints (/predict, /predict_knockout)
│   ├── predict_matches.py            # Group stage fixtures, predictions, and actual results
│   ├── group_standings.py            # Actual group standings and advancement (top 2 + best 8 third-place)
│   ├── track_results.py              # Tracks predicted vs. actual outcomes and running accuracy
│   ├── predict_knockout.py           # Knockout stage predictor, rerunnable per round through the final
│   └── generate_dashboard.py         # Generates the HTML dashboard (group stage + knockout bracket)
├── tests/
│   └── test_features.py              # Unit tests for feature engineering functions
├── configs/
│   └── config.yaml                   # Centralized configuration for model and features
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- Git
- A Kaggle account (to download the dataset)

### Steps

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/worldcup-predictor.git
cd worldcup-predictor
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the data (see the Data section below) and place all CSV files in the `data/` folder.

---

## Data

This project uses the **International Football Results** dataset by Mart Jurisoo, available for free on Kaggle.

Dataset URL: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Download the following files and place them in the `data/` folder:

- `results.csv` — core match results (required)
- `shootouts.csv` — penalty shootout outcomes (recommended)
- `goalscorers.csv` — individual goal data (optional)

Data files are excluded from version control via `.gitignore`. Anyone cloning this repo must download the data separately from the link above.

---

## How to Run

All scripts are run from the project root directory with the virtual environment active.

### 1. Run the data pipeline

Loads and cleans the raw match data:

```bash
python src/data_pipeline.py
```

### 2. Generate features

Calculates Elo ratings and rolling form for all historical matches:

```bash
python src/features.py
```

### 3. Train a model

Trains a model and logs the experiment to MLflow. The `--model` argument accepts `logistic_regression`, `random_forest`, `xgboost`, or `lightgbm`:

```bash
python src/train.py --model xgboost --n_estimators 100 --learning_rate 0.1
```

To view all experiment runs, start the MLflow UI in a separate terminal:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

### 4. Run all four models for comparison

```bash
python src/train.py --model logistic_regression
python src/train.py --model random_forest
python src/train.py --model xgboost
python src/train.py --model lightgbm
```

### 5. Start the prediction API

```bash
uvicorn src.app:app --reload
```

The API will be available at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

Example prediction request:

```json
POST /predict
{
  "home_team": "England",
  "away_team": "France",
  "home_elo": 1958.0,
  "away_elo": 2005.7,
  "home_form": 0.6,
  "away_form": 0.7,
  "neutral": false
}
```

Example response:

```json
{
  "home_team": "England",
  "away_team": "France",
  "away_win_probability": 0.3452,
  "draw_probability": 0.3958,
  "home_win_probability": 0.259,
  "predicted_outcome": "Draw"
}
```

The `/predict_knockout` endpoint takes the same request shape but redistributes draw probability proportionally between the two teams (since knockout matches can't end in a draw), returning each team's probability of advancing.

### 6. Generate group stage predictions

Runs the model against all 72 group stage fixtures and compares them to the actual final scores, saving the result to `data/group_stage_predictions.csv`:

```bash
python src/predict_matches.py
```

### 7. View group standings

Displays the actual final group standings and advancement (top 2 per group, plus the 8 best third-place teams, using FIFA's points/goal-difference/goals-for tiebreakers):

```bash
python src/group_standings.py
```

### 8. Track prediction results

Compares predicted outcomes to actual results for every group stage match and logs it to `data/prediction_tracker.csv`, printing overall accuracy:

```bash
python src/track_results.py
```

### 9. Generate knockout stage predictions

Predicts advancement probabilities for a knockout round using Elo/form ratings rolled forward through every match completed so far, and saves the result to `data/knockout_predictions.csv`. Defaults to the Round of 32; pass `--round` to predict a later round once the previous round's fixtures are known:

```bash
python src/predict_knockout.py --round round_of_32
python src/predict_knockout.py --round round_of_16
python src/predict_knockout.py --round quarterfinals
python src/predict_knockout.py --round semifinals
python src/predict_knockout.py --round final
```

### 10. Generate the HTML dashboard

Creates a visual interactive dashboard at `dashboard.html`, including group standings, the full knockout bracket with advancement probabilities, and a spotlight section for the final:

```bash
python src/generate_dashboard.py
```

Open `dashboard.html` directly in any browser to view it.

---

## Model and Features

### Feature Engineering

**Elo Ratings:** Each team is assigned a dynamic strength rating that updates after every match based on the result and the relative strength of the opponent. Teams start at a baseline of 1500. Competitive matches use a higher K-factor (40) than friendlies (20), reflecting the greater significance of tournament results. Knockout stage predictions roll this same Elo update forward through every completed tournament match, so each round's ratings reflect the tournament so far.

**Rolling Form:** A rolling average of each team's results over their last 5 matches, calculated on a 0 to 1 scale (1 = win, 0.5 = draw, 0 = loss).

**Feature columns used for training:**

- `home_elo` — Elo rating of the home team at match time
- `away_elo` — Elo rating of the away team at match time
- `elo_diff` — difference between home and away Elo ratings
- `home_form` — home team rolling form over last 5 matches
- `away_form` — away team rolling form over last 5 matches
- `form_diff` — difference in form between home and away teams
- `neutral` — whether the match is played at a neutral venue

### Target Variable

Three-class classification:

- 2 = home win
- 1 = draw
- 0 = away win

### Model Comparison

All experiments tracked in MLflow. Results on a 20% held-out test set:

| Model | Accuracy |
|---|---|
| XGBoost (100 est, lr=0.10) | 58.28% |
| LightGBM | 57.87% |
| Logistic Regression | 57.84% |
| XGBoost (200 est, lr=0.05) | 58.23% |
| XGBoost (300 est, lr=0.01) | 58.09% |
| Random Forest | 53.31% |

XGBoost with default hyperparameters was selected as the final model and registered in the MLflow Model Registry. A 58% accuracy on a 3-class problem compares favorably to professional betting models, which typically achieve 55 to 65% on football match prediction.

---

## Results

### Group Stage

With the group stage complete, `src/track_results.py` scores every one of the 72 fixtures against its actual outcome:

- **Correct predictions:** 43 / 72
- **Overall accuracy:** 59.7%

This is in line with the model's held-out test accuracy (58.3%), suggesting the model generalized well to the real tournament rather than overfitting to historical data. The full match-by-match breakdown is in `data/prediction_tracker.csv`, and the group-by-group view (predicted probabilities, prediction, and actual result) is in `data/group_stage_predictions.csv`.

### Knockout Stage

`data/knockout_predictions.csv` holds advancement probabilities for every knockout round played so far (Round of 32 through the semifinals) plus the final. The model correctly picked the advancing team in every round through the semifinals when compared against the real bracket results. For the final, Spain enters with an Elo rating of roughly 2100 and Argentina roughly 2083 — both updated from their pre-tournament baselines using their actual group stage and knockout results — giving Spain a narrow ~51/49 edge to win the tournament. See `dashboard.html` for the full visual breakdown, including the final's spotlight prediction.

---

## Next Steps

### Unit Testing

`tests/test_features.py` will be expanded to include unit tests for the Elo calculation logic, form feature generation, and the match result categorization function. This reinforces software engineering best practices and ensures the feature pipeline remains reliable as the codebase grows.

### Configuration Management

Hardcoded values across scripts (Elo base rating, K-factors, feature column names, model name and version) will be consolidated into `configs/config.yaml` and loaded at runtime. This makes the project easier to modify and is standard practice in production ML codebases.

### Streamlit Prediction Interface

A lightweight Streamlit app will be added to allow interactive predictions: select any two international teams, adjust their Elo and form values, and get an instant probability breakdown. This serves as a more accessible demo than the raw API and is easy to showcase in interviews or share with others.
