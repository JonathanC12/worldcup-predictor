# World Cup 2026 Match Predictor

A full end-to-end machine learning project that predicts match outcomes for the 2026 FIFA World Cup group stage. The project covers data ingestion, feature engineering, model training with experiment tracking, a REST API for live predictions, and an interactive HTML dashboard to visualize results.

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

This project was built to demonstrate production-oriented data science skills, including writing modular Python scripts, tracking machine learning experiments, serving model predictions via a REST API, and deploying containerized applications to the cloud.

The model predicts one of three outcomes for any international football match: home win, draw, or away win. It is trained on over 45,000 historical international match results and uses Elo ratings and rolling team form as its core features. All 72 group stage fixtures for the 2026 World Cup are predicted and displayed in a visual HTML dashboard, including projected group standings with advancement status.

---

## Technologies Used

- **Python 3.11** — core language for all scripts and pipelines
- **pandas, numpy** — data manipulation and feature engineering
- **XGBoost, LightGBM, scikit-learn** — model training and evaluation
- **MLflow** — experiment tracking, model registry, and artifact logging
- **FastAPI, uvicorn** — REST API for serving predictions
- **Docker** — containerization for deployment
- **AWS (S3, ECR, ECS, CloudWatch)** — cloud storage and deployment
- **pytest** — unit testing
- **python-dotenv** — environment variable management
- **boto3** — AWS SDK for Python

---

## Project Structure

```
worldcup-predictor/
├── data/
│   ├── results.csv                   # Raw match data (download separately, see Data section)
│   ├── shootouts.csv                 # Penalty shootout results (download separately)
│   ├── goalscorers.csv               # Goalscorer data (download separately)
│   ├── group_stage_predictions.csv   # Generated predictions for all 72 fixtures
│   └── prediction_tracker.csv        # Live result tracking during the tournament
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Data loading and cleaning
│   ├── features.py                   # Elo rating system and form feature engineering
│   ├── train.py                      # Model training with MLflow experiment tracking
│   ├── app.py                        # FastAPI prediction endpoint
│   ├── predict_matches.py            # Batch predictions for all group stage fixtures
│   ├── group_standings.py            # Predicted group standings with advancement logic
│   ├── generate_dashboard.py         # Generates the HTML visual dashboard
│   └── track_results.py              # Logs actual results vs predictions during tournament
├── tests/
│   └── test_features.py              # Unit tests for feature engineering functions
├── configs/
│   └── config.yaml                   # Centralized configuration for model and features
├── Dockerfile                        # Container definition for API deployment
├── .dockerignore
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
- Docker Desktop (for containerized deployment)
- An AWS account (for cloud deployment)

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

5. Configure AWS credentials if using cloud features:

```bash
aws configure
```

Create a `.env` file in the project root for any additional environment variables:

```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

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

### 6. Generate group stage predictions

Runs predictions for all 72 group stage fixtures and saves results to `data/group_stage_predictions.csv`:

```bash
python src/predict_matches.py
```

### 7. View group standings

Displays predicted group standings with advancement status (top 2 per group advance automatically; the 8 best third-place teams also advance):

```bash
python src/group_standings.py
```

### 8. Generate the HTML dashboard

Creates a visual interactive dashboard at `dashboard.html`:

```bash
python src/generate_dashboard.py
```

Open `dashboard.html` directly in any browser to view predictions, probability bars, and group standings.

### 9. Run with Docker

Build the image:

```bash
docker build -t worldcup-predictor .
```

Run the container:

```bash
docker run -p 8000:8000 worldcup-predictor
```

---

## Model and Features

### Feature Engineering

**Elo Ratings:** Each team is assigned a dynamic strength rating that updates after every match based on the result and the relative strength of the opponent. Teams start at a baseline of 1500. Competitive matches use a higher K-factor (40) than friendlies (20), reflecting the greater significance of tournament results.

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

Group stage predictions for all 72 fixtures are stored in `data/group_stage_predictions.csv`. The full interactive dashboard is available at `dashboard.html` and includes probability bars for every match, predicted group standings, and advancement indicators for the top 2 teams per group and potential third-place qualifiers.

---

## Next Steps

### Live Tournament Tracking

The 2026 World Cup group stage runs from June 11 to June 28. The next development priority is `src/track_results.py`, a script that accepts actual match results as input, compares them against the stored predictions, and logs each outcome to `data/prediction_tracker.csv`. This will produce a running accuracy log across the entire tournament, allowing real-time evaluation of model calibration and performance.

### Knockout Stage Predictions

Once the Round of 32 bracket is determined, predictions will be extended to cover all knockout matches. This phase will incorporate `shootouts.csv` to improve handling of matches that go to extra time and penalties, and may introduce a separate model or adjusted logic for knockout-stage dynamics.

### Docker and Cloud Deployment

The project includes a `Dockerfile` and is ready for containerized deployment. The remaining steps are to enable hardware virtualization in BIOS to run Docker Desktop locally, build and test the image, push it to AWS ECR, and deploy it via AWS ECS Fargate. Model artifacts will be migrated from local MLflow storage to S3, and CloudWatch will be configured for monitoring prediction requests and latency in production.

### Unit Testing

`tests/test_features.py` will be expanded to include unit tests for the Elo calculation logic, form feature generation, and the match result categorization function. This reinforces software engineering best practices and ensures the feature pipeline remains reliable as the codebase grows.

### Configuration Management

Hardcoded values across scripts (Elo base rating, K-factors, feature column names, model name and version) will be consolidated into `configs/config.yaml` and loaded at runtime. This makes the project easier to modify and is standard practice in production ML codebases.

### Streamlit Prediction Interface

A lightweight Streamlit app will be added to allow interactive predictions: select any two international teams, adjust their Elo and form values, and get an instant probability breakdown. This serves as a more accessible demo than the raw API and is easy to showcase in interviews or share with others.