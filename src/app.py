import mlflow
import mlflow.sklearn
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the registered model from MLflow
MODEL_NAME = "worldcup-predictor-xgboost"
MODEL_VERSION = "1"

app = FastAPI(
    title="World Cup Match Predictor",
    description="Predicts the outcome of international football matches using Elo ratings and form",
    version="1.0.0"
)

# Load model at startup
logger.info(f"Loading model {MODEL_NAME} version {MODEL_VERSION}...")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
logger.info("Model loaded successfully")

# Request schema
class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    home_form: float = 0.5
    away_form: float = 0.5
    neutral: bool = False

# Response schema
class MatchResponse(BaseModel):
    home_team: str
    away_team: str
    away_win_probability: float
    draw_probability: float
    home_win_probability: float
    predicted_outcome: str

@app.get("/")
def root():
    return {"message": "World Cup Match Predictor API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict", response_model=MatchResponse)
def predict(request: MatchRequest):
    try:
        # Build feature vector
        features = pd.DataFrame([{
            "home_elo": request.home_elo,
            "away_elo": request.away_elo,
            "elo_diff": request.home_elo - request.away_elo,
            "home_form": request.home_form,
            "away_form": request.away_form,
            "form_diff": request.home_form - request.away_form,
            "neutral": int(request.neutral)
        }])

        # Get probabilities
        proba = model.predict_proba(features)[0]
        away_win_prob = round(float(proba[0]), 4)
        draw_prob = round(float(proba[1]), 4)
        home_win_prob = round(float(proba[2]), 4)

        # Determine predicted outcome
        max_prob = max(away_win_prob, draw_prob, home_win_prob)
        if max_prob == home_win_prob:
            outcome = f"{request.home_team} win"
        elif max_prob == draw_prob:
            outcome = "Draw"
        else:
            outcome = f"{request.away_team} win"

        logger.info(f"Prediction: {request.home_team} vs {request.away_team} -> {outcome}")

        return MatchResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            away_win_probability=away_win_prob,
            draw_probability=draw_prob,
            home_win_probability=home_win_prob,
            predicted_outcome=outcome
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))