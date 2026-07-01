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

# Knockout response schema
class KnockoutResponse(BaseModel):
    home_team: str
    away_team: str
    home_advance_probability: float
    away_advance_probability: float
    extra_time_probability: float
    predicted_outcome: str
    confidence: str

@app.post("/predict_knockout", response_model=KnockoutResponse)
def predict_knockout(request: MatchRequest):
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

        # Get base probabilities
        proba = model.predict_proba(features)[0]
        away_win = float(proba[0])
        draw = float(proba[1])
        home_win = float(proba[2])

        # In knockout football, draws lead to extra time and penalties
        # Redistribute draw probability proportionally based on relative
        # win probabilities, since the stronger team tends to win shootouts
        total_win = home_win + away_win
        home_share = home_win / total_win if total_win > 0 else 0.5
        away_share = away_win / total_win if total_win > 0 else 0.5

        home_advance = round(home_win + (draw * home_share), 4)
        away_advance = round(away_win + (draw * away_share), 4)

        # Determine confidence level based on margin
        margin = abs(home_advance - away_advance)
        if margin > 0.25:
            confidence = "High"
        elif margin > 0.12:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Predicted outcome
        if home_advance > away_advance:
            outcome = f"{request.home_team} advance"
        else:
            outcome = f"{request.away_team} advance"

        logger.info(
            f"Knockout prediction: {request.home_team} vs {request.away_team} "
            f"-> {outcome} ({confidence} confidence)"
        )

        return KnockoutResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            home_advance_probability=home_advance,
            away_advance_probability=away_advance,
            extra_time_probability=round(draw, 4),
            predicted_outcome=outcome,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Knockout prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))