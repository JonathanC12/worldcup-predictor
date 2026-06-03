import pandas as pd
import numpy as np
import logging
import argparse
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from data_pipeline import load_raw_data, clean_data
from features import calculate_elo_ratings, add_form_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Features the model will train on
FEATURE_COLS = ["home_elo", "away_elo", "elo_diff", "home_form", "away_form", "form_diff", "neutral"]

def get_model(model_name: str, n_estimators: int, learning_rate: float):
    """Return a model instance based on name and hyperparameters."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=n_estimators),
        "xgboost": XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, eval_metric="mlogloss"),
        "lightgbm": LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]

def prepare_training_data(df: pd.DataFrame):
    """Prepare features and target variable."""
    df = df.copy()

    # Convert neutral venue to integer
    df["neutral"] = df["neutral"].astype(int)

    # Target: 1 = home win, 0 = draw, -1 = away win
    def categorize_result(result):
        if result == 1.0:
            return 2  # home win
        elif result == 0.5:
            return 1  # draw
        else:
            return 0  # away win

    df["target"] = df["result"].apply(categorize_result)

    X = df[FEATURE_COLS]
    y = df["target"]
    return X, y

def train(model_name: str, n_estimators: int, learning_rate: float, test_size: float):
    """Full training pipeline with MLflow tracking."""

    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_raw_data()
    df = clean_data(df)
    df = calculate_elo_ratings(df)
    df = add_form_features(df)
    X, y = prepare_training_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Start MLflow run
    mlflow.set_experiment("worldcup-predictor")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features", FEATURE_COLS)

        # Train
        logger.info(f"Training {model_name}...")
        model = get_model(model_name, n_estimators, learning_rate)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("log_loss", loss)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Model: {model_name} | Accuracy: {accuracy:.4f} | Log Loss: {loss:.4f}")
        print(f"\n✅ {model_name} | Accuracy: {accuracy:.4f} | Log Loss: {loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a World Cup match predictor")
    parser.add_argument("--model", type=str, default="xgboost", help="Model to train")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    args = parser.parse_args()

    train(args.model, args.n_estimators, args.learning_rate, args.test_size)