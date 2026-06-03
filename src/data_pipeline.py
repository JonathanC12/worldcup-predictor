import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build path to data folder relative to this file's location
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_raw_data(filename: str = "results.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Data cleaned and sorted by date")
    return df

if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    print(df.head())
    print(df.shape)