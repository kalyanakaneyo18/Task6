"""Shared data loading utilities."""

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET, ID_COLUMN


def load_raw_data() -> pd.DataFrame:
    """Load the raw house price dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def load_data_for_ml() -> tuple[pd.DataFrame, pd.Series]:
    """Load data and split into features and target (ID column dropped)."""
    df = load_raw_data()
    X = df.drop(columns=[ID_COLUMN, TARGET])
    y = df[TARGET]
    return X, y
