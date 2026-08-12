"""Central configuration for paths, column names, and model settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "house_prices.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "house_price_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"

LOGS_DIR = PROJECT_ROOT / "logs"
API_LOG_PATH = LOGS_DIR / "api.log"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "Price"
ID_COLUMN = "Property_ID"

FEATURES = ["Area", "Bedrooms", "Bathrooms", "Age", "Location", "Property_Type"]

NUMERICAL_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Age"]
CATEGORICAL_FEATURES = ["Location", "Property_Type"]

ENGINEERED_FEATURES = [
    "Area_per_Bedroom",
    "Total_Rooms",
    "Age_Group",
    "Has_Garden_Room_Budget",
]
