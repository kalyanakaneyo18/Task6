"""Step 8 - Load the saved pipeline and make predictions.

The saved pipeline accepts raw property information (a DataFrame with
columns Area, Bedrooms, Bathrooms, Age, Location, Property_Type) and
returns predicted prices.
"""

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH

logger = logging.getLogger(__name__)

_model = None


def load_model(force: bool = False):
    """Load (and cache) the saved pipeline from disk."""
    global _model
    if _model is None or force:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python -m src.train --tune` first."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


def predict_price(
    area: float,
    bedrooms: int,
    bathrooms: int,
    age: int,
    location: str,
    property_type: str,
) -> float:
    """Predict the price of a single property from raw inputs."""
    row = pd.DataFrame(
        [
            {
                "Area": float(area),
                "Bedrooms": int(bedrooms),
                "Bathrooms": int(bathrooms),
                "Age": int(age),
                "Location": location,
                "Property_Type": property_type,
            }
        ]
    )
    model = load_model()
    return float(np.round(model.predict(row)[0], 2))
