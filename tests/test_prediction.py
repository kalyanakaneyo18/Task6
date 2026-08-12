"""Tests for model loading and prediction."""

import numpy as np
import pytest

from src.config import MODEL_PATH
from src.predict import load_model, predict_price


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_predicts_multiple(model):
    X_sample = {
        "area": [2500, 1200, 4800],
        "bedrooms": [3, 2, 5],
        "bathrooms": [2, 1, 3],
        "age": [10, 30, 5],
        "location": ["City Center", "Rural", "Suburb"],
        "property_type": ["Villa", "House", "Apartment"],
    }
    prices = [predict_price(*args) for args in zip(
        X_sample["area"],
        X_sample["bedrooms"],
        X_sample["bathrooms"],
        X_sample["age"],
        X_sample["location"],
        X_sample["property_type"],
    )]
    assert len(prices) == 3
    assert all(p > 0 for p in prices)


def test_prediction_is_reasonable():
    price = predict_price(3000, 3, 2, 10, "City Center", "Villa")
    # Prices in the dataset range from ~3.7M to ~58.7M
    assert 1_000_000 < price < 100_000_000


def test_predict_price_is_numeric():
    price = predict_price(1500, 2, 1, 25, "Rural", "House")
    assert isinstance(price, (int, float, np.floating))
