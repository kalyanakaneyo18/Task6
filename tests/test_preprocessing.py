"""Tests for data loading and preprocessing."""

import pandas as pd
import pytest

from src.data_loader import load_data_for_ml, load_raw_data
from src.preprocessing import FeatureEngineering, build_full_pipeline
from sklearn.linear_model import LinearRegression


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


def test_raw_data_shape_and_columns(raw_df):
    assert raw_df.shape[1] == 8
    assert set(raw_df.columns) == {
        "Property_ID",
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "Location",
        "Property_Type",
        "Price",
    }


def test_no_missing_values(raw_df):
    assert raw_df.isna().sum().sum() == 0


def test_no_duplicates(raw_df):
    assert raw_df.duplicated().sum() == 0


def test_price_positive(raw_df):
    assert (raw_df["Price"] > 0).all()


def test_load_data_for_ml_drops_id_and_target():
    X, y = load_data_for_ml()
    assert "Property_ID" not in X.columns
    assert "Price" not in X.columns
    assert len(X) == len(y) == 300


def test_feature_engineering_adds_columns():
    eng = FeatureEngineering(enable=True)
    X, _ = load_data_for_ml()
    out = eng.fit_transform(X.head())
    for col in ["Area_per_Bedroom", "Total_Rooms", "Age_Group", "Has_Garden_Room_Budget"]:
        assert col in out.columns


def test_pipeline_fits_and_predicts():
    X, y = load_data_for_ml()
    pipe = build_full_pipeline(LinearRegression())
    pipe.fit(X.iloc[:50], y.iloc[:50])
    pred = pipe.predict(X.iloc[50:55])
    assert len(pred) == 5
    assert (pred > 0).all()


def test_pipeline_handles_unknown_category():
    X, y = load_data_for_ml()
    pipe = build_full_pipeline(LinearRegression())
    pipe.fit(X.iloc[:50], y.iloc[:50])
    unknown = X.iloc[[0]].copy()
    unknown["Location"] = "Mars Colony"
    pred = pipe.predict(unknown)
    assert len(pred) == 1
