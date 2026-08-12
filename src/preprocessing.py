"""Step 3 & 4 - Data preprocessing and feature engineering.

Provides a reusable scikit-learn preprocessing pipeline plus a custom
feature engineering transformer. The same transformations are applied
during training and prediction so the saved pipeline can accept raw
property information.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ENGINEERED_FEATURES,
)

FEATURES_BASE = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds engineered features. Falls back to raw columns if features
    cannot be computed (e.g. division by zero bedrooms)."""

    def __init__(self, enable: bool = True):
        self.enable = enable

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.copy()
        if self.enable:
            X["Area_per_Bedroom"] = np.where(
                X["Bedrooms"] > 0, X["Area"] / X["Bedrooms"].replace(0, np.nan), 0.0
            )
            X["Total_Rooms"] = X["Bedrooms"] + X["Bathrooms"]
            X["Age_Group"] = pd.cut(
                X["Age"],
                bins=[-1, 10, 25, 49],
                labels=["New", "Moderate", "Old"],
            )
            X["Age_Group"] = X["Age_Group"].astype(str)
            X["Has_Garden_Room_Budget"] = (
                (X["Area"] > 2500) & (X["Bathrooms"] >= 2)
            ).astype(int)
        return X


def build_preprocessing_pipeline(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """ColumnTransformer that scales numeric features and one-hot encodes
    categorical features. Unknown categories are ignored safely."""
    numeric = numeric or NUMERICAL_FEATURES
    categorical = categorical or CATEGORICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_feature_pipeline(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """ColumnTransformer for pipelines that already include engineered
    features."""
    numeric = numeric or NUMERICAL_FEATURES + [
        "Area_per_Bedroom",
        "Total_Rooms",
        "Has_Garden_Room_Budget",
    ]
    categorical = categorical or CATEGORICAL_FEATURES + ["Age_Group"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_full_pipeline(model, use_engineering: bool = True) -> Pipeline:
    """Build the full pipeline: feature engineering + preprocessing + model."""
    if use_engineering:
        preprocessor = build_feature_pipeline()
        return Pipeline(
            [
                ("engineer", FeatureEngineering(enable=True)),
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
    return Pipeline(
        [
            ("preprocess", build_preprocessing_pipeline()),
            ("model", model),
        ]
    )
