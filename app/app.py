"""Step 10 - Simple Streamlit interface for the prediction model.

The user enters property details and the app returns an estimated price.
When the FastAPI backend is running (default localhost:8000) the app calls
the API; otherwise it falls back to in-process model prediction.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from src.predict import predict_price

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="House Price Predictor", page_icon="house", layout="centered")

st.title("Real Estate Price Prediction")
st.caption("Enter the property details below and click **Predict** to estimate its price.")


def get_api_options() -> tuple[list[str], list[str]]:
    return (["Rural", "Suburb", "City Center"], ["House", "Villa", "Apartment"])


with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", min_value=1.0, max_value=10000.0, value=2500.0)
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=5, value=3)
        age = st.slider("Age (years)", min_value=0, max_value=49, value=15)
    with col2:
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=3, value=2)
        location = st.selectbox("Location", get_api_options()[0])
        property_type = st.selectbox("Property Type", get_api_options()[1])

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {
        "area": area,
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "age": int(age),
        "location": location,
        "property_type": property_type,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        response.raise_for_status()
        price = response.json()["predicted_price"]
        source = "API"
    except requests.RequestException:
        price = predict_price(
            area, int(bedrooms), int(bathrooms), int(age), location, property_type
        )
        source = "local model"

    st.success(f"**Estimated price: {price:,.0f} PKR**")
    st.caption(f"Predicted by: {source}")

    with st.expander("Input summary"):
        st.dataframe(pd.DataFrame([payload]))