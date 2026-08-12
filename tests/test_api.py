"""Tests for the FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


VALID_PAYLOAD = {
    "area": 3000,
    "bedrooms": 3,
    "bathrooms": 2,
    "age": 10,
    "location": "City Center",
    "property_type": "Villa",
}


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_price" in body
    assert body["predicted_price"] > 0


def test_predict_invalid_area(client):
    payload = {**VALID_PAYLOAD, "area": -100}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_bedrooms(client):
    payload = {**VALID_PAYLOAD, "bedrooms": 99}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["location"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_empty_body(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_accepts_case_insensitive_location(client):
    payload = {**VALID_PAYLOAD, "location": "city center"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
