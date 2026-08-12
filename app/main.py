"""Step 9 - FastAPI backend for the Real Estate Price Prediction System.

Endpoints:
  GET  /health  - service health check
  POST /predict - returns a predicted property price
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import PredictionResponse, PropertyRequest
from src.logging_config import setup_logger
from src.predict import load_model, predict_price

logger = setup_logger("api")

PREDICTION_COUNT = 0
ERROR_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API, loading model...")
    try:
        load_model()
        logger.info("Model loaded successfully.")
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.error("Failed to load model on startup: %s", exc)
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Real Estate Price Prediction API",
    description="Predicts property prices from area, rooms, age, location and type.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "%s %s -> %d (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health", tags=["system"])
async def health() -> dict:
    """Return service health status and model availability."""
    try:
        load_model()
        model_ready = True
    except Exception:
        model_ready = False
    return {
        "status": "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "total_predictions": PREDICTION_COUNT,
        "total_errors": ERROR_COUNT,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(property_input: PropertyRequest) -> PredictionResponse:
    """Predict the price of a property from its raw features."""
    global PREDICTION_COUNT, ERROR_COUNT
    start = time.perf_counter()
    try:
        price = predict_price(
            area=property_input.area,
            bedrooms=property_input.bedrooms,
            bathrooms=property_input.bathrooms,
            age=property_input.age,
            location=property_input.location,
            property_type=property_input.property_type,
        )
        PREDICTION_COUNT += 1
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Prediction request completed in %.2f ms -> price=%.2f",
            duration_ms,
            price,
        )
        return PredictionResponse(predicted_price=price)
    except Exception as exc:
        ERROR_COUNT += 1
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    global ERROR_COUNT
    ERROR_COUNT += 1
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "Real Estate Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
