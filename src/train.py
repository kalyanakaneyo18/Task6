"""Steps 5-8 - Train, evaluate, tune, and save the best model.

Flow:
  1. Load data and split into train/test.
  2. Train Linear Regression, Random Forest, and XGBoost
     (each with and without feature engineering).
  3. Evaluate with MAE / RMSE / R² / MAPE and print a comparison table.
  4. Optionally hyperparameter tune the best candidates.
  5. Save the final preprocessing + model pipeline to models/.

Run: python -m src.train [--tune]
"""

import argparse
import json
import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from src.config import (
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_data_for_ml
from src.evaluation import compute_metrics, format_comparison_table
from src.preprocessing import build_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, use_engineering=True):
    results = []
    for name, model in get_models().items():
        pipe = build_full_pipeline(model, use_engineering=use_engineering)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics.update(
            {
                "model": name,
                "engineering": "yes" if use_engineering else "no",
            }
        )
        results.append(metrics)
        logger.info(
            "%s (eng=%s): R2=%.4f RMSE=%.0f MAPE=%.2f%%",
            name,
            use_engineering,
            metrics["R2"],
            metrics["RMSE"],
            metrics["MAPE"],
        )
    return results


def save_final_pipeline(X_train, y_train, model_name: str, use_engineering: bool):
    """Refit the best model on all training data and persist the pipeline."""
    model = get_models()[model_name]
    pipeline = build_full_pipeline(model, use_engineering=use_engineering)
    pipeline.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Saved final pipeline to %s", MODEL_PATH)
    return pipeline


def tune_model(X_train, y_train, model_name: str, use_engineering: bool):
    """GridSearchCV on the best candidate pipeline."""
    model = get_models()[model_name]
    pipe = build_full_pipeline(model, use_engineering=use_engineering)

    if model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }
    elif model_name == "XGBoost":
        param_grid = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 6],
        }
    else:
        return pipe

    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    logger.info("Best params for %s: %s", model_name, grid.best_params_)
    logger.info("Best CV RMSE for %s: %.4f", model_name, -grid.best_score_)
    return grid.best_estimator_


def main() -> None:
    parser = argparse.ArgumentParser(description="Train house price models.")
    parser.add_argument("--tune", action="store_true", help="run hyperparameter tuning")
    args = parser.parse_args()

    X, y = load_data_for_ml()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    results = []
    results += train_and_evaluate(X_train, X_test, y_train, y_test, True)
    results += train_and_evaluate(X_train, X_test, y_train, y_test, False)

    print("\n" + format_comparison_table(results))

    best = min(
        (r for r in results if r["model"] in ("Random Forest", "XGBoost")),
        key=lambda r: r["RMSE"],
    )
    logger.info("Best model overall: %s", best)

    best_pipe = None
    if args.tune:
        best_pipe = tune_model(
            X_train, y_train, best["model"], best["engineering"] == "yes"
        )

    final_model_name = best["model"]
    final_engineering = best["engineering"] == "yes"
    if best_pipe is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(best_pipe, MODEL_PATH)
        logger.info("Saved tuned pipeline to %s", MODEL_PATH)
    else:
        save_final_pipeline(X_train, y_train, final_model_name, final_engineering)

    metadata = {
        "best_model": final_model_name,
        "feature_engineering": final_engineering,
        "test_metrics": {k: v for k, v in best.items() if k in ("MAE", "RMSE", "R2", "MAPE")},
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Metrics written to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
