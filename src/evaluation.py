"""Regression evaluation helpers (MAE, RMSE, R², MAPE)."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return regression metrics as a dict with 4 decimal places."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAPE": round(
            float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred))
                                 / np.asarray(y_true))) * 100), 4
        ),
    }


def format_comparison_table(results: list[dict]) -> str:
    """Render the model comparison table as text."""
    header = f"{'Model':<32}{'Engineering':<12}{'MAE':>14}{'RMSE':>14}{'R2':>10}{'MAPE%':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['model']:<32}{r['engineering']:<12}"
            f"{r['MAE']:>14,}{r['RMSE']:>14,}{r['R2']:>10}{r['MAPE']:>10}"
        )
    return "\n".join(lines)
