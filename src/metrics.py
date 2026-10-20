"""
Evaluation Metrics for Traffic Flow Forecasting (PS-003).
Computes MAE, RMSE, MAPE, and R-squared on ground truth vs predicted traffic flows.
"""

from typing import Dict
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression metrics for traffic flow forecasting.
    Expects 1D or flattened arrays in original physical units (vehicles / interval).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}

    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Mean Absolute Percentage Error (with safe epsilon)
    nonzero_mask = np.abs(y_true) > 1e-3
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    else:
        mape = 0.0

    # R-squared
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if ss_tot > 1e-6:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 0.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "r2": round(r2, 4)
    }
