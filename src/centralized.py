"""
Centralized Baseline Training for Traffic Flow Forecasting (PS-003).
Combines all client data into a single central server dataset and trains a benchmark GRU.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model import TrafficGRU, get_weights_dict
from src.metrics import compute_metrics


def train_centralized_model(
    client_dict: Dict[str, dict],
    global_eval: dict,
    epochs: int = 10,
    lr: float = 0.005,
    batch_size: int = 32,
    hidden_size: int = 32,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Dict[str, Any]:
    """
    Trains a centralized benchmark GRU on the pooled training data of all clients.
    """
    dev = torch.device(device)

    # Pool all client training sets
    pooled_X = []
    pooled_y = []
    for c_id, c_info in client_dict.items():
        pooled_X.append(c_info["X_train"])
        pooled_y.append(c_info["y_train"])

    X_train_all = np.concatenate(pooled_X, axis=0)
    y_train_all = np.concatenate(pooled_y, axis=0)

    # Initialize model
    model = TrafficGRU(input_size=1, hidden_size=hidden_size, num_layers=1, output_size=1).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(X_train_all, dtype=torch.float32),
        torch.tensor(y_train_all, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Quick evaluation on global test set during training
        model.eval()
        with torch.no_grad():
            eval_x = torch.tensor(global_eval["X_test"], dtype=torch.float32).to(dev)
            eval_preds_scaled = model(eval_x).cpu().numpy()

        eval_preds_real = global_eval["scaler"].inverse_transform(eval_preds_scaled).flatten()
        eval_metrics = compute_metrics(global_eval["raw_y_test"].flatten(), eval_preds_real)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "mae": eval_metrics["mae"],
            "rmse": eval_metrics["rmse"]
        })

        if progress_callback:
            progress_callback(epoch, epochs, eval_metrics["mae"], eval_metrics["rmse"])

    # Final evaluation per client
    client_metrics = {}
    client_predictions = {}
    for c_id, c_info in client_dict.items():
        with torch.no_grad():
            cx = torch.tensor(c_info["X_test"], dtype=torch.float32).to(dev)
            c_pred_scaled = model(cx).cpu().numpy()

        c_pred_real = c_info["scaler"].inverse_transform(c_pred_scaled).flatten()
        c_actual_real = c_info["raw_y_test"].flatten()

        c_met = compute_metrics(c_actual_real, c_pred_real)
        client_metrics[c_id] = c_met
        client_predictions[c_id] = {
            "predictions": c_pred_real,
            "actual": c_actual_real
        }

    # Final global test set metrics
    with torch.no_grad():
        gx = torch.tensor(global_eval["X_test"], dtype=torch.float32).to(dev)
        g_pred_scaled = model(gx).cpu().numpy()

    g_pred_real = global_eval["scaler"].inverse_transform(g_pred_scaled).flatten()
    final_global_metrics = compute_metrics(global_eval["raw_y_test"].flatten(), g_pred_real)

    return {
        "model": model,
        "parameters": model.count_parameters(),
        "epochs": epochs,
        "mae": final_global_metrics["mae"],
        "rmse": final_global_metrics["rmse"],
        "mape": final_global_metrics["mape"],
        "r2": final_global_metrics["r2"],
        "history": history,
        "client_metrics": client_metrics,
        "client_predictions": client_predictions,
        "weights": get_weights_dict(model),
        "total_training_samples": len(X_train_all)
    }
