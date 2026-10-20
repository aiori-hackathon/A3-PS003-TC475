"""
Traffic Client Simulator for Federated Learning (PS-003).
Represents a smart city intersection node with private localized traffic flow sensor data.
"""

from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model import TrafficGRU, get_weights_dict, set_weights_dict
from src.metrics import compute_metrics


class TrafficClient:
    """
    Virtual traffic intersection node.
    Data-locality approach: Trains locally on its sensor data; never transmits raw observations to the server.
    """
    def __init__(
        self,
        client_id: str,
        name: str,
        pattern_desc: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        raw_y_test: np.ndarray,
        scaler: Any,
        hidden_size: int = 32,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.name = name
        self.pattern_desc = pattern_desc
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.raw_y_test = raw_y_test
        self.scaler = scaler
        self.device = torch.device(device)

        # PyTorch model instance for local training
        self.model = TrafficGRU(input_size=1, hidden_size=hidden_size, num_layers=1, output_size=1).to(self.device)
        self.num_train_samples = len(X_train)

    def train_locally(
        self,
        global_weights: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        lr: float = 0.005,
        batch_size: int = 32
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Loads global model weights, performs local training on private dataset,
        and returns updated local weights, sample count, and final epoch loss.
        """
        # Load broadcast global weights into local model
        set_weights_dict(self.model, global_weights)
        self.model.train()

        # Build PyTorch DataLoader
        dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        final_loss = 0.0
        for epoch in range(local_epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            final_loss = total_loss / max(1, num_batches)

        # Return updated local weights and sample weight
        local_weights = get_weights_dict(self.model)
        return local_weights, self.num_train_samples, final_loss

    def evaluate(
        self,
        weights: Dict[str, torch.Tensor] = None
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluates model on local test set.
        Returns metrics dict (MAE, RMSE, MAPE, R2 in true vehicle scale),
        predicted values, and actual true values.
        """
        if weights is not None:
            set_weights_dict(self.model, weights)

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            preds_scaled = self.model(x_tensor).cpu().numpy()

        # Invert scaling to obtain real traffic flow predictions
        preds_real = self.scaler.inverse_transform(preds_scaled).flatten()
        actual_real = self.raw_y_test.flatten()

        metrics = compute_metrics(actual_real, preds_real)
        return metrics, preds_real, actual_real
