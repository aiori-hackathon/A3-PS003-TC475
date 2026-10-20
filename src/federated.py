"""
Federated Learning Simulation with FedAvg and Communication Cost Tracking (PS-003).
Data-locality approach: Raw traffic data remains at simulated clients during federated training;
only model updates/weights are exchanged.
This prototype does not implement formal differential privacy, encryption-based secure
aggregation, or other formal privacy guarantees.
"""

from typing import Dict, List, Tuple, Any, Callable, Optional
import copy
import torch

from src.model import TrafficGRU, get_weights_dict, set_weights_dict, quantize_to_fp16
from src.client import TrafficClient
from src.communication import calculate_round_communication, calculate_total_communication
from src.metrics import compute_metrics


def fedavg_aggregate(
    weights_list: List[Dict[str, torch.Tensor]],
    sample_counts: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Weighted parameter averaging (FedAvg, McMahan et al.):
    W_{global} = sum_{k=1}^K (n_k / N) * W_k
    """
    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count cannot be zero for FedAvg aggregation.")

    aggregated_weights: Dict[str, torch.Tensor] = {}
    param_keys = weights_list[0].keys()

    for key in param_keys:
        weighted_sum = None
        for weights, count in zip(weights_list, sample_counts):
            weight_tensor = weights[key].float()
            factor = float(count) / float(total_samples)
            if weighted_sum is None:
                weighted_sum = factor * weight_tensor
            else:
                weighted_sum += factor * weight_tensor

        aggregated_weights[key] = weighted_sum

    return aggregated_weights


class FederatedServer:
    """
    Central Coordinator for Federated Learning.
    Coordinates weight broadcast, client model reception, FedAvg aggregation,
    and communication bandwidth monitoring.
    """
    def __init__(
        self,
        hidden_size: int = 32,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.global_model = TrafficGRU(input_size=1, hidden_size=hidden_size, num_layers=1, output_size=1).to(self.device)
        self.num_params = self.global_model.count_parameters()
        self.global_weights = get_weights_dict(self.global_model)

    def run_simulation(
        self,
        clients: List[TrafficClient],
        global_eval: dict,
        num_rounds: int = 10,
        local_epochs: int = 1,
        learning_rate: float = 0.005,
        batch_size: int = 32,
        precision: str = "FP32",
        lr_decay: float = 1.0,
        early_stopping: bool = False,
        round_callback: Optional[Callable[[dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes full Federated Learning workflow.
        Yields/updates live progress through round_callback.
        """
        num_clients = len(clients)
        history = []
        cumulative_comm_bytes = 0

        # Calculate exact per-round communication cost
        round_comm_meta = calculate_round_communication(
            num_parameters=self.num_params,
            num_clients=num_clients,
            precision=precision
        )

        current_lr = learning_rate
        best_mae = float("inf")
        patience = 5
        patience_counter = 0

        for round_idx in range(1, num_rounds + 1):
            client_updates = []
            sample_counts = []
            client_losses = {}
            client_status = {}

            # STEP 1: Broadcast & Local Training at each intersection node
            for client in clients:
                # If FP16 precision selected, simulate quantized downlink
                if precision.upper() == "FP16":
                    downlink_weights = quantize_to_fp16(self.global_weights)
                else:
                    downlink_weights = copy.deepcopy(self.global_weights)

                # Client performs local optimization on private data
                local_weights, n_samples, loss = client.train_locally(
                    global_weights=downlink_weights,
                    local_epochs=local_epochs,
                    lr=current_lr,
                    batch_size=batch_size
                )

                # If FP16 precision selected, simulate quantized uplink
                if precision.upper() == "FP16":
                    uplink_weights = quantize_to_fp16(local_weights)
                else:
                    uplink_weights = local_weights

                client_updates.append(uplink_weights)
                sample_counts.append(n_samples)
                client_losses[client.client_id] = round(loss, 4)
                client_status[client.client_id] = "Trained & Updated"

            # STEP 2: FedAvg Aggregation on Server
            self.global_weights = fedavg_aggregate(client_updates, sample_counts)
            set_weights_dict(self.global_model, self.global_weights)

            # STEP 3: Evaluation of Updated Global Model on Global Test Set
            self.global_model.eval()
            with torch.no_grad():
                gx = torch.tensor(global_eval["X_test"], dtype=torch.float32).to(self.device)
                g_preds_scaled = self.global_model(gx).cpu().numpy()

            g_preds_real = global_eval["scaler"].inverse_transform(g_preds_scaled).flatten()
            g_actual_real = global_eval["raw_y_test"].flatten()
            global_metrics = compute_metrics(g_actual_real, g_preds_real)

            # Evaluate on each individual client test set
            client_eval_metrics = {}
            for client in clients:
                c_met, _, _ = client.evaluate(self.global_weights)
                client_eval_metrics[client.client_id] = c_met

            # Accumulate communication
            cumulative_comm_bytes += round_comm_meta["total_round_bytes"]
            cumulative_comm_mb = cumulative_comm_bytes / (1024.0 * 1024.0)

            round_info = {
                "round": round_idx,
                "total_rounds": num_rounds,
                "mae": global_metrics["mae"],
                "rmse": global_metrics["rmse"],
                "mape": global_metrics["mape"],
                "r2": global_metrics["r2"],
                "client_losses": client_losses,
                "client_status": client_status,
                "client_metrics": client_eval_metrics,
                "round_comm_kb": round(round_comm_meta["total_round_kb"], 2),
                "total_comm_mb": round(cumulative_comm_mb, 4),
                "precision": precision.upper(),
                "learning_rate": current_lr
            }
            history.append(round_info)

            if round_callback:
                round_callback(round_info)

            # Learning rate decay
            current_lr *= lr_decay

            # Early stopping check
            if early_stopping:
                if global_metrics["mae"] < best_mae - 0.01:
                    best_mae = global_metrics["mae"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # Final predictions on each client for visualization
        client_predictions = {}
        for client in clients:
            c_met, preds, acts = client.evaluate(self.global_weights)
            client_predictions[client.client_id] = {
                "predictions": preds,
                "actual": acts,
                "metrics": c_met
            }

        total_comm_summary = calculate_total_communication(
            num_parameters=self.num_params,
            num_clients=num_clients,
            num_rounds=len(history),
            precision=precision
        )

        return {
            "model": self.global_model,
            "num_parameters": self.num_params,
            "rounds_completed": len(history),
            "final_mae": history[-1]["mae"],
            "final_rmse": history[-1]["rmse"],
            "final_mape": history[-1]["mape"],
            "final_r2": history[-1]["r2"],
            "history": history,
            "client_predictions": client_predictions,
            "communication_summary": total_comm_summary,
            "precision": precision.upper(),
        }
