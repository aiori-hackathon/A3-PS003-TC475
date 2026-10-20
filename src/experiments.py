"""
Experiment Management and Multi-Seed Robustness Runner for PS-003.
Manages running, persisting, and comparing Centralized, FedAvg (IID & Non-IID),
and Optimized FL (FP16) experiments.
"""

from typing import Dict, List, Any, Optional, Callable
import os
import random
import pandas as pd
import numpy as np
import torch

from src.data_loader import load_traffic_data
from src.preprocessing import partition_client_data
from src.client import TrafficClient
from src.federated import FederatedServer
from src.centralized import train_centralized_model
from src.communication import calculate_total_communication


RESULTS_FILE = "results/experiments.csv"


def ensure_results_dir(path: str = RESULTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df = pd.DataFrame(columns=[
            "timestamp", "method", "data_setting", "precision", "clients",
            "rounds_epochs", "mae", "rmse", "mape", "r2", "comm_kb", "comm_mb", "seed"
        ])
        df.to_csv(path, index=False)


def save_experiment_record(record: dict, path: str = RESULTS_FILE):
    ensure_results_dir(path)
    df = pd.read_csv(path)
    new_row = pd.DataFrame([record])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)


def load_all_experiment_records(path: str = RESULTS_FILE) -> pd.DataFrame:
    ensure_results_dir(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def run_federated_experiment(
    df: pd.DataFrame,
    num_clients: int = 3,
    mode: str = "NON-IID",
    precision: str = "FP32",
    num_rounds: int = 5,
    local_epochs: int = 1,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    window_length: int = 12,
    train_ratio: float = 0.8,
    seed: int = 42,
    hidden_size: int = 32,
    device: str = "cpu",
    round_callback: Optional[Callable[[dict], None]] = None
) -> Dict[str, Any]:
    """
    Runs a single full Federated Learning experiment.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    client_dict, global_eval = partition_client_data(
        df=df,
        num_clients=num_clients,
        mode=mode,
        window_length=window_length,
        train_ratio=train_ratio,
        seed=seed
    )

    clients = []
    for c_id, c_data in client_dict.items():
        client = TrafficClient(
            client_id=c_id,
            name=c_data["name"],
            pattern_desc=c_data["pattern_desc"],
            X_train=c_data["X_train"],
            y_train=c_data["y_train"],
            X_test=c_data["X_test"],
            y_test=c_data["y_test"],
            raw_y_test=c_data["raw_y_test"],
            scaler=c_data["scaler"],
            hidden_size=hidden_size,
            device=device
        )
        clients.append(client)

    server = FederatedServer(hidden_size=hidden_size, device=device)
    fl_results = server.run_simulation(
        clients=clients,
        global_eval=global_eval,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        precision=precision,
        round_callback=round_callback
    )

    # Save to CSV
    comm_summary = fl_results["communication_summary"]
    record = {
        "timestamp": str(pd.Timestamp.now()),
        "method": "FedAvg",
        "data_setting": mode,
        "precision": precision,
        "clients": num_clients,
        "rounds_epochs": f"{num_rounds} rounds",
        "mae": fl_results["final_mae"],
        "rmse": fl_results["final_rmse"],
        "mape": fl_results["final_mape"],
        "r2": fl_results["final_r2"],
        "comm_kb": round(comm_summary["total_kb"], 2),
        "comm_mb": round(comm_summary["total_mb"], 4),
        "seed": seed
    }
    save_experiment_record(record)

    return {
        **fl_results,
        "client_dict": client_dict,
        "global_eval": global_eval,
        "record": record
    }


def run_centralized_experiment(
    df: pd.DataFrame,
    num_clients: int = 3,
    mode: str = "NON-IID",
    epochs: int = 5,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    window_length: int = 12,
    train_ratio: float = 0.8,
    seed: int = 42,
    hidden_size: int = 32,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Dict[str, Any]:
    """
    Runs a centralized baseline training run.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    client_dict, global_eval = partition_client_data(
        df=df,
        num_clients=num_clients,
        mode=mode,
        window_length=window_length,
        train_ratio=train_ratio,
        seed=seed
    )

    cent_results = train_centralized_model(
        client_dict=client_dict,
        global_eval=global_eval,
        epochs=epochs,
        lr=learning_rate,
        batch_size=batch_size,
        hidden_size=hidden_size,
        device=device,
        progress_callback=progress_callback
    )

    record = {
        "timestamp": str(pd.Timestamp.now()),
        "method": "Centralized",
        "data_setting": "Combined",
        "precision": "FP32",
        "clients": num_clients,
        "rounds_epochs": f"{epochs} epochs",
        "mae": cent_results["mae"],
        "rmse": cent_results["rmse"],
        "mape": cent_results["mape"],
        "r2": cent_results["r2"],
        "comm_kb": 0.0,
        "comm_mb": 0.0,
        "seed": seed
    }
    save_experiment_record(record)

    return {
        **cent_results,
        "client_dict": client_dict,
        "global_eval": global_eval,
        "record": record
    }


def run_three_seeds_experiment(
    df: pd.DataFrame,
    seeds: List[int] = [42, 43, 44],
    num_clients: int = 3,
    mode: str = "NON-IID",
    precision: str = "FP32",
    num_rounds: int = 5,
    local_epochs: int = 1,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    window_length: int = 12,
    train_ratio: float = 0.8,
    hidden_size: int = 32,
    device: str = "cpu",
    seed_callback: Optional[Callable[[int, int, dict], None]] = None
) -> Dict[str, Any]:
    """
    Executes Federated Learning across 3 distinct seeds to evaluate robustness.
    Calculates Mean and Standard Deviation for MAE, RMSE, and Communication.
    """
    seed_results = []
    maes = []
    rmses = []
    comm_mbs = []

    for i, s in enumerate(seeds):
        res = run_federated_experiment(
            df=df,
            num_clients=num_clients,
            mode=mode,
            precision=precision,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            window_length=window_length,
            train_ratio=train_ratio,
            seed=s,
            hidden_size=hidden_size,
            device=device,
            round_callback=None
        )

        maes.append(res["final_mae"])
        rmses.append(res["final_rmse"])
        comm_mbs.append(res["communication_summary"]["total_mb"])

        seed_results.append({
            "seed": s,
            "mae": res["final_mae"],
            "rmse": res["final_rmse"],
            "comm_mb": res["communication_summary"]["total_mb"],
        })

        if seed_callback:
            seed_callback(i + 1, len(seeds), seed_results[-1])

    mean_mae = float(np.mean(maes))
    std_mae = float(np.std(maes))
    mean_rmse = float(np.mean(rmses))
    std_rmse = float(np.std(rmses))
    mean_comm_mb = float(np.mean(comm_mbs))
    std_comm_mb = float(np.std(comm_mbs))

    return {
        "seeds": seed_results,
        "mean_mae": round(mean_mae, 3),
        "std_mae": round(std_mae, 3),
        "mean_rmse": round(mean_rmse, 3),
        "std_rmse": round(std_rmse, 3),
        "mean_comm_mb": round(mean_comm_mb, 4),
        "std_comm_mb": round(std_comm_mb, 4),
    }
