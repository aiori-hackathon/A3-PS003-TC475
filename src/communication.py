"""
Communication Cost Calculator for PS-003.
Evaluates model update communication bandwidth in Federated Learning.
Calculates bidirectional payload (Upload + Download) for FP32 and FP16 precision.
"""

from typing import Dict, Any


def get_bytes_per_parameter(precision: str = "FP32") -> int:
    """Returns number of bytes required per model parameter."""
    if precision.upper() == "FP16":
        return 2
    return 4  # Default FP32


def calculate_round_communication(
    num_parameters: int,
    num_clients: int,
    precision: str = "FP32"
) -> Dict[str, Any]:
    """
    Computes exact communication overhead for a single federated training round.

    Formula:
    Communication per round = P * B * 2 * K
    where:
    - P: number of trainable model parameters
    - B: bytes per parameter (4 for FP32, 2 for FP16)
    - 2: factor for bidirectional transmission (1 upload + 1 download)
    - K: number of participating clients
    """
    bytes_per_param = get_bytes_per_parameter(precision)

    # Per-client breakdown
    upload_bytes_per_client = num_parameters * bytes_per_param
    download_bytes_per_client = num_parameters * bytes_per_param
    per_client_round_bytes = upload_bytes_per_client + download_bytes_per_client

    # Total round across all K clients
    total_upload_bytes = upload_bytes_per_client * num_clients
    total_download_bytes = download_bytes_per_client * num_clients
    total_round_bytes = per_client_round_bytes * num_clients

    return {
        "parameters": num_parameters,
        "precision": precision.upper(),
        "bytes_per_param": bytes_per_param,
        "num_clients": num_clients,
        "upload_bytes_per_client": upload_bytes_per_client,
        "download_bytes_per_client": download_bytes_per_client,
        "per_client_round_bytes": per_client_round_bytes,
        "total_upload_bytes": total_upload_bytes,
        "total_download_bytes": total_download_bytes,
        "total_round_bytes": total_round_bytes,
        "total_round_kb": total_round_bytes / 1024.0,
        "total_round_mb": total_round_bytes / (1024.0 * 1024.0),
    }


def calculate_total_communication(
    num_parameters: int,
    num_clients: int,
    num_rounds: int,
    precision: str = "FP32"
) -> Dict[str, Any]:
    """
    Computes cumulative communication cost across all federated rounds.
    """
    round_comm = calculate_round_communication(num_parameters, num_clients, precision)
    total_bytes = round_comm["total_round_bytes"] * num_rounds
    total_kb = total_bytes / 1024.0
    total_mb = total_bytes / (1024.0 * 1024.0)

    return {
        **round_comm,
        "num_rounds": num_rounds,
        "total_bytes": total_bytes,
        "total_kb": total_kb,
        "total_mb": total_mb,
    }


def compare_fp32_vs_fp16(
    num_parameters: int,
    num_clients: int,
    num_rounds: int
) -> Dict[str, Any]:
    """
    Compares baseline FP32 against optimized FP16 communication cost.
    """
    fp32_res = calculate_total_communication(num_parameters, num_clients, num_rounds, precision="FP32")
    fp16_res = calculate_total_communication(num_parameters, num_clients, num_rounds, precision="FP16")

    saved_bytes = fp32_res["total_bytes"] - fp16_res["total_bytes"]
    saving_percentage = (saved_bytes / fp32_res["total_bytes"]) * 100.0 if fp32_res["total_bytes"] > 0 else 0.0

    return {
        "parameters": num_parameters,
        "clients": num_clients,
        "rounds": num_rounds,
        "fp32_total_mb": round(fp32_res["total_mb"], 4),
        "fp16_total_mb": round(fp16_res["total_mb"], 4),
        "fp32_round_kb": round(fp32_res["total_round_kb"], 2),
        "fp16_round_kb": round(fp16_res["total_round_kb"], 2),
        "saved_mb": round(saved_bytes / (1024.0 * 1024.0), 4),
        "saving_percentage": round(saving_percentage, 1),
    }
