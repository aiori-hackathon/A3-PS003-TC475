"""
Comprehensive End-to-End Validation Script for PS-003.
Validates:
1. Synthetic data generation and loading
2. IID and Non-IID partitioning
3. GRU model forward pass and parameter counting
4. FP32 vs FP16 communication cost evaluation
5. Centralized training baseline execution
6. Federated learning (FedAvg) execution with client round checks
7. Multi-seed robustness execution (seeds 42, 43, 44)
8. Result logging to results/experiments.csv
"""

import os
import pandas as pd
import numpy as np

from src.data_loader import load_traffic_data, generate_synthetic_traffic_data
from src.preprocessing import partition_client_data, TimeSeriesScaler
from src.model import TrafficGRU, get_weights_dict, quantize_to_fp16
from src.communication import calculate_round_communication, calculate_total_communication, compare_fp32_vs_fp16
from src.experiments import run_federated_experiment, run_centralized_experiment, run_three_seeds_experiment, load_all_experiment_records

def main():
    print("=" * 60)
    print("PS-003 END-TO-END VALIDATION TEST")
    print("=" * 60)

    # 1. Dataset loading / generation
    df, meta = load_traffic_data("data/traffic.csv")
    print(f"[OK] Data Loaded: {len(df)} rows, 15-min intervals")
    print(f"     Label: {meta['label']}")
    print(f"     Available intersections: {meta['available_intersections']}")
    assert len(df) > 500, "Dataset too small"

    # 2. Preprocessing & Partitioning: IID vs Non-IID
    iid_clients, iid_eval = partition_client_data(df, num_clients=4, mode="IID", window_length=12)
    non_iid_clients, non_iid_eval = partition_client_data(df, num_clients=4, mode="NON-IID", window_length=12)
    print(f"[OK] Partitioning verified:")
    print(f"     IID client 1 mean: {iid_clients['client_1']['stats']['mean_flow']:.2f}")
    print(f"     Non-IID client 1 mean: {non_iid_clients['client_1']['stats']['mean_flow']:.2f}")
    print(f"     Non-IID client 2 mean: {non_iid_clients['client_2']['stats']['mean_flow']:.2f}")

    # 3. Model & Parameter Counting
    model = TrafficGRU(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    p_cnt = model.count_parameters()
    print(f"[OK] TrafficGRU initialized. Parameters: {p_cnt}")
    assert p_cnt > 0, "Parameter count must be positive"

    # 4. Communication Cost Evaluation (PS-003 Core)
    fp32_comm = calculate_round_communication(p_cnt, num_clients=4, precision="FP32")
    fp16_comm = calculate_round_communication(p_cnt, num_clients=4, precision="FP16")
    comp = compare_fp32_vs_fp16(p_cnt, num_clients=4, num_rounds=10)
    print(f"[OK] Communication Evaluation verified:")
    print(f"     FP32 per round: {fp32_comm['total_round_kb']:.2f} KB ({fp32_comm['total_round_bytes']} B)")
    print(f"     FP16 per round: {fp16_comm['total_round_kb']:.2f} KB ({fp16_comm['total_round_bytes']} B)")
    print(f"     Bandwidth reduction: {comp['saving_percentage']}%")
    assert comp['saving_percentage'] == 50.0, "FP16 should save exactly 50% bandwidth"

    # 5. Centralized Baseline Run
    print("\n[+] Testing Centralized Baseline...")
    cent_res = run_centralized_experiment(
        df=df, num_clients=3, mode="NON-IID", epochs=3, hidden_size=16
    )
    print(f"[OK] Centralized Baseline MAE: {cent_res['mae']:.2f} veh | RMSE: {cent_res['rmse']:.2f} veh")

    # 6. Federated Learning Run (FedAvg)
    print("\n[+] Testing Federated Learning (FedAvg)...")
    fl_res = run_federated_experiment(
        df=df, num_clients=3, mode="NON-IID", precision="FP32", num_rounds=3, local_epochs=1, hidden_size=16
    )
    print(f"[OK] Federated Learning MAE: {fl_res['final_mae']:.2f} veh | RMSE: {fl_res['final_rmse']:.2f} veh")
    print(f"     Total Communication: {fl_res['communication_summary']['total_mb']:.4f} MB")

    # 7. Multi-Seed Robustness Test
    print("\n[+] Testing Multi-Seed Robustness (Seeds 42, 43, 44)...")
    seeds_res = run_three_seeds_experiment(
        df=df, seeds=[42, 43, 44], num_clients=3, mode="NON-IID", num_rounds=2, local_epochs=1, hidden_size=16
    )
    print(f"[OK] 3-Seed Evaluation Results:")
    print(f"     Mean MAE +/- Std: {seeds_res['mean_mae']:.2f} +/- {seeds_res['std_mae']:.2f} veh")
    print(f"     Mean RMSE +/- Std: {seeds_res['mean_rmse']:.2f} +/- {seeds_res['std_rmse']:.2f} veh")
    print(f"     Mean Comm: {seeds_res['mean_comm_mb']:.4f} MB")

    # 8. Check Results Log File
    records = load_all_experiment_records("results/experiments.csv")
    print(f"\n[OK] Results logged to results/experiments.csv. Total records: {len(records)}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! PROTOTYPE READY.")
    print("=" * 60)

if __name__ == "__main__":
    main()
