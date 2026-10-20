"""
End-to-End Validation Suite for PS-003.
Validates:
A. Centralized baseline
B. Federated IID
C. Federated NON-IID
D. FP32 communication
E. FP16 communication comparison
F. 3-seed experiment
G. Streamlit AppTest reactive UI flow (3 clients, NON-IID, 5 rounds, 1 epoch, FP32, seed 42)
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_loader import load_traffic_data
from src.preprocessing import partition_client_data
from src.model import TrafficGRU
from src.communication import calculate_round_communication, calculate_total_communication, compare_fp32_vs_fp16
from src.experiments import (
    run_federated_experiment,
    run_centralized_experiment,
    run_three_seeds_experiment,
    load_all_experiment_records
)

def run_suite():
    print("=" * 70)
    print("PS-003 COMPREHENSIVE END-TO-END VALIDATION SUITE")
    print("=" * 70)

    df, meta = load_traffic_data("data/traffic.csv")
    print(f"\n[1] Data Loaded: {len(df)} intervals, label: '{meta['label']}'")

    # A. Centralized Baseline
    print("\n--- [A] CENTRALIZED BASELINE ---")
    cent_res = run_centralized_experiment(
        df=df, num_clients=3, mode="NON-IID", epochs=5, learning_rate=0.005, batch_size=32, hidden_size=32
    )
    print(f"Centralized Baseline -> MAE: {cent_res['mae']:.2f} veh | RMSE: {cent_res['rmse']:.2f} veh | R2: {cent_res['r2']:.3f}")
    assert cent_res["mae"] > 0, "Centralized MAE must be positive"
    assert cent_res["rmse"] > 0, "Centralized RMSE must be positive"

    # B. Federated IID
    print("\n--- [B] FEDERATED IID (FedAvg, 3 clients, 5 rounds, FP32) ---")
    iid_res = run_federated_experiment(
        df=df, num_clients=3, mode="IID", precision="FP32", num_rounds=5, local_epochs=1, learning_rate=0.005, batch_size=32, hidden_size=32, seed=42
    )
    print(f"FedAvg IID -> Rounds: {iid_res['rounds_completed']} | MAE: {iid_res['final_mae']:.2f} veh | RMSE: {iid_res['final_rmse']:.2f} veh | Comm: {iid_res['communication_summary']['total_round_kb']:.2f} KB/round, {iid_res['communication_summary']['total_mb']:.4f} MB total")
    assert iid_res["rounds_completed"] == 5

    # C. Federated NON-IID (Primary Benchmark: 3 clients, 5 rounds, 1 epoch, FP32, seed 42)
    print("\n--- [C] FEDERATED NON-IID (FedAvg, 3 clients, 5 rounds, FP32, Seed 42) ---")
    noniid_res = run_federated_experiment(
        df=df, num_clients=3, mode="NON-IID", precision="FP32", num_rounds=5, local_epochs=1, learning_rate=0.005, batch_size=32, hidden_size=32, seed=42
    )
    print(f"FedAvg NON-IID -> Rounds: {noniid_res['rounds_completed']} | MAE: {noniid_res['final_mae']:.2f} veh | RMSE: {noniid_res['final_rmse']:.2f} veh | Comm: {noniid_res['communication_summary']['total_round_kb']:.2f} KB/round, {noniid_res['communication_summary']['total_mb']:.4f} MB total")
    assert noniid_res["rounds_completed"] == 5
    assert len(noniid_res["history"]) == 5
    for h in noniid_res["history"]:
        print(f"  Round {h['round']}: MAE={h['mae']:.2f} veh, RMSE={h['rmse']:.2f} veh, Comm/Round={h['round_comm_kb']} KB")

    # D & E. FP32 vs FP16 Communication Evaluation
    print("\n--- [D & E] FP32 VS FP16 COMMUNICATION COMPARISON ---")
    model = TrafficGRU(hidden_size=32)
    p_cnt = model.count_parameters()
    fp32_comm = calculate_round_communication(p_cnt, num_clients=3, precision="FP32")
    fp16_comm = calculate_round_communication(p_cnt, num_clients=3, precision="FP16")
    comp_comm = compare_fp32_vs_fp16(p_cnt, num_clients=3, num_rounds=5)

    print(f"Parameter Count: {p_cnt:,} weights")
    print(f"FP32 Round Comm: {fp32_comm['total_round_kb']:.2f} KB ({fp32_comm['total_round_bytes']} B)")
    print(f"FP16 Round Comm: {fp16_comm['total_round_kb']:.2f} KB ({fp16_comm['total_round_bytes']} B)")
    print(f"Total FP32 (5 rounds): {comp_comm['fp32_total_mb']:.4f} MB | Total FP16 (5 rounds): {comp_comm['fp16_total_mb']:.4f} MB")
    print(f"Bandwidth Reduction: {comp_comm['saving_percentage']}% (Saved: {comp_comm['saved_mb']:.4f} MB)")
    assert comp_comm["saving_percentage"] == 50.0, "FP16 must reduce communication by exactly 50%"

    # F. 3-Seed Robustness Evaluation
    print("\n--- [F] 3-SEED ROBUSTNESS EVALUATION (Seeds 42, 43, 44) ---")
    seeds_res = run_three_seeds_experiment(
        df=df, seeds=[42, 43, 44], num_clients=3, mode="NON-IID", precision="FP32", num_rounds=5, local_epochs=1, hidden_size=32
    )
    print(f"Seed 42: MAE={seeds_res['seeds'][0]['mae']:.2f} veh")
    print(f"Seed 43: MAE={seeds_res['seeds'][1]['mae']:.2f} veh")
    print(f"Seed 44: MAE={seeds_res['seeds'][2]['mae']:.2f} veh")
    print(f"Mean MAE +/- Std: {seeds_res['mean_mae']:.2f} +/- {seeds_res['std_mae']:.2f} veh")
    print(f"Mean RMSE +/- Std: {seeds_res['mean_rmse']:.2f} +/- {seeds_res['std_rmse']:.2f} veh")
    assert seeds_res["std_mae"] >= 0

    # G. Streamlit AppTest Dashboard Integration
    print("\n--- [G] STREAMLIT DASHBOARD REACTIVE FLOW TEST ---")
    at = AppTest.from_file("app.py", default_timeout=240)
    at.run(timeout=60)

    # Configure 3 clients, NON-IID, 5 rounds, 1 epoch, FP32, seed 42
    [s for s in at.sidebar.slider if "Number of Intersections" in s.label][0].set_value(3)
    [r for r in at.sidebar.radio if "Data Distribution Mode" in r.label][0].set_value("NON-IID")
    [s for s in at.sidebar.slider if s.label == "Federated Rounds"][0].set_value(5)
    [s for s in at.sidebar.slider if "Local Epochs" in s.label][0].set_value(1)
    [r for r in at.sidebar.radio if "Communication Precision" in r.label][0].set_value("FP32")
    [n for n in at.sidebar.number_input if "Random Seed" in n.label][0].set_value(42)

    # Click run button
    run_btn = [b for b in at.sidebar.button if "Run Federated Learning" in b.label][0]
    run_btn.click()
    at.run(timeout=180)

    assert len(at.exception) == 0, f"AppTest threw exceptions: {at.exception}"

    top_m = {m.label: m.value for m in at.metric[:5]}
    print("Dashboard Top KPI Metric Cards:")
    for k, v in top_m.items():
        print(f"  {k} = {v}")

    fl_last = at.session_state.last_fl_result
    assert fl_last is not None, "last_fl_result not in session_state"
    assert fl_last["rounds_completed"] == 5, f"Expected 5 rounds, got {fl_last['rounds_completed']}"

    assert top_m["Federated Rounds"] == "5", f"Expected '5', got '{top_m['Federated Rounds']}'"
    assert f"{fl_last['final_mae']:.2f}" in top_m["MAE"], f"MAE mismatch: {top_m['MAE']}"
    assert f"{fl_last['final_rmse']:.2f}" in top_m["RMSE"], f"RMSE mismatch: {top_m['RMSE']}"
    assert f"{fl_last['communication_summary']['total_round_kb']:.2f}" in top_m["Communication / Round"], f"Comm mismatch: {top_m['Communication / Round']}"

    print("\n" + "=" * 70)
    print("ALL VALIDATION SUITE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_suite()
