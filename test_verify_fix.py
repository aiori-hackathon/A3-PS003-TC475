import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
from streamlit.testing.v1 import AppTest

def test_full_flow():
    print("Initializing AppTest for app.py...")
    at = AppTest.from_file("app.py", default_timeout=240)
    at.run(timeout=60)

    # Verify initial state
    initial_rounds = [m.value for m in at.metric if m.label == "Federated Rounds"]
    print("Initial Federated Rounds metric:", initial_rounds)
    assert initial_rounds[0] in ["Not Run", "5"], f"Initial state should be Not Run or 5, got {initial_rounds[0]}"

    # Configure:
    # 3 clients, NON-IID, 5 federated rounds, 1 local epoch, FP32, seed 42
    print("Configuring experiment parameters: 3 clients, NON-IID, 5 rounds, 1 epoch, FP32, seed 42...")
    clients_slider = [s for s in at.sidebar.slider if "Number of Intersections" in s.label][0]
    clients_slider.set_value(3)

    mode_radio = [r for r in at.sidebar.radio if "Data Distribution Mode" in r.label][0]
    mode_radio.set_value("NON-IID")

    rounds_slider = [s for s in at.sidebar.slider if s.label == "Federated Rounds"][0]
    rounds_slider.set_value(5)

    epochs_slider = [s for s in at.sidebar.slider if "Local Epochs" in s.label][0]
    epochs_slider.set_value(1)

    prec_radio = [r for r in at.sidebar.radio if "Communication Precision" in r.label][0]
    prec_radio.set_value("FP32")

    seed_input = [n for n in at.sidebar.number_input if "Random Seed" in n.label][0]
    seed_input.set_value(42)

    # 1. Click 'Run Federated Learning'
    print("Triggering '🚀 Run Federated Learning'...")
    run_btn = [b for b in at.sidebar.button if "Run Federated Learning" in b.label][0]
    run_btn.click()
    at.run(timeout=180)

    assert len(at.exception) == 0, f"Exceptions encountered: {at.exception}"

    fl_res = at.session_state.last_fl_result
    assert fl_res is not None, "session_state.last_fl_result must not be None"
    assert fl_res["rounds_completed"] == 5, f"Expected 5 rounds, got {fl_res['rounds_completed']}"
    assert len(fl_res["history"]) == 5, f"Expected 5 history entries, got {len(fl_res['history'])}"

    print("\n--- ROUND HISTORY ---")
    for h in fl_res["history"]:
        print(f"Round {h['round']}: MAE={h['mae']:.2f} veh, RMSE={h['rmse']:.2f} veh, Comm={h['round_comm_kb']} KB")

    # Verify Top KPI Metric Cards
    top_metrics = {m.label: m.value for m in at.metric[:5]}
    print("\n--- TOP DASHBOARD METRIC CARDS ---")
    for k, v in top_metrics.items():
        print(f"{k} = {v}")

    expected_mae_str = f"{fl_res['final_mae']:.2f} veh"
    expected_rmse_str = f"{fl_res['final_rmse']:.2f} veh"
    expected_comm_str = f"{fl_res['communication_summary']['total_round_kb']:.2f} KB"

    assert top_metrics["Federated Rounds"] == str(fl_res["rounds_completed"]), f"Expected '{fl_res['rounds_completed']}', got '{top_metrics['Federated Rounds']}'"
    assert top_metrics["MAE"] == expected_mae_str, f"Expected '{expected_mae_str}', got '{top_metrics['MAE']}'"
    assert top_metrics["RMSE"] == expected_rmse_str, f"Expected '{expected_rmse_str}', got '{top_metrics['RMSE']}'"
    assert top_metrics["Communication / Round"] == expected_comm_str, f"Expected '{expected_comm_str}', got '{top_metrics['Communication / Round']}'"

    # Verify Benchmark result storage
    bench = at.session_state.benchmark_results
    assert bench["FedAvg_NonIID"] is not None, "Benchmark FedAvg_NonIID should be populated"
    print(f"\n[OK] Benchmark Non-IID MAE: {bench['FedAvg_NonIID']['final_mae']:.2f} veh")

    # 2. Test Streamlit rerun/session-state persistence by changing another control
    print("\nTesting rerun resilience: changing Sequence Window slider to trigger rerun...")
    slider_w = [s for s in at.sidebar.slider if "Sliding Sequence Window" in s.label][0]
    slider_w.set_value(14)
    at.run(timeout=60)

    top_after_rerun = {m.label: m.value for m in at.metric[:5]}
    print("--- TOP METRICS AFTER RERUN ---")
    for k, v in top_after_rerun.items():
        print(f"{k} = {v}")

    assert top_after_rerun["Federated Rounds"] == str(fl_res["rounds_completed"]), "Federated Rounds lost on rerun!"
    assert top_after_rerun["MAE"] == expected_mae_str, "MAE lost on rerun!"
    assert top_after_rerun["RMSE"] == expected_rmse_str, "RMSE lost on rerun!"
    assert top_after_rerun["Communication / Round"] == expected_comm_str, "Communication lost on rerun!"

    # 3. Test Run Centralized Baseline
    print("\nTesting '🏢 Run Centralized Baseline'...")
    cent_btn = [b for b in at.sidebar.button if "Run Centralized Baseline" in b.label][0]
    cent_btn.click()
    at.run(timeout=180)

    cent_res = at.session_state.last_cent_result
    assert cent_res is not None, "Centralized baseline result must not be None"
    print(f"[OK] Centralized Baseline MAE: {cent_res['mae']:.2f} veh, RMSE: {cent_res['rmse']:.2f} veh")
    assert bench["Centralized"] is not None, "Benchmark Centralized should be populated"

    # 4. Test Run 3 Seeds
    print("\nTesting '🎲 Run 3 Seeds (Robustness)'...")
    seed_btn = [b for b in at.sidebar.button if "Run 3 Seeds" in b.label][0]
    seed_btn.click()
    at.run(timeout=180)

    seeds_res = at.session_state.three_seed_results
    assert seeds_res is not None, "Three seed results must not be None"
    print(f"[OK] 3 Seeds Mean MAE: {seeds_res['mean_mae']:.2f} +/- {seeds_res['std_mae']:.2f} veh")

    print("\n" + "=" * 60)
    print("ALL TESTS AND VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_flow()
