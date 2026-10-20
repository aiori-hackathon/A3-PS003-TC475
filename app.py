"""
Streamlit Web Dashboard for PS-003:
Federated Traffic-Flow Prediction with Communication-Cost Evaluation
Domain: Smart Cities | AIORI Hackathon 3
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.data_loader import load_traffic_data, generate_synthetic_traffic_data, DATASET_LABEL
from src.preprocessing import partition_client_data
from src.model import TrafficGRU
from src.communication import calculate_round_communication, calculate_total_communication, compare_fp32_vs_fp16
from src.federated import FederatedServer
from src.client import TrafficClient
from src.centralized import train_centralized_model
from src.experiments import (
    run_federated_experiment,
    run_centralized_experiment,
    run_three_seeds_experiment,
    load_all_experiment_records,
    RESULTS_FILE
)

# Page configuration
st.set_page_config(
    page_title="Federated Traffic Prediction | PS-003",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.1rem;
        font-weight: 500;
    }
    .data-locality-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }
    .transparency-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 15px;
        color: #166534;
    }
    .arch-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .live-status-card {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        font-family: monospace;
        margin-bottom: 14px;
    }
    .stMetric {
        background: #F9FAFB;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Device setup (auto-fallback to CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Session state initialization
if "traffic_df" not in st.session_state:
    df, meta = load_traffic_data("data/traffic.csv")
    st.session_state.traffic_df = df
    st.session_state.data_meta = meta

FL_CACHE_FILE = "results/latest_fl_result.pt"
CENT_CACHE_FILE = "results/latest_cent_result.pt"
ROBUST_CACHE_FILE = "results/latest_robust_result.pt"
BENCH_CACHE_FILE = "results/latest_bench_results.pt"

if "last_fl_result" not in st.session_state:
    if os.path.exists(FL_CACHE_FILE):
        try:
            st.session_state.last_fl_result = torch.load(FL_CACHE_FILE, weights_only=False)
        except Exception:
            st.session_state.last_fl_result = None
    else:
        st.session_state.last_fl_result = None

if "last_cent_result" not in st.session_state:
    if os.path.exists(CENT_CACHE_FILE):
        try:
            st.session_state.last_cent_result = torch.load(CENT_CACHE_FILE, weights_only=False)
        except Exception:
            st.session_state.last_cent_result = None
    else:
        st.session_state.last_cent_result = None

if "three_seed_results" not in st.session_state:
    if os.path.exists(ROBUST_CACHE_FILE):
        try:
            st.session_state.three_seed_results = torch.load(ROBUST_CACHE_FILE, weights_only=False)
        except Exception:
            st.session_state.three_seed_results = None
    else:
        st.session_state.three_seed_results = None

if "benchmark_results" not in st.session_state:
    if os.path.exists(BENCH_CACHE_FILE):
        try:
            st.session_state.benchmark_results = torch.load(BENCH_CACHE_FILE, weights_only=False)
        except Exception:
            st.session_state.benchmark_results = {
                "Centralized": None,
                "FedAvg_IID": None,
                "FedAvg_NonIID": None,
                "Optimized_FL": None
            }
    else:
        st.session_state.benchmark_results = {
            "Centralized": None,
            "FedAvg_IID": None,
            "FedAvg_NonIID": None,
            "Optimized_FL": None
        }

# Header Section
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown('<div class="main-header">Federated Traffic-Flow Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AIORI Hackathon 3 — PS-003 | Smart Cities: Federated Learning with Communication-Cost Evaluation</div>', unsafe_allow_html=True)
with col_head2:
    st.caption(f"**Hardware Device:** `{device.upper()}`")
    st.caption(f"**PyTorch:** `{torch.__version__}`")

# 1. PRIVACY WORDING (Strict Compliance with Point 1)
st.markdown("""
<div class="data-locality-box">
    <strong>🔒 Data-Locality & Scope Notice:</strong><br>
    Data-locality approach: Raw traffic data remains at simulated clients during federated training; only model updates/weights are exchanged.<br>
    <em>This prototype does not implement formal differential privacy, encryption-based secure aggregation, or other formal privacy guarantees.</em>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
st.sidebar.title("🎛️ Simulation Controls")

st.sidebar.subheader("Quick Hackathon Action")
demo_col1, demo_col2 = st.sidebar.columns(2)
start_demo_clicked = demo_col1.button("⚡ START DEMO", use_container_width=True, help="Fast 1-click execution for live judge demonstration")
reset_data_clicked = demo_col2.button("🔄 Reset Data", use_container_width=True, help="Regenerate synthetic traffic dataset")

if reset_data_clicked:
    df = generate_synthetic_traffic_data(num_days=28, freq_minutes=15, output_path="data/traffic.csv")
    st.session_state.traffic_df = df
    st.session_state.data_meta = {
        "is_synthetic": True,
        "label": DATASET_LABEL,
        "total_rows": len(df),
        "available_intersections": [c for c in df.columns if c.startswith("intersection_")],
        "start_time": str(df["timestamp"].iloc[0]),
        "end_time": str(df["timestamp"].iloc[-1]),
    }
    st.session_state.last_fl_result = None
    st.session_state.last_cent_result = None
    st.session_state.three_seed_results = None
    st.session_state.benchmark_results = {
        "Centralized": None,
        "FedAvg_IID": None,
        "FedAvg_NonIID": None,
        "Optimized_FL": None
    }
    for cf in [FL_CACHE_FILE, CENT_CACHE_FILE, ROBUST_CACHE_FILE, BENCH_CACHE_FILE]:
        if os.path.exists(cf):
            try:
                os.remove(cf)
            except Exception:
                pass
    st.sidebar.success("Dataset regenerated & state reset!")

st.sidebar.markdown("---")
st.sidebar.subheader("Client & Data Configuration")

num_clients = st.sidebar.slider("Number of Intersections / Clients", min_value=3, max_value=5, value=3, step=1)
dataset_mode = st.sidebar.radio("Data Distribution Mode", ["NON-IID", "IID"], help="NON-IID: Spatial intersection variance | IID: Homogeneous uniform mixing")
window_length = st.sidebar.slider("Sliding Sequence Window (W)", min_value=6, max_value=24, value=12, help="Past time intervals used to predict next measurement (e.g. 12 = 3 hours)")
train_ratio = st.sidebar.slider("Train / Test Split Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05, help="Chronological partition (strictly avoids lookahead bias)")

st.sidebar.markdown("---")
st.sidebar.subheader("Federated Learning Parameters")

federated_rounds = st.sidebar.slider("Federated Rounds", min_value=2, max_value=20, value=5, step=1)
local_epochs = st.sidebar.slider("Local Epochs per Round (E)", min_value=1, max_value=5, value=1, step=1)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.003, 0.005, 0.01, 0.02], value=0.005)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
precision_mode = st.sidebar.radio("Communication Precision", ["FP32", "FP16"], help="FP32 = 4 bytes/parameter | FP16 = 2 bytes/parameter (50% bandwidth reduction)")
random_seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
hidden_units = st.sidebar.select_slider("GRU Hidden Units", options=[16, 32, 64], value=32)

st.sidebar.markdown("---")
run_experiment_clicked = st.sidebar.button("🚀 Run Federated Learning", type="primary", use_container_width=True)
run_centralized_clicked = st.sidebar.button("🏢 Run Centralized Baseline", use_container_width=True)
run_three_seeds_clicked = st.sidebar.button("🎲 Run 3 Seeds (Robustness)", use_container_width=True)
sidebar_status_box = st.sidebar.empty()

# Top KPI Metric Cards (Reactive Placeholder)
kpi_placeholder = st.empty()

def render_top_kpi_cards():
    with kpi_placeholder.container():
        latest_fl = st.session_state.last_fl_result
        latest_cent = st.session_state.last_cent_result

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Active Clients", f"{num_clients} Intersections", f"Mode: {dataset_mode}")
        with mcol2:
            rounds_disp = f"{latest_fl['rounds_completed']}" if latest_fl else "Not Run"
            st.metric("Federated Rounds", rounds_disp, f"E={local_epochs} local")
        with mcol3:
            mae_val = f"{latest_fl['final_mae']:.2f} veh" if latest_fl else "—"
            delta_mae = f"vs Cent: {latest_fl['final_mae'] - latest_cent['mae']:+.2f}" if (latest_fl and latest_cent) else None
            st.metric("MAE", mae_val, delta_mae, delta_color="inverse")
        with mcol4:
            rmse_val = f"{latest_fl['final_rmse']:.2f} veh" if latest_fl else "—"
            st.metric("RMSE", rmse_val)
        with mcol5:
            if latest_fl:
                c_round_kb = latest_fl["communication_summary"]["total_round_kb"]
                c_tot_mb = latest_fl["communication_summary"]["total_mb"]
                st.metric("Communication / Round", f"{c_round_kb:.2f} KB", f"Total: {c_tot_mb:.3f} MB ({precision_mode})")
            else:
                dummy_model = TrafficGRU(hidden_size=hidden_units)
                p_count = dummy_model.count_parameters()
                theo_round = calculate_round_communication(p_count, num_clients, precision_mode)
                st.metric("Communication / Round", f"{theo_round['total_round_kb']:.2f} KB", f"Params: {p_count} ({precision_mode})")

render_top_kpi_cards()

# Tabs for Navigation
tab_overview, tab_data, tab_simulation, tab_predictions, tab_communication, tab_benchmark, tab_robustness, tab_logs = st.tabs([
    "🏛️ Architecture & Transparency",
    "📊 Data & IID vs Non-IID",
    "⚡ Training & Live Status",
    "📈 Traffic Predictions",
    "📡 Communication Cost",
    "⚖️ Performance Benchmark",
    "🎲 Multi-Seed Robustness",
    "📋 Experiment History"
])

# ==========================================
# TAB 1: ARCHITECTURE & TRANSPARENCY
# ==========================================
with tab_overview:
    st.subheader("System Architecture & Dataset Transparency")

    # Point 2: Clear dataset transparency card
    st.markdown(f"""
    <div class="transparency-card">
        <strong>📋 Dataset Source Transparency:</strong><br>
        • <strong>Dataset Label:</strong> <code>{DATASET_LABEL}</code><br>
        • <strong>Target Variable:</strong> Traffic Flow (Vehicles / 15-minute interval)<br>
        • <strong>Target Type:</strong> Vehicle volume count across simulated intersection sensor channels (not speed, not occupancy)<br>
        • <strong>Number of Observations:</strong> <code>{len(st.session_state.traffic_df):,}</code> intervals (28 days × 96 intervals/day)<br>
        • <strong>Train / Test Partition:</strong> Chronological split (80% Train: ~2,150 observations / 20% Test: ~538 observations; no random lookahead shuffle)<br>
        • <strong>Simulated Sensor Locations:</strong> 5 urban intersection profiles (Client 1: Normal bimodal, Client 2: Morning peak heavy, Client 3: Evening peak heavy, Client 4: Arterial highway, Client 5: Residential feeder)
    </div>
    """, unsafe_allow_html=True)

    col_arch_diag, col_arch_spec = st.columns([3, 2])
    with col_arch_diag:
        st.markdown("""
        ```
        ========================================================================================
                                     FEDERATED LEARNING ARCHITECTURE
        ========================================================================================

                             [ Global GRU Model W_t ] 
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │ Broadcast Model W_t      │ Broadcast Model W_t      │ Broadcast Model W_t
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ Intersection 1    │      │ Intersection 2    │      │ Intersection K    │
   │ (Simulated Client)│      │ (Simulated Client)│      │ (Simulated Client)│
   │ ───────────────── │      │ ───────────────── │      │ ───────────────── │
   │ 🔒 Local Sensor   │      │ 🔒 Local Sensor   │      │ 🔒 Local Sensor   │
   │    Data Stays Here│      │    Data Stays Here│      │    Data Stays Here│
   │ Local GRU Training│      │ Local GRU Training│      │ Local GRU Training│
   └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        │ Upload Trained Weights {W_k}
                                        ▼
                            ┌───────────────────────┐
                            │     FedAvg Server     │
                            │  Weighted Aggregation │
                            │  W_{t+1} = Σ (n_i/N)W_i│
                            └───────────┬───────────┘
                                        │
                                        ▼
                            [ New Global Model W_{t+1} ]
        ========================================================================================
        ```
        """)
        st.info("💡 **Data-locality approach:** Raw traffic data remains at simulated clients during federated training; only model updates/weights are exchanged.")

    with col_arch_spec:
        st.markdown('<div class="arch-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Technical Specifications")
        model_sample = TrafficGRU(input_size=1, hidden_size=hidden_units, num_layers=1, output_size=1)
        param_cnt = model_sample.count_parameters()
        comm_fp32 = calculate_round_communication(param_cnt, num_clients, "FP32")
        comm_fp16 = calculate_round_communication(param_cnt, num_clients, "FP16")

        st.markdown(f"""
        - **Model Architecture:** Compact PyTorch GRU (1 layer + Linear head)
        - **Sequence Window (W):** `{window_length}` intervals
        - **Hidden Size:** `{hidden_units}` units
        - **Total Parameters (P):** `{param_cnt:,}` weights
        - **Parameter Precision:** `{precision_mode}` ({4 if precision_mode=='FP32' else 2} bytes/parameter)
        - **Active Clients (K):** `{num_clients}` intersections
        - **Bidirectional Factor:** `2` (1 Upload + 1 Download per client)
        - **Communication per Round:**
          - **FP32:** `{comm_fp32['total_round_kb']:.2f} KB` (`{comm_fp32['total_round_bytes']:,} bytes`)
          - **FP16:** `{comm_fp16['total_round_kb']:.2f} KB` (`{comm_fp16['total_round_bytes']:,} bytes`)
          - **Savings with FP16:** **50.0% Bandwidth Reduction**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: DATA & IID VS NON-IID (Point 7)
# ==========================================
with tab_data:
    st.subheader("Data Overview & IID vs Non-IID Distribution Analysis")
    df = st.session_state.traffic_df
    meta = st.session_state.data_meta

    st.markdown(f"**Dataset Label:** `{meta['label']}` | **Total Records:** `{meta['total_rows']:,}` intervals | **Interval Duration:** 15 minutes")

    # Time series snippet
    st.markdown("##### 🚗 Traffic Flow Time Series (Selected Window)")
    plot_points = st.slider("Visualization Window (time intervals)", min_value=96, max_value=min(960, len(df)), value=288, step=48)
    sub_df = df.iloc[:plot_points]

    fig_ts, ax_ts = plt.subplots(figsize=(12, 3.8))
    for i in range(1, num_clients + 1):
        col_name = f"intersection_{i}"
        if col_name in sub_df.columns:
            ax_ts.plot(sub_df.index, sub_df[col_name], label=f"Client {i} ({col_name})", alpha=0.85, linewidth=1.5)
    ax_ts.set_xlabel("Time Interval (15-min index)")
    ax_ts.set_ylabel("Traffic Flow (Vehicles / 15-min)")
    ax_ts.set_title(f"Traffic Sensor Streams Across Selected Intersections ({DATASET_LABEL})")
    ax_ts.grid(True, linestyle="--", alpha=0.5)
    ax_ts.legend(loc="upper right")
    fig_ts.tight_layout()
    st.pyplot(fig_ts)

    st.markdown("---")
    # Point 7: Side-by-side IID vs Non-IID visualization
    st.markdown("##### ⚖️ IID vs Non-IID Client Distribution Comparison")
    st.caption("Demonstrating how Non-IID creates distinct client traffic distributions while IID creates homogeneous distributions.")

    iid_client_dict, _ = partition_client_data(df, num_clients=num_clients, mode="IID", window_length=window_length, train_ratio=train_ratio, seed=random_seed)
    non_iid_client_dict, _ = partition_client_data(df, num_clients=num_clients, mode="NON-IID", window_length=window_length, train_ratio=train_ratio, seed=random_seed)

    col_dist_chart1, col_dist_chart2 = st.columns(2)
    with col_dist_chart1:
        fig_iid, ax_iid = plt.subplots(figsize=(5.5, 3.6))
        for cid, cinfo in iid_client_dict.items():
            ax_iid.hist(cinfo["raw_y_test"].flatten(), bins=18, alpha=0.45, label=cid)
        ax_iid.set_title("Mode 1: IID Distribution (Homogeneous)")
        ax_iid.set_xlabel("Vehicles / 15-min")
        ax_iid.set_ylabel("Frequency")
        ax_iid.legend()
        ax_iid.grid(True, linestyle=":", alpha=0.5)
        fig_iid.tight_layout()
        st.pyplot(fig_iid)

    with col_dist_chart2:
        fig_noniid, ax_noniid = plt.subplots(figsize=(5.5, 3.6))
        for cid, cinfo in non_iid_client_dict.items():
            ax_noniid.hist(cinfo["raw_y_test"].flatten(), bins=18, alpha=0.45, label=cid)
        ax_noniid.set_title("Mode 2: Non-IID Distribution (Heterogeneous)")
        ax_noniid.set_xlabel("Vehicles / 15-min")
        ax_noniid.set_ylabel("Frequency")
        ax_noniid.legend()
        ax_noniid.grid(True, linestyle=":", alpha=0.5)
        fig_noniid.tight_layout()
        st.pyplot(fig_noniid)

    # Statistical comparison table
    st.markdown("##### 📊 Statistical Comparison Table (IID vs Non-IID)")
    comp_stats_rows = []
    for i in range(1, num_clients + 1):
        cid = f"client_{i}"
        iid_st = iid_client_dict[cid]["stats"]
        non_iid_st = non_iid_client_dict[cid]["stats"]
        comp_stats_rows.append({
            "Client": f"Client {i}",
            "Intersection Profile": non_iid_client_dict[cid]["pattern_desc"][:40] + "...",
            "IID Mean Flow": f"{iid_st['mean_flow']:.1f}",
            "IID Std Dev": f"{iid_st['std_flow']:.1f}",
            "Non-IID Mean Flow": f"{non_iid_st['mean_flow']:.1f}",
            "Non-IID Std Dev": f"{non_iid_st['std_flow']:.1f}",
            "Non-IID Peak Flow": f"{non_iid_st['max_flow']:.1f}",
        })
    st.dataframe(pd.DataFrame(comp_stats_rows), use_container_width=True)

# ==========================================
# TRAINING EXECUTION (Point 8, Point 10)
# ==========================================
training_placeholder = tab_simulation.empty()

def execute_fl_simulation(num_r, local_e, lr, b_size, clients_cnt, mode_val, prec_val, seed_val, h_size, is_demo=False):
    run_label = '⚡ Demo Mode' if is_demo else 'Federated Learning'
    sidebar_status_box.info(f"⏳ Running {run_label} ({clients_cnt} clients, {num_r} rounds)...")
    with st.spinner(f"Running {run_label} Simulation..."):
        with training_placeholder.container():
            st.markdown(f"### 🚀 Running {run_label}")
            
            try:
                progress_bar = st.progress(0)
                live_status_box = st.empty()
                live_metrics_box = st.empty()

                # Point 8: Exact Live Training formatting
                def round_callback(info):
                    current_r = info["round"]
                    total_r = info["total_rounds"]
                    progress_bar.progress(current_r / total_r)

                    # Build exact checklist text required by Point 8:
                    client_checks = "\n".join([f"Client {idx+1} — Local Training ✓" for idx in range(clients_cnt)])
                    status_text = f"""
```
Round {current_r}/{total_r}

{client_checks}

Model Updates Received ✓
FedAvg Aggregation ✓
Global Model Updated ✓
```
"""
                    live_status_box.markdown(status_text)
                    
                    # Show live metrics required by Point 8 (in-place container)
                    with live_metrics_box.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAE", f"{info['mae']:.2f} veh")
                        m2.metric("RMSE", f"{info['rmse']:.2f} veh")
                        m3.metric("Communication / Round", f"{info['round_comm_kb']:.2f} KB", f"Total: {info['total_comm_mb']:.3f} MB")

                    if is_demo:
                        time.sleep(0.35)

                res = run_federated_experiment(
                    df=st.session_state.traffic_df,
                    num_clients=clients_cnt,
                    mode=mode_val,
                    precision=prec_val,
                    num_rounds=num_r,
                    local_epochs=local_e,
                    learning_rate=lr,
                    batch_size=b_size,
                    window_length=window_length,
                    train_ratio=train_ratio,
                    seed=seed_val,
                    hidden_size=h_size,
                    device=device,
                    round_callback=round_callback
                )

                st.session_state.last_fl_result = res

                # Update benchmark result tracker
                if prec_val == "FP16":
                    st.session_state.benchmark_results["Optimized_FL"] = res
                elif mode_val == "IID":
                    st.session_state.benchmark_results["FedAvg_IID"] = res
                else:
                    st.session_state.benchmark_results["FedAvg_NonIID"] = res

                os.makedirs("results", exist_ok=True)
                try:
                    torch.save(res, FL_CACHE_FILE)
                    torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                except Exception:
                    pass

                # Update Top KPI Cards immediately
                render_top_kpi_cards()

                success_msg = f"{run_label} completed! Final MAE: {res['final_mae']:.2f} veh | RMSE: {res['final_rmse']:.2f} veh | Total Comm: {res['communication_summary']['total_mb']:.3f} MB"
                st.success(success_msg)
                sidebar_status_box.success(f"✓ {run_label} complete!")
            except Exception as e:
                import traceback
                st.error(f"❌ Federated Learning Failed: {str(e)}")
                sidebar_status_box.error(f"❌ FL Failed: {str(e)}")
                st.code(traceback.format_exc())

# Trigger Demo Mode (Point 10: 4 clients, 4 rounds, 1 epoch, small GRU)
if start_demo_clicked:
    execute_fl_simulation(
        num_r=4,
        local_e=1,
        lr=0.01,
        b_size=32,
        clients_cnt=4,
        mode_val="NON-IID",
        prec_val="FP16",
        seed_val=42,
        h_size=16,
        is_demo=True
    )

# Trigger Full FL Experiment
if run_experiment_clicked:
    execute_fl_simulation(
        num_r=federated_rounds,
        local_e=local_epochs,
        lr=learning_rate,
        b_size=batch_size,
        clients_cnt=num_clients,
        mode_val=dataset_mode,
        prec_val=precision_mode,
        seed_val=random_seed,
        h_size=hidden_units,
        is_demo=False
    )

# Trigger Centralized Baseline
if run_centralized_clicked:
    sidebar_status_box.info(f"⏳ Training Centralized Baseline ({num_clients} clients pooled)...")
    with st.spinner("Training Centralized Baseline on pooled client dataset..."):
        with training_placeholder.container():
            st.markdown("### 🏢 Training Centralized Baseline GRU")
            st.info("Training centralized benchmark model on combined client data...")

            try:
                cent_bar = st.progress(0)
                c_status = st.empty()

                def cent_callback(epoch, total_e, mae, rmse):
                    cent_bar.progress(epoch / total_e)
                    c_status.markdown(f"**Epoch {epoch}/{total_e}:** Global Test MAE: `{mae:.2f}` | RMSE: `{rmse:.2f}`")

                c_res = run_centralized_experiment(
                    df=st.session_state.traffic_df,
                    num_clients=num_clients,
                    mode=dataset_mode,
                    epochs=federated_rounds * local_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    window_length=window_length,
                    train_ratio=train_ratio,
                    seed=random_seed,
                    hidden_size=hidden_units,
                    device=device,
                    progress_callback=cent_callback
                )
                st.session_state.last_cent_result = c_res
                st.session_state.benchmark_results["Centralized"] = c_res
                os.makedirs("results", exist_ok=True)
                try:
                    torch.save(c_res, CENT_CACHE_FILE)
                    torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                except Exception:
                    pass
                render_top_kpi_cards()
                st.success(f"Centralized Baseline trained! Global Test MAE: {c_res['mae']:.2f} | RMSE: {c_res['rmse']:.2f}")
                sidebar_status_box.success(f"✓ Centralized Baseline complete! (MAE: {c_res['mae']:.2f})")
            except Exception as e:
                import traceback
                st.error(f"❌ Centralized Baseline Failed: {str(e)}")
                sidebar_status_box.error(f"❌ Centralized Failed: {str(e)}")
                st.code(traceback.format_exc())

# Trigger 3-Seeds Experiment
if run_three_seeds_clicked:
    sidebar_status_box.info("⏳ Evaluating 3-Seed Robustness (Seeds 42, 43, 44)...")
    with st.spinner("Evaluating multi-seed robustness across seeds 42, 43, 44..."):
        with training_placeholder.container():
            st.markdown("### 🎲 Running 3-Seed Robustness Evaluation (Seeds 42, 43, 44)")
            try:
                seed_prog = st.progress(0)
                seed_stat = st.empty()

                def seed_cb(idx, total_s, s_info):
                    seed_prog.progress(idx / total_s)
                    seed_stat.markdown(f"✓ **Seed {s_info['seed']} Finished:** MAE: `{s_info['mae']:.2f}` | Comm: `{s_info['comm_mb']:.3f} MB`")

                robust_res = run_three_seeds_experiment(
                    df=st.session_state.traffic_df,
                    seeds=[42, 43, 44],
                    num_clients=num_clients,
                    mode=dataset_mode,
                    precision=precision_mode,
                    num_rounds=federated_rounds,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    window_length=window_length,
                    train_ratio=train_ratio,
                    hidden_size=hidden_units,
                    device=device,
                    seed_callback=seed_cb
                )
                st.session_state.three_seed_results = robust_res
                os.makedirs("results", exist_ok=True)
                try:
                    torch.save(robust_res, ROBUST_CACHE_FILE)
                except Exception:
                    pass
                render_top_kpi_cards()
                st.success("3-Seed robustness evaluation complete! View detailed metrics in 'Multi-Seed Robustness' tab.")
                sidebar_status_box.success(f"✓ 3 Seeds complete! Mean MAE: {robust_res['mean_mae']:.2f} ± {robust_res['std_mae']:.2f}")
            except Exception as e:
                import traceback
                st.error(f"❌ Multi-Seed Robustness Evaluation Failed: {str(e)}")
                sidebar_status_box.error(f"❌ 3-Seed Failed: {str(e)}")
                st.code(traceback.format_exc())

# Default display inside Tab 3 if idle
with tab_simulation:
    if st.session_state.last_fl_result is None:
        st.info("👈 Use the sidebar to launch **⚡ START DEMO** (fast 4-round demo) or **🚀 Run Federated Learning** to view live round-by-round client training and FedAvg updates.")
    else:
        fl_res = st.session_state.last_fl_result
        st.markdown("#### 🏁 Latest Federated Training Run Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Federated Rounds", f"{fl_res['rounds_completed']}")
        c2.metric("Final MAE", f"{fl_res['final_mae']:.2f} veh")
        c3.metric("Final RMSE", f"{fl_res['final_rmse']:.2f} veh")
        c4.metric("Total Communication", f"{fl_res['communication_summary']['total_mb']:.3f} MB")

        st.markdown("##### 📜 Round-by-Round Progression")
        hist_df = pd.DataFrame(fl_res["history"])[["round", "mae", "rmse", "round_comm_kb", "total_comm_mb", "precision"]]
        st.dataframe(hist_df, use_container_width=True)

# ==========================================
# TAB 4: PREDICTIONS
# ==========================================
with tab_predictions:
    st.subheader("Traffic Flow Prediction Results (Actual vs Predicted)")

    if st.session_state.last_fl_result is None and st.session_state.last_cent_result is None:
        st.warning("⚠️ No trained models available yet. Please run an experiment or start Demo mode.")
    else:
        available_cids = []
        if st.session_state.last_fl_result:
            available_cids = list(st.session_state.last_fl_result["client_predictions"].keys())
        elif st.session_state.last_cent_result:
            available_cids = list(st.session_state.last_cent_result["client_predictions"].keys())

        selected_cid = st.selectbox("Select Intersection / Client for Inspection", available_cids, format_func=lambda x: f"Intersection / {x.replace('_', ' ').title()}")

        col_p_plot, col_p_met = st.columns([3, 1])

        with col_p_plot:
            fig_pred, ax_pred = plt.subplots(figsize=(10, 4.5))

            if st.session_state.last_fl_result:
                c_fl = st.session_state.last_fl_result["client_predictions"][selected_cid]
                y_act = c_fl["actual"]
                y_fl = c_fl["predictions"]
                inspect_len = min(192, len(y_act))
                ax_pred.plot(y_act[:inspect_len], label="Ground Truth (Actual Flow)", color="#1F2937", linewidth=2.0)
                ax_pred.plot(y_fl[:inspect_len], label="FedAvg Global Model Prediction", color="#3B82F6", linestyle="--", linewidth=2.0)

            if st.session_state.last_cent_result and selected_cid in st.session_state.last_cent_result["client_predictions"]:
                c_cent = st.session_state.last_cent_result["client_predictions"][selected_cid]
                y_cent = c_cent["predictions"]
                inspect_len = min(192, len(y_cent))
                ax_pred.plot(y_cent[:inspect_len], label="Centralized Baseline Prediction", color="#10B981", linestyle=":", linewidth=2.0)

            ax_pred.set_title(f"Traffic Flow Forecast: {selected_cid.upper()} (Test Set Slice)")
            ax_pred.set_xlabel("Time Interval (15-min index)")
            ax_pred.set_ylabel("Traffic Flow (Vehicles / 15-min)")
            ax_pred.legend()
            ax_pred.grid(True, linestyle="--", alpha=0.5)
            fig_pred.tight_layout()
            st.pyplot(fig_pred)

        with col_p_met:
            st.markdown(f"##### 🎯 Client Metrics ({selected_cid.upper()})")
            if st.session_state.last_fl_result:
                fl_c_met = st.session_state.last_fl_result["client_predictions"][selected_cid]["metrics"]
                st.markdown("**FedAvg Performance:**")
                st.write(f"- **MAE:** `{fl_c_met['mae']:.2f}` veh")
                st.write(f"- **RMSE:** `{fl_c_met['rmse']:.2f}` veh")
                st.write(f"- **MAPE:** `{fl_c_met['mape']:.1f}%`")
                st.write(f"- **R²:** `{fl_c_met['r2']:.3f}`")

            if st.session_state.last_cent_result and selected_cid in st.session_state.last_cent_result["client_metrics"]:
                cent_c_met = st.session_state.last_cent_result["client_metrics"][selected_cid]
                st.markdown("---")
                st.markdown("**Centralized Baseline:**")
                st.write(f"- **MAE:** `{cent_c_met['mae']:.2f}` veh")
                st.write(f"- **RMSE:** `{cent_c_met['rmse']:.2f}` veh")
                st.write(f"- **MAPE:** `{cent_c_met['mape']:.1f}%`")
                st.write(f"- **R²:** `{cent_c_met['r2']:.3f}`")

# ==========================================
# TAB 5: COMMUNICATION COST (Point 4)
# ==========================================
with tab_communication:
    st.subheader("Communication Cost Evaluation (Core Requirement of PS-003)")

    st.markdown("""
    In Federated Learning, edge intersection nodes exchange **model updates** rather than raw time-series data.
    The exact communication cost per round is determined strictly by the **trainable parameter count ($P$)**,
    **floating-point precision ($B$)**, the bidirectional upload/download factor (**$2$**), and the number of active clients (**$K$**).
    """)

    st.latex(r"\text{Communication per Round} = P \times B \times 2 \times K")

    cur_p = TrafficGRU(hidden_size=hidden_units).count_parameters()
    comm_stats_fp32 = calculate_total_communication(cur_p, num_clients, federated_rounds, "FP32")
    comm_stats_fp16 = calculate_total_communication(cur_p, num_clients, federated_rounds, "FP16")
    comp_data = compare_fp32_vs_fp16(cur_p, num_clients, federated_rounds)

    # Point 4: Exact metrics display
    c_box1, c_box2, c_box3, c_box4 = st.columns(4)
    c_box1.metric("Parameter Count (P)", f"{cur_p:,} weights", "PyTorch GRU")
    c_box2.metric("FP32 Comm / Round", f"{comm_stats_fp32['total_round_kb']:.2f} KB", f"{comm_stats_fp32['total_round_bytes']:,} Bytes")
    c_box3.metric("FP16 Comm / Round", f"{comm_stats_fp16['total_round_kb']:.2f} KB", f"-50.0% Bandwidth")
    c_box4.metric("Saved across Experiment", f"{comp_data['saved_mb']:.3f} MB", "50% Savings")

    st.markdown("---")
    col_comm_table, col_comm_chart = st.columns([1, 1])

    with col_comm_table:
        st.markdown("##### 📋 Exact Bandwidth Breakdown")
        comm_df = pd.DataFrame([
            {
                "Parameter / Field": "Parameter Count (P)",
                "Baseline FL (FP32)": f"{cur_p:,} weights",
                "Optimized FL (FP16)": f"{cur_p:,} weights",
                "Bandwidth Savings": "Identical Model"
            },
            {
                "Parameter / Field": "Bytes per Parameter (B)",
                "Baseline FL (FP32)": "4 Bytes / param",
                "Optimized FL (FP16)": "2 Bytes / param",
                "Bandwidth Savings": "50.0% Reduction"
            },
            {
                "Parameter / Field": "Client Upload Bytes (P * B)",
                "Baseline FL (FP32)": f"{comm_stats_fp32['upload_bytes_per_client']:,} Bytes",
                "Optimized FL (FP16)": f"{comm_stats_fp16['upload_bytes_per_client']:,} Bytes",
                "Bandwidth Savings": "50.0% Reduction"
            },
            {
                "Parameter / Field": "Client Download Bytes (P * B)",
                "Baseline FL (FP32)": f"{comm_stats_fp32['download_bytes_per_client']:,} Bytes",
                "Optimized FL (FP16)": f"{comm_stats_fp16['download_bytes_per_client']:,} Bytes",
                "Bandwidth Savings": "50.0% Reduction"
            },
            {
                "Parameter / Field": "Per-Client Roundtrip (2 * P * B)",
                "Baseline FL (FP32)": f"{comm_stats_fp32['per_client_round_bytes']:,} Bytes",
                "Optimized FL (FP16)": f"{comm_stats_fp16['per_client_round_bytes']:,} Bytes",
                "Bandwidth Savings": "50.0% Reduction"
            },
            {
                "Parameter / Field": f"Total Bytes / Round ({num_clients} Clients)",
                "Baseline FL (FP32)": f"{comm_stats_fp32['total_round_bytes']:,} B ({comm_stats_fp32['total_round_kb']:.2f} KB)",
                "Optimized FL (FP16)": f"{comm_stats_fp16['total_round_bytes']:,} B ({comm_stats_fp16['total_round_kb']:.2f} KB)",
                "Bandwidth Savings": "50.0% Reduction"
            },
            {
                "Parameter / Field": f"Total Communication ({federated_rounds} Rounds)",
                "Baseline FL (FP32)": f"{comm_stats_fp32['total_mb']:.4f} MB",
                "Optimized FL (FP16)": f"{comm_stats_fp16['total_mb']:.4f} MB",
                "Bandwidth Savings": "50.0% Reduction"
            }
        ])
        st.dataframe(comm_df, use_container_width=True)

    with col_comm_chart:
        r_range = np.arange(1, federated_rounds + 1)
        fp32_cum = [calculate_total_communication(cur_p, num_clients, r, "FP32")["total_mb"] for r in r_range]
        fp16_cum = [calculate_total_communication(cur_p, num_clients, r, "FP16")["total_mb"] for r in r_range]

        fig_comm, ax_comm = plt.subplots(figsize=(6, 4))
        ax_comm.plot(r_range, fp32_cum, label="Baseline FL (FP32)", marker="o", color="#EF4444", linewidth=2)
        ax_comm.plot(r_range, fp16_cum, label="Optimized FL (FP16)", marker="s", color="#10B981", linewidth=2)
        ax_comm.fill_between(r_range, fp16_cum, fp32_cum, color="#10B981", alpha=0.15, label="Bandwidth Savings (50%)")
        ax_comm.set_title("Communication Cost vs Federated Rounds (FP32 vs FP16)")
        ax_comm.set_xlabel("Federated Rounds")
        ax_comm.set_ylabel("Total Communication (MB)")
        ax_comm.legend()
        ax_comm.grid(True, linestyle="--", alpha=0.5)
        fig_comm.tight_layout()
        st.pyplot(fig_comm)

# ==========================================
# TAB 6: PERFORMANCE BENCHMARK (Point 9)
# ==========================================
with tab_benchmark:
    st.subheader("Performance Benchmark: Centralized vs Federated Learning")

    st.markdown("""
    This table compares all evaluated approaches. If an experiment has not been executed yet,
    it strictly displays **'Not Run'** without inventing or fabricating numbers.
    """)

    # Quick Run Buttons inside Benchmark Tab
    st.markdown("##### ⚡ Quick Benchmark Runners")
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    run_b_cent = b_col1.button("Run Centralized", use_container_width=True)
    run_b_iid = b_col2.button("Run FedAvg (IID)", use_container_width=True)
    run_b_noniid = b_col3.button("Run FedAvg (Non-IID)", use_container_width=True)
    run_b_opt = b_col4.button("Run Optimized FL", use_container_width=True)
    run_b_all = b_col5.button("🔥 Run All 4 Benchmarks", type="primary", use_container_width=True)

    bench_status_box = st.empty()

    if run_b_cent:
        bench_status_box.info("⏳ Training Centralized Baseline GRU on pooled client data...")
        with st.spinner("Training Centralized Baseline on pooled client data..."):
            try:
                c_res = run_centralized_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode=dataset_mode,
                    epochs=federated_rounds * local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.last_cent_result = c_res
                st.session_state.benchmark_results["Centralized"] = c_res
                os.makedirs("results", exist_ok=True)
                torch.save(c_res, CENT_CACHE_FILE)
                torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                render_top_kpi_cards()
                bench_status_box.success(f"✓ Centralized Baseline completed! MAE: {c_res['mae']:.2f} veh | RMSE: {c_res['rmse']:.2f} veh | Comm: N/A")
            except Exception as e:
                import traceback
                bench_status_box.error(f"❌ Centralized Baseline Failed: {str(e)}")
                st.code(traceback.format_exc())

    if run_b_iid:
        bench_status_box.info("⏳ Running FedAvg (IID Partition, FP32)...")
        with st.spinner("Executing FedAvg (IID)..."):
            try:
                res = run_federated_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="IID", precision="FP32",
                    num_rounds=federated_rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.benchmark_results["FedAvg_IID"] = res
                os.makedirs("results", exist_ok=True)
                torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                render_top_kpi_cards()
                bench_status_box.success(f"✓ FedAvg (IID) completed! MAE: {res['final_mae']:.2f} veh | RMSE: {res['final_rmse']:.2f} veh | Comm: {res['communication_summary']['total_round_kb']:.2f} KB/round")
            except Exception as e:
                import traceback
                bench_status_box.error(f"❌ FedAvg (IID) Failed: {str(e)}")
                st.code(traceback.format_exc())

    if run_b_noniid:
        bench_status_box.info("⏳ Running FedAvg (Non-IID Partition, FP32)...")
        with st.spinner("Executing FedAvg (Non-IID)..."):
            try:
                res = run_federated_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="NON-IID", precision="FP32",
                    num_rounds=federated_rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.last_fl_result = res
                st.session_state.benchmark_results["FedAvg_NonIID"] = res
                os.makedirs("results", exist_ok=True)
                torch.save(res, FL_CACHE_FILE)
                torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                render_top_kpi_cards()
                bench_status_box.success(f"✓ FedAvg (Non-IID) completed! MAE: {res['final_mae']:.2f} veh | RMSE: {res['final_rmse']:.2f} veh | Comm: {res['communication_summary']['total_round_kb']:.2f} KB/round")
            except Exception as e:
                import traceback
                bench_status_box.error(f"❌ FedAvg (Non-IID) Failed: {str(e)}")
                st.code(traceback.format_exc())

    if run_b_opt:
        bench_status_box.info("⏳ Running Optimized FL (Non-IID Partition, FP16 Quantized)...")
        with st.spinner("Executing Optimized FL (FP16)..."):
            try:
                res = run_federated_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="NON-IID", precision="FP16",
                    num_rounds=federated_rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.benchmark_results["Optimized_FL"] = res
                os.makedirs("results", exist_ok=True)
                torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                render_top_kpi_cards()
                bench_status_box.success(f"✓ Optimized FL (FP16) completed! MAE: {res['final_mae']:.2f} veh | RMSE: {res['final_rmse']:.2f} veh | Comm: {res['communication_summary']['total_round_kb']:.2f} KB/round (-50% bandwidth)")
            except Exception as e:
                import traceback
                bench_status_box.error(f"❌ Optimized FL Failed: {str(e)}")
                st.code(traceback.format_exc())

    if run_b_all:
        bench_status_box.info("⏳ Executing all 4 benchmarks sequentially...")
        with st.spinner("Running All 4 Benchmarks (Centralized, FedAvg IID, FedAvg Non-IID, Optimized FL)..."):
            try:
                bench_progress = st.progress(0)

                # 1. Centralized
                c_res = run_centralized_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="NON-IID",
                    epochs=federated_rounds * local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.last_cent_result = c_res
                st.session_state.benchmark_results["Centralized"] = c_res
                bench_progress.progress(25)

                # 2. FedAvg IID
                res_iid = run_federated_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="IID", precision="FP32",
                    num_rounds=federated_rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.benchmark_results["FedAvg_IID"] = res_iid
                bench_progress.progress(50)

                # 3. FedAvg Non-IID
                res_noniid = run_federated_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="NON-IID", precision="FP32",
                    num_rounds=federated_rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.benchmark_results["FedAvg_NonIID"] = res_noniid
                st.session_state.last_fl_result = res_noniid
                bench_progress.progress(75)

                # 4. Optimized FL (FP16)
                res_opt = run_federated_experiment(
                    df=st.session_state.traffic_df, num_clients=num_clients, mode="NON-IID", precision="FP16",
                    num_rounds=federated_rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                    batch_size=batch_size, window_length=window_length, train_ratio=train_ratio,
                    seed=random_seed, hidden_size=hidden_units, device=device
                )
                st.session_state.benchmark_results["Optimized_FL"] = res_opt
                bench_progress.progress(100)

                os.makedirs("results", exist_ok=True)
                torch.save(c_res, CENT_CACHE_FILE)
                torch.save(res_noniid, FL_CACHE_FILE)
                torch.save(st.session_state.benchmark_results, BENCH_CACHE_FILE)
                render_top_kpi_cards()
                bench_status_box.success("✓ All 4 benchmarks executed and persisted successfully!")
            except Exception as e:
                import traceback
                bench_status_box.error(f"❌ Benchmarks Run Failed: {str(e)}")
                st.code(traceback.format_exc())

    # Point 9: Exact Comparison Table Format
    b_data = st.session_state.benchmark_results
    cent_r = b_data.get("Centralized")
    iid_r = b_data.get("FedAvg_IID")
    noniid_r = b_data.get("FedAvg_NonIID")
    opt_r = b_data.get("Optimized_FL")

    final_comparison_rows = [
        {
            "Method": "Centralized",
            "Data Setting": "Combined",
            "MAE": f"{cent_r['mae']:.2f} veh" if cent_r else "Not Run",
            "RMSE": f"{cent_r['rmse']:.2f} veh" if cent_r else "Not Run",
            "Communication/Round": "N/A (Centralized Cloud)"
        },
        {
            "Method": "FedAvg IID",
            "Data Setting": "IID",
            "MAE": f"{iid_r['final_mae']:.2f} veh" if iid_r else "Not Run",
            "RMSE": f"{iid_r['final_rmse']:.2f} veh" if iid_r else "Not Run",
            "Communication/Round": f"{iid_r['communication_summary']['total_round_kb']:.2f} KB (FP32)" if iid_r else "Not Run"
        },
        {
            "Method": "FedAvg Non-IID",
            "Data Setting": "NON-IID",
            "MAE": f"{noniid_r['final_mae']:.2f} veh" if noniid_r else "Not Run",
            "RMSE": f"{noniid_r['final_rmse']:.2f} veh" if noniid_r else "Not Run",
            "Communication/Round": f"{noniid_r['communication_summary']['total_round_kb']:.2f} KB (FP32)" if noniid_r else "Not Run"
        },
        {
            "Method": "Optimized FL",
            "Data Setting": "NON-IID",
            "MAE": f"{opt_r['final_mae']:.2f} veh" if opt_r else "Not Run",
            "RMSE": f"{opt_r['final_rmse']:.2f} veh" if opt_r else "Not Run",
            "Communication/Round": f"{opt_r['communication_summary']['total_round_kb']:.2f} KB (FP16)" if opt_r else "Not Run"
        }
    ]
    st.markdown("##### 📊 Final Comparison Table")
    st.table(pd.DataFrame(final_comparison_rows))

    # Real Graphs from Experiment Data
    st.markdown("---")
    st.markdown("##### 📈 Experiment Graphs (Real Experiment Data)")

    if st.session_state.last_fl_result is not None:
        fl_h = st.session_state.last_fl_result["history"]
        rounds_x = [h["round"] for h in fl_h]
        mae_y = [h["mae"] for h in fl_h]
        rmse_y = [h["rmse"] for h in fl_h]
        comm_y = [h["total_comm_mb"] for h in fl_h]

        gcol1, gcol2, gcol3 = st.columns(3)

        # Graph 1: MAE vs Federated Round
        with gcol1:
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            ax1.plot(rounds_x, mae_y, marker="o", color="#3B82F6", linewidth=2, label="FedAvg MAE")
            if cent_r:
                ax1.axhline(cent_r["mae"], color="#10B981", linestyle="--", label="Centralized MAE")
            ax1.set_title("Graph 1: MAE vs Federated Round")
            ax1.set_xlabel("Federated Round")
            ax1.set_ylabel("MAE (veh)")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend()
            fig1.tight_layout()
            st.pyplot(fig1)

        # Graph 2: RMSE vs Federated Round
        with gcol2:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.plot(rounds_x, rmse_y, marker="s", color="#8B5CF6", linewidth=2, label="FedAvg RMSE")
            if cent_r:
                ax2.axhline(cent_r["rmse"], color="#10B981", linestyle="--", label="Centralized RMSE")
            ax2.set_title("Graph 2: RMSE vs Federated Round")
            ax2.set_xlabel("Federated Round")
            ax2.set_ylabel("RMSE (veh)")
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend()
            fig2.tight_layout()
            st.pyplot(fig2)

        # Graph 3: Communication Cost vs Federated Round
        with gcol3:
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            ax3.plot(rounds_x, comm_y, marker="^", color="#F59E0B", linewidth=2)
            ax3.set_title("Graph 3: Comm Cost vs Round")
            ax3.set_xlabel("Federated Round")
            ax3.set_ylabel("Total Communication (MB)")
            ax3.grid(True, linestyle=":", alpha=0.5)
            fig3.tight_layout()
            st.pyplot(fig3)

        # Comparison Bar Graphs if Centralized is available
        if cent_r is not None:
            gcol4, gcol5 = st.columns(2)
            with gcol4:
                fig4, ax4 = plt.subplots(figsize=(6, 3.5))
                methods = ["Centralized Baseline", f"FedAvg ({dataset_mode})"]
                maes = [cent_r["mae"], st.session_state.last_fl_result["final_mae"]]
                ax4.bar(methods, maes, color=["#10B981", "#3B82F6"], width=0.45)
                ax4.set_title("Graph 4: Centralized vs Federated MAE")
                ax4.set_ylabel("MAE (Vehicles)")
                for idx, v in enumerate(maes):
                    ax4.text(idx, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")
                fig4.tight_layout()
                st.pyplot(fig4)

            with gcol5:
                fig5, ax5 = plt.subplots(figsize=(6, 3.5))
                rmses = [cent_r["rmse"], st.session_state.last_fl_result["final_rmse"]]
                ax5.bar(methods, rmses, color=["#10B981", "#8B5CF6"], width=0.45)
                ax5.set_title("Graph 5: Centralized vs Federated RMSE")
                ax5.set_ylabel("RMSE (Vehicles)")
                for idx, v in enumerate(rmses):
                    ax5.text(idx, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")
                fig5.tight_layout()
                st.pyplot(fig5)
    else:
        st.info("Run an experiment or click '⚡ START DEMO' to plot live convergence graphs from experiment data.")

# ==========================================
# TAB 7: MULTI-SEED ROBUSTNESS (Point 6F)
# ==========================================
with tab_robustness:
    st.subheader("Multi-Seed Robustness Evaluation (Seeds 42, 43, 44)")
    st.markdown("""
    Federated Learning convergence can vary depending on parameter initialization and data order.
    Evaluating across **3 random seeds** demonstrates model stability and robustness.
    """)

    tab7_status_box = st.empty()
    r_col_btn, r_col_info = st.columns([1, 2])
    with r_col_btn:
        tab7_run_robust = st.button("🎲 Run 3 Seeds Robustness (42, 43, 44)", type="primary", use_container_width=True, key="tab7_robust_btn")

    if tab7_run_robust:
        tab7_status_box.info("⏳ Running multi-seed robustness across seeds 42, 43, and 44...")
        with st.spinner("Evaluating Seeds 42, 43, 44 on CPU..."):
            try:
                seed_prog7 = st.progress(0)
                seed_stat7 = st.empty()
                def tab7_seed_cb(idx, total_s, s_info):
                    seed_prog7.progress(idx / total_s)
                    seed_stat7.markdown(f"✓ **Seed {s_info['seed']} Finished:** MAE: `{s_info['mae']:.2f}` | Comm: `{s_info['comm_mb']:.3f} MB`")

                robust_res = run_three_seeds_experiment(
                    df=st.session_state.traffic_df,
                    seeds=[42, 43, 44],
                    num_clients=num_clients,
                    mode=dataset_mode,
                    precision=precision_mode,
                    num_rounds=federated_rounds,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    window_length=window_length,
                    train_ratio=train_ratio,
                    hidden_size=hidden_units,
                    device=device,
                    seed_callback=tab7_seed_cb
                )
                st.session_state.three_seed_results = robust_res
                os.makedirs("results", exist_ok=True)
                torch.save(robust_res, ROBUST_CACHE_FILE)
                render_top_kpi_cards()
                tab7_status_box.success(f"✓ 3-Seed robustness evaluation completed! Mean MAE: {robust_res['mean_mae']:.2f} ± {robust_res['std_mae']:.2f} veh")
            except Exception as e:
                import traceback
                tab7_status_box.error(f"❌ Multi-Seed Robustness Evaluation Failed: {str(e)}")
                st.code(traceback.format_exc())

    if st.session_state.three_seed_results is None:
        st.info("Click **'🎲 Run 3 Seeds Robustness'** above or in the sidebar to evaluate stability across Seeds 42, 43, 44.")
    else:
        rob_data = st.session_state.three_seed_results
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Mean MAE ± Std", f"{rob_data['mean_mae']:.2f} ± {rob_data['std_mae']:.2f} veh")
        rcol2.metric("Mean RMSE ± Std", f"{rob_data['mean_rmse']:.2f} ± {rob_data['std_rmse']:.2f} veh")
        rcol3.metric("Mean Comm (MB)", f"{rob_data['mean_comm_mb']:.4f} MB", f"Std: {rob_data['std_comm_mb']:.4f}")

        st.markdown("##### 📋 Measured Seed Results")
        s_table = pd.DataFrame(rob_data["seeds"])
        st.table(s_table)

        fig_seed, ax_seed = plt.subplots(figsize=(8, 3.5))
        seed_names = [f"Seed {s['seed']}" for s in rob_data["seeds"]]
        seed_maes = [s["mae"] for s in rob_data["seeds"]]
        seed_rmses = [s["rmse"] for s in rob_data["seeds"]]

        x = np.arange(len(seed_names))
        width = 0.35
        ax_seed.bar(x - width/2, seed_maes, width, label="MAE (veh)", color="#3B82F6")
        ax_seed.bar(x + width/2, seed_rmses, width, label="RMSE (veh)", color="#8B5CF6")
        ax_seed.axhline(rob_data['mean_mae'], color="#3B82F6", linestyle="--", label=f"Mean MAE ({rob_data['mean_mae']:.2f})")
        ax_seed.set_xticks(x)
        ax_seed.set_xticklabels(seed_names)
        ax_seed.set_title("Performance Stability Across 3 Random Seeds")
        ax_seed.set_ylabel("Error (Vehicles / 15-min)")
        ax_seed.legend()
        ax_seed.grid(True, linestyle=":", alpha=0.5)
        fig_seed.tight_layout()
        st.pyplot(fig_seed)

# ==========================================
# TAB 8: EXPERIMENT HISTORY & LOGS
# ==========================================
with tab_logs:
    st.subheader("Persistent Experiment Log & Results Export")
    exp_records = load_all_experiment_records(RESULTS_FILE)

    if not exp_records.empty:
        st.dataframe(exp_records, use_container_width=True)
        csv_bytes = exp_records.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Experiment History (CSV)",
            data=csv_bytes,
            file_name="federated_traffic_experiments.csv",
            mime="text/csv"
        )
    else:
        st.info("No saved experiment runs yet. Run experiments to log results to results/experiments.csv.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem;">
    <strong>PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation</strong><br>
    AIORI Hackathon 3 | Smart Cities Domain | Powered by PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
