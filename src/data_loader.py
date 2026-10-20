"""
Data Loader and Synthetic Traffic Dataset Generator for PS-003.
Generates realistic traffic sensor flow records across simulated intersections.
Clearly labelled as: Demo/Synthetic Traffic Data.
"""

import os
import numpy as np
import pandas as pd


DATASET_LABEL = "Demo / Synthetic Traffic Data"


def generate_synthetic_traffic_data(
    num_days: int = 28,
    freq_minutes: int = 15,
    seed: int = 42,
    output_path: str = "data/traffic.csv"
) -> pd.DataFrame:
    """
    Generates synthetic traffic flow measurements for 5 distinct urban intersections.

    Intersection Characteristics:
    - Client 1: Normal urban bimodal traffic (balanced morning & evening peaks).
    - Client 2: Commuter inbound corridor (heavy morning peak 07:00-09:30).
    - Client 3: Commercial / Outbound corridor (heavy evening peak 17:00-20:30).
    - Client 4: High-volume arterial / ring road (elevated base flow, steady midday).
    - Client 5: Residential collector (low overall volume, mild peaks).

    NOTE: This is synthetic data generated strictly for demonstration and benchmarking.
    """
    np.random.seed(seed)

    # 15-minute intervals across the requested duration
    start_date = pd.Timestamp("2026-03-01 00:00:00")
    periods = int((num_days * 24 * 60) / freq_minutes)
    date_range = pd.date_range(start=start_date, periods=periods, freq=f"{freq_minutes}min")

    hours = date_range.hour + date_range.minute / 60.0
    day_of_week = date_range.dayofweek
    is_weekend = (day_of_week >= 5).astype(float)

    def gaussian_peak(center, width, height, h):
        # Circular difference on a 24-hour cycle
        diff = np.minimum(np.abs(h - center), 24.0 - np.abs(h - center))
        return height * np.exp(-0.5 * (diff / width) ** 2)

    # Base diurnal traffic flow pattern (vehicles per 15 min interval)
    # Client 1: Normal bimodal (peaks at 8.5 and 17.5)
    c1_base = 60 + (1 - 0.35 * is_weekend) * (
        gaussian_peak(8.5, 1.4, 160, hours) +
        gaussian_peak(17.5, 1.6, 175, hours) +
        gaussian_peak(13.0, 2.5, 60, hours)
    ) + 0.35 * is_weekend * gaussian_peak(14.0, 3.5, 110, hours)
    c1_noise = np.random.normal(0, 10, periods)
    c1_flow = np.clip(c1_base + c1_noise, 15, 450)

    # Client 2: Morning peak dominant (heavy commuter influx)
    c2_base = 50 + (1 - 0.45 * is_weekend) * (
        gaussian_peak(8.0, 1.2, 260, hours) +   # Dominant morning peak
        gaussian_peak(17.5, 1.6, 90, hours) +   # Modest evening
        gaussian_peak(12.5, 2.0, 45, hours)
    ) + 0.35 * is_weekend * gaussian_peak(13.5, 3.0, 80, hours)
    c2_noise = np.random.normal(0, 11, periods)
    c2_flow = np.clip(c2_base + c2_noise, 12, 480)

    # Client 3: Evening peak dominant (commercial / outbound corridor)
    c3_base = 55 + (1 - 0.30 * is_weekend) * (
        gaussian_peak(8.5, 1.5, 80, hours) +    # Modest morning
        gaussian_peak(18.0, 1.8, 250, hours) +  # Dominant evening peak
        gaussian_peak(14.0, 2.2, 70, hours)
    ) + 0.40 * is_weekend * gaussian_peak(16.0, 3.2, 130, hours)
    c3_noise = np.random.normal(0, 12, periods)
    c3_flow = np.clip(c3_base + c3_noise, 15, 500)

    # Client 4: High-volume arterial highway (high baseline flow throughout)
    c4_base = 180 + (1 - 0.20 * is_weekend) * (
        gaussian_peak(8.5, 1.8, 170, hours) +
        gaussian_peak(17.5, 1.8, 180, hours) +
        gaussian_peak(13.0, 3.0, 110, hours)
    ) + 0.30 * is_weekend * gaussian_peak(14.0, 4.0, 140, hours)
    c4_noise = np.random.normal(0, 15, periods)
    c4_flow = np.clip(c4_base + c4_noise, 80, 600)

    # Client 5: Residential collector (low overall traffic)
    c5_base = 25 + (1 - 0.25 * is_weekend) * (
        gaussian_peak(8.2, 1.1, 75, hours) +
        gaussian_peak(17.8, 1.4, 85, hours) +
        gaussian_peak(12.5, 2.0, 35, hours)
    ) + 0.25 * is_weekend * gaussian_peak(14.0, 3.0, 50, hours)
    c5_noise = np.random.normal(0, 6, periods)
    c5_flow = np.clip(c5_base + c5_noise, 5, 230)

    df = pd.DataFrame({
        "timestamp": date_range,
        "intersection_1": np.round(c1_flow, 1),
        "intersection_2": np.round(c2_flow, 1),
        "intersection_3": np.round(c3_flow, 1),
        "intersection_4": np.round(c4_flow, 1),
        "intersection_5": np.round(c5_flow, 1),
        "dataset_type": DATASET_LABEL,
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def load_traffic_data(data_path: str = "data/traffic.csv") -> tuple[pd.DataFrame, dict]:
    """
    Loads traffic data from local CSV or creates synthetic dataset if missing.
    Returns (DataFrame, metadata_dict).
    """
    is_synthetic = False

    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df = generate_synthetic_traffic_data(num_days=28, freq_minutes=15, output_path=data_path)
        is_synthetic = True
    else:
        df = pd.read_csv(data_path)
        if "dataset_type" in df.columns and any(lbl in df["dataset_type"].values for lbl in ["Demo / Synthetic Traffic Data", "Demo/Synthetic Traffic Data"]):
            is_synthetic = True

    # Validate timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Clean numeric columns
    numeric_cols = [c for c in df.columns if c.startswith("intersection_")]
    if not numeric_cols:
        # Fallback: grab all numeric columns except timestamp
        numeric_cols = [c for c in df.columns if c not in ["timestamp", "dataset_type"]]
        if not numeric_cols:
            raise ValueError("No numeric traffic sensor columns found in dataset.")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()

    metadata = {
        "is_synthetic": is_synthetic,
        "label": DATASET_LABEL if is_synthetic else "Local Traffic Sensor Dataset",
        "total_rows": len(df),
        "available_intersections": numeric_cols,
        "start_time": str(df["timestamp"].iloc[0]) if "timestamp" in df.columns else "N/A",
        "end_time": str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else "N/A",
    }

    return df, metadata
