"""
Preprocessing and Time-Series Windowing Module for PS-003.
Handles chronological train/test splitting, MinMax scaling, sliding window creation,
and IID vs Non-IID client data partitioning.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class TimeSeriesScaler:
    """
    MinMax Scaler fitted strictly on training data to prevent lookahead bias.
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None

    def fit(self, data: np.ndarray):
        self.data_min = float(np.min(data))
        self.data_max = float(np.max(data))
        if self.data_max == self.data_min:
            self.data_max += 1e-6
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.data_max - self.data_min)
        return (data - self.data_min) * scale + self.feature_range[0]

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.data_max - self.data_min)
        return (data - self.feature_range[0]) / scale + self.data_min


def create_sliding_windows(
    series: np.ndarray,
    window_length: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 1D time-series into sliding input-output pairs.
    Input X: shape [N - window_length, window_length, 1]
    Output y: shape [N - window_length, 1]
    """
    X, y = [], []
    for i in range(len(series) - window_length):
        X.append(series[i : i + window_length])
        y.append(series[i + window_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Reshape X to (samples, seq_len, 1)
    if X.ndim == 2:
        X = np.expand_dims(X, axis=-1)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=-1)

    return X, y


def partition_client_data(
    df: pd.DataFrame,
    num_clients: int = 4,
    mode: str = "NON-IID",
    window_length: int = 12,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Dict[str, dict], dict]:
    """
    Partitions traffic data across simulated clients into either IID or NON-IID distributions.

    Modes:
    - NON-IID: Each client receives data exclusively from a distinct intersection
      (Client 1 -> Intersection 1, Client 2 -> Intersection 2, etc.), capturing distinct spatial dynamics.
    - IID: Observations across intersections are uniformly pooled and partitioned round-robin / shuffled
      so that each client experiences an identical statistical mixture of flow regimes.

    Returns:
    - client_dict: {
        'client_1': {
            'name': 'Intersection / Client 1',
            'pattern_desc': '...',
            'X_train': np.ndarray,
            'y_train': np.ndarray,
            'X_test': np.ndarray,
            'y_test': np.ndarray,
            'scaler': TimeSeriesScaler,
            'stats': dict
        },
        ...
      }
    - global_eval: {
        'X_test': np.ndarray,
        'y_test': np.ndarray,
        'raw_y_test': np.ndarray,
        'scaler': TimeSeriesScaler
      }
    """
    num_clients = max(3, min(5, num_clients))
    intersection_cols = [f"intersection_{i+1}" for i in range(num_clients)]

    # Validate columns exist
    for col in intersection_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} missing from dataframe.")

    client_descriptions = {
        1: ("Intersection / Client 1", "Normal Bimodal Urban Traffic (Balanced AM/PM peaks)"),
        2: ("Intersection / Client 2", "Inbound Commuter Corridor (High Morning Peak)"),
        3: ("Intersection / Client 3", "Outbound / Commercial Corridor (High Evening Peak)"),
        4: ("Intersection / Client 4", "High-Volume Arterial Highway (Heavy Baseline Flow)"),
        5: ("Intersection / Client 5", "Residential Collector Feeder (Low Flow Baseline)")
    }

    client_dict = {}

    if mode.upper() == "NON-IID":
        # Each client gets its specific intersection series
        all_X_test = []
        all_y_test = []
        all_raw_y_test = []

        for idx, col in enumerate(intersection_cols):
            client_id = f"client_{idx+1}"
            raw_series = df[col].to_numpy(dtype=np.float32)

            # Chronological split strictly BEFORE windowing
            split_idx = int(len(raw_series) * train_ratio)
            train_raw = raw_series[:split_idx]
            test_raw = raw_series[split_idx:]

            # Scaler fitted exclusively on local training series
            scaler = TimeSeriesScaler()
            train_scaled = scaler.fit_transform(train_raw)
            test_scaled = scaler.transform(test_raw)

            X_train, y_train = create_sliding_windows(train_scaled, window_length)
            X_test, y_test = create_sliding_windows(test_scaled, window_length)

            # Raw unscaled y_test for true metric evaluation
            _, raw_y_test = create_sliding_windows(test_raw, window_length)

            name, desc = client_descriptions[idx + 1]

            client_dict[client_id] = {
                "name": name,
                "pattern_desc": desc,
                "source_col": col,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "raw_y_test": raw_y_test,
                "scaler": scaler,
                "stats": {
                    "mean_flow": float(np.mean(raw_series)),
                    "std_flow": float(np.std(raw_series)),
                    "max_flow": float(np.max(raw_series)),
                    "min_flow": float(np.min(raw_series)),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                }
            }

            all_X_test.append(X_test)
            all_y_test.append(y_test)
            all_raw_y_test.append(raw_y_test)

        # Global test set = concatenated test sets of all active clients
        global_X_test = np.concatenate(all_X_test, axis=0)
        global_y_test = np.concatenate(all_y_test, axis=0)
        global_raw_y_test = np.concatenate(all_raw_y_test, axis=0)

    else:
        # IID MODE:
        # First pool all intersection sequences, normalize globally on train portion,
        # then partition sequences uniformly across clients.
        pooled_train_raw = []
        pooled_test_raw = []

        # Chronological split per intersection
        for col in intersection_cols:
            raw_s = df[col].to_numpy(dtype=np.float32)
            s_idx = int(len(raw_s) * train_ratio)
            pooled_train_raw.append(raw_s[:s_idx])
            pooled_test_raw.append(raw_s[s_idx:])

        # Global scaler fitted across all train portions
        combined_train_raw = np.concatenate(pooled_train_raw)
        global_scaler = TimeSeriesScaler()
        global_scaler.fit(combined_train_raw)

        all_train_X_list, all_train_y_list = [], []
        all_test_X_list, all_test_y_list, all_raw_test_y_list = [], [], []

        for col in intersection_cols:
            raw_s = df[col].to_numpy(dtype=np.float32)
            s_idx = int(len(raw_s) * train_ratio)
            train_s = global_scaler.transform(raw_s[:s_idx])
            test_s = global_scaler.transform(raw_s[s_idx:])

            X_tr, y_tr = create_sliding_windows(train_s, window_length)
            X_te, y_te = create_sliding_windows(test_s, window_length)
            _, raw_y_te = create_sliding_windows(raw_s[s_idx:], window_length)

            all_train_X_list.append(X_tr)
            all_train_y_list.append(y_tr)
            all_test_X_list.append(X_te)
            all_test_y_list.append(y_te)
            all_raw_test_y_list.append(raw_y_te)

        full_train_X = np.concatenate(all_train_X_list, axis=0)
        full_train_y = np.concatenate(all_train_y_list, axis=0)

        # Uniform IID shuffling across clients
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(full_train_X))
        full_train_X = full_train_X[perm]
        full_train_y = full_train_y[perm]

        train_slices = np.array_split(np.arange(len(full_train_X)), num_clients)

        all_X_test = []
        all_y_test = []
        all_raw_y_test = []

        for idx in range(num_clients):
            client_id = f"client_{idx+1}"
            client_indices = train_slices[idx]
            c_X_train = full_train_X[client_indices]
            c_y_train = full_train_y[client_indices]

            # Test set is also balanced across all intersections
            c_X_test = all_test_X_list[idx]
            c_y_test = all_test_y_list[idx]
            c_raw_y_test = all_raw_test_y_list[idx]

            raw_sample_y = global_scaler.inverse_transform(c_y_train)

            name, _ = client_descriptions[idx + 1]

            client_dict[client_id] = {
                "name": name,
                "pattern_desc": "IID Balanced Sample (Uniform distribution across all intersections)",
                "source_col": "Mixed / Uniform",
                "X_train": c_X_train,
                "y_train": c_y_train,
                "X_test": c_X_test,
                "y_test": c_y_test,
                "raw_y_test": c_raw_y_test,
                "scaler": global_scaler,
                "stats": {
                    "mean_flow": float(np.mean(raw_sample_y)),
                    "std_flow": float(np.std(raw_sample_y)),
                    "max_flow": float(np.max(raw_sample_y)),
                    "min_flow": float(np.min(raw_sample_y)),
                    "train_samples": len(c_X_train),
                    "test_samples": len(c_X_test),
                }
            }

            all_X_test.append(c_X_test)
            all_y_test.append(c_y_test)
            all_raw_y_test.append(c_raw_y_test)

        global_X_test = np.concatenate(all_X_test, axis=0)
        global_y_test = np.concatenate(all_y_test, axis=0)
        global_raw_y_test = np.concatenate(all_raw_y_test, axis=0)

    global_eval = {
        "X_test": global_X_test,
        "y_test": global_y_test,
        "raw_y_test": global_raw_y_test,
        "scaler": client_dict["client_1"]["scaler"]
    }

    return client_dict, global_eval
