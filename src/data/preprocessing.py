from __future__ import annotations

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.datasets import load_svmlight_file
from src.data.splits import batch_based_split

#
# Constants
#

GAS_MAP = {
    1: "ethanol",
    2: "ethylene",
    3: "ammonia",
    4: "acetaldehyde",
    5: "acetone",
    6: "toluene",
}

#
# Utilities
#


def extract_number(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    return int(match.group()) if match else 0


def generate_feature_names(num_sensors: int = 16) -> List[str]:
    feature_names = []
    for sensor in range(num_sensors):
        feature_names += [
            f"sensor{sensor}_deltaR",
            f"sensor{sensor}_normDeltaR",
            f"sensor{sensor}_EMAi_0.001",
            f"sensor{sensor}_EMAi_0.01",
            f"sensor{sensor}_EMAi_0.1",
            f"sensor{sensor}_EMAd_0.001",
            f"sensor{sensor}_EMAd_0.01",
            f"sensor{sensor}_EMAd_0.1",
        ]
    return feature_names


#
# Loading .dat files
#


def load_dat_file_svmlight(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    X, y = load_svmlight_file(file_path)
    return X.toarray(), y.astype(int)


#
# Build Base DataFrame
#


def build_base_dataframe(file_paths: List[Path]) -> pd.DataFrame:
    file_paths = sorted(file_paths, key=extract_number)

    if not file_paths:
        raise ValueError("No .dat files found.")

    feature_names = generate_feature_names()

    dfs = []
    global_timestamp = 0

    for i, file in enumerate(file_paths):
        X, y = load_dat_file_svmlight(file)

        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, got {X.shape[1]}"
                f"in file '{file.name}'."
            )

        df = pd.DataFrame(X, columns=feature_names)

        df["target"] = y
        df["target_str"] = df["target"].map(GAS_MAP)

        unknown_mask = df["target_str"].insa()
        if unknown_mask.any():
            unknown_vals = df.loc[unknown_mask, "target"].unique().tolist()
            raise ValueError(
                f"File '{file.name}' contains labels not in GAS_MAP: {unknown_vals}"
            )

        df["batch_id"] = i + 1
        df["batch"] = f"batch_{i + 1}"

        # Create global timestamp
        n_rows = len(df)
        df["timestamp"] = np.arange(global_timestamp, global_timestamp + n_rows)
        global_timestamp += n_rows

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Reorder columns
    cols = ["timestamp", "batch_id", "batch", "target", "target_str"] + feature_names
    full_df = full_df[cols]

    return full_df


#
# Label Encoding
#


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    unique_labels = sorted(df["target_str"].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}

    df = df.copy()
    df["target_encoded"] = df["target_str"].map(label_map)

    return df, label_map


#
# Normalization
#


def fit_normalizer(
    df_train: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from training data only.

    Returns:
        mean : np.ndarray of shape (num_features,)
        std : np.ndarray of shape (num_features,)
    """
    X_train = df_train[feature_cols].values
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    return mean, std


def apply_normalizer(
    df: pd.DataFrame, feature_cols: List[str], mean: np.ndarray, std: np.ndarray
) -> pd.DataFrame:
    """
    Apply pre-computed normalization stats to a DataFrame.

    Returns a new DataFrame; the input is not mutated.
    """
    df = df.copy()
    df[feature_cols] = (df[feature_cols].values - mean) / std
    return df


#
# Windowing
#


def create_windows_from_df(
    df: pd.DataFrame, feature_cols: List[str], window_size: int, stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over the time series and return stacked arrays.

    Every complete window of length `window_size` is included. Any trailing rows that
    can't fill a complete window (at most window_size - 1 rows) are intentionally dropped,
    padding them would fabricate a signal that drift detectors could misinterpret as real
    distribution shift.
    """
    X = df[feature_cols].values
    y = df["target_encoded"].values
    timestamps = df["timestamp"].values
    batch_ids = df["batch_id"].values

    X_windows, y_windows, t_windows, b_windows = [], [], [], []

    T = len(df)

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size

        X_windows.append(X[start:end])
        y_windows.append(y[end - 1])
        t_windows.append(timestamps[end - 1])
        b_windows.append(batch_ids[end - 1])

    X_windows = np.array(X_windows)  # [N, W, F]
    X_windows = np.transpose(X_windows, (0, 2, 1))  # [N, F, W]

    return (
        X_windows,
        np.array(y_windows),
        np.array(t_windows),
        np.array(b_windows),
    )


#
# Full Pipeline
#


def build_full_pipeline(
    raw_data_dir: Path,
    output_dir: Path,
    window_size: int,
    stride: int,
    train_batches: List[int],
    test_batches: List[int],
    normalize: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(exist_ok=True)

    file_paths = list(raw_data_dir.glob("*.dat"))

    # 1. Build base dataset
    df = build_base_dataframe(file_paths)

    # 2. Encode labels
    df, label_map = encode_labels(df)

    feature_cols = [col for col in df.columns if col.startswith("sensor")]

    # 3. Normalize, fit on train rows, apply to all rows
    if normalize:
        train_mask = df["batch_id"].isin(train_batches)
        mean, std = fit_normalizer(df[train_mask], feature_cols)

        df = apply_normalizer(df, feature_cols, mean, std)

        # Persist normalization stats for inference to replicate
        norm_stats = {"mean": mean.tolist(), "std": std.tolist()}
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

    # 4. Save parquet
    parquet_path = output_dir / "gas_sensor.parquet"
    df.to_parquet(parquet_path, index=False)

    # 5. Windowing
    X, y, t, b = create_windows_from_df(df, feature_cols, window_size, stride)

    # 6. Split
    train_idx, test_idx = batch_based_split(
        b, train_batches=train_batches, test_batches=test_batches
    )

    # 7. Save numpy artifacts
    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", y)
    np.save(output_dir / "timestamps.npy", t)
    np.save(output_dir / "batch_ids.npy", b)

    np.save(output_dir / "splits/train_idx.npy", train_idx)
    np.save(output_dir / "splits/test_idx.npy", test_idx)

    # 8. Save label map
    pd.Series(label_map).to_json(output_dir / "label_map.json")
