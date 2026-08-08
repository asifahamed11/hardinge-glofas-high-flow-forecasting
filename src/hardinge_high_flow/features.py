"""Causal feature engineering and sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureData:
    unscaled: pd.DataFrame
    scaled: pd.DataFrame
    context: pd.DataFrame
    feature_names: tuple[str, ...]
    extreme_thresholds: dict[str, float]
    scaler: StandardScaler


@dataclass(frozen=True)
class SequenceData:
    inputs: np.ndarray
    labels: np.ndarray
    issue_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex


def _validate_input(frame: pd.DataFrame, base_columns: list[str]) -> None:
    required = {
        *base_columns,
        "split",
        "target_high_flow",
        "target_value",
        "glofas_discharge_m3s",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature input columns missing: {sorted(missing)}")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Feature input dates must be unique and sorted.")


def _extreme_thresholds(
    frame: pd.DataFrame,
    columns: list[str],
    quantile: float,
) -> dict[str, float]:
    training = frame.loc[frame["split"] == "train"]
    if training.empty:
        raise ValueError("Training rows are required for feature fitting.")
    return {column: float(training[column].quantile(quantile)) for column in columns}


def _engineer_one_split(
    frame: pd.DataFrame,
    base_columns: list[str],
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    features = frame[base_columns].copy()
    rolling_columns = [
        column
        for column in base_columns
        if any(
            token in column
            for token in ("precipitation", "humidity", "soil_moisture", "runoff")
        )
    ]
    precipitation_columns = [
        column for column in base_columns if "precipitation" in column
    ]

    for column in rolling_columns:
        for window in config["rolling_mean_windows"]:
            features[f"{column}_mean_{window}d"] = (
                frame[column].rolling(int(window), min_periods=int(window)).mean()
            )
        features[f"{column}_change_1d"] = frame[column].diff(1)

    for column in precipitation_columns:
        for window in config["rolling_sum_windows"]:
            features[f"{column}_sum_{window}d"] = (
                frame[column].rolling(int(window), min_periods=int(window)).sum()
            )

    variability_columns = [
        column
        for column in rolling_columns
        if any(token in column for token in ("precipitation", "runoff"))
    ]
    for column in variability_columns:
        for window in config["rolling_std_windows"]:
            features[f"{column}_std_{window}d"] = (
                frame[column].rolling(int(window), min_periods=int(window)).std()
            )

    lag_columns = [
        column
        for column in rolling_columns
        if any(
            token in column for token in ("precipitation", "soil_moisture", "runoff")
        )
    ]
    for column in lag_columns:
        for lag in config["lag_days"]:
            features[f"{column}_lag_{lag}d"] = frame[column].shift(int(lag))

    if bool(config["add_seasonality"]):
        day_of_year = frame.index.dayofyear.to_numpy()
        features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.2425)
        features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.2425)

    if bool(config["add_interactions"]):
        active_columns = set(base_columns)

        if {
            "nasa_precipitation_mm",
            "nasa_relative_humidity_percent",
        }.issubset(active_columns):
            features["nasa_precipitation_humidity"] = (
                frame["nasa_precipitation_mm"]
                * frame["nasa_relative_humidity_percent"]
            )

        if {
            "era5_precipitation_mm",
            "era5_soil_moisture_m3m3",
        }.issubset(active_columns):
            features["era5_precipitation_soil"] = (
                frame["era5_precipitation_mm"]
                * frame["era5_soil_moisture_m3m3"]
            )

        if {
            "era5_runoff_mm",
            "era5_soil_moisture_m3m3",
        }.issubset(active_columns):
            features["era5_runoff_soil"] = (
                frame["era5_runoff_mm"]
                * frame["era5_soil_moisture_m3m3"]
            )

        if {
            "nasa_precipitation_mm",
            "era5_precipitation_mm",
        }.issubset(active_columns):
            features["mean_precipitation_mm"] = (
                frame["nasa_precipitation_mm"]
                + frame["era5_precipitation_mm"]
            ) / 2
            features["precipitation_product_difference_mm"] = (
                frame["nasa_precipitation_mm"]
                - frame["era5_precipitation_mm"]
            )

    for column, threshold in thresholds.items():
        features[f"{column}_training_extreme"] = (frame[column] >= threshold).astype(
            np.int8
        )

    return features


def prepare_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> FeatureData:
    feature_config = config["features"]
    base_columns = list(feature_config["base_columns"])
    if bool(feature_config["include_streamflow"]):
        base_columns.append("glofas_discharge_m3s")
    _validate_input(frame, base_columns)

    precipitation_columns = [
        column for column in base_columns if "precipitation" in column
    ]
    thresholds = _extreme_thresholds(
        frame,
        precipitation_columns,
        float(feature_config["extreme_quantile"]),
    )

    split_features = []
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split]
        if subset.empty:
            raise ValueError(f"Feature split is empty: {split}")
        expected = pd.date_range(subset.index.min(), subset.index.max(), freq="D")
        if not subset.index.equals(expected):
            raise ValueError(f"Feature split contains date gaps: {split}")
        split_features.append(
            _engineer_one_split(
                subset,
                base_columns,
                feature_config,
                thresholds,
            )
        )

    unscaled = pd.concat(split_features).sort_index()
    valid_rows = ~unscaled.isna().any(axis=1)
    unscaled = unscaled.loc[valid_rows].astype(np.float32)
    context_columns = [
        "split",
        "target_high_flow",
        "target_value",
        "glofas_discharge_m3s",
        "high_flow_event_id",
        "target_source",
    ]
    context = frame.loc[unscaled.index, context_columns].copy()

    training_mask = context["split"].eq("train")
    scaler = StandardScaler()
    scaler.fit(unscaled.loc[training_mask])
    scaled_values = scaler.transform(unscaled)
    scaled = pd.DataFrame(
        scaled_values,
        index=unscaled.index,
        columns=unscaled.columns,
    ).astype(np.float32)
    return FeatureData(
        unscaled=unscaled,
        scaled=scaled,
        context=context,
        feature_names=tuple(unscaled.columns),
        extreme_thresholds=thresholds,
        scaler=scaler,
    )


def create_sequences(
    features: FeatureData,
    split: str,
    sequence_length: int,
    horizon: int,
) -> SequenceData:
    mask = features.context["split"].eq(split)
    split_features = features.scaled.loc[mask]
    split_context = features.context.loc[mask]
    if len(split_features) < sequence_length + horizon:
        raise ValueError(f"Insufficient rows for {split}, horizon {horizon}.")

    inputs = []
    labels = []
    issue_dates = []
    target_dates = []
    for issue_position in range(
        sequence_length - 1,
        len(split_features) - horizon,
    ):
        target_position = issue_position + horizon
        issue_date = split_features.index[issue_position]
        target_date = split_features.index[target_position]
        if target_date - issue_date != pd.Timedelta(days=horizon):
            continue
        start_position = issue_position - sequence_length + 1
        window_dates = split_features.index[start_position : issue_position + 1]
        if len(window_dates) != sequence_length:
            continue
        if window_dates[-1] - window_dates[0] != pd.Timedelta(days=sequence_length - 1):
            continue
        inputs.append(
            split_features.iloc[start_position : issue_position + 1].to_numpy()
        )
        labels.append(int(split_context["target_high_flow"].iloc[target_position]))
        issue_dates.append(issue_date)
        target_dates.append(target_date)

    if not inputs:
        raise ValueError(f"No valid sequences for {split}, horizon {horizon}.")
    return SequenceData(
        inputs=np.asarray(inputs, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        issue_dates=pd.DatetimeIndex(issue_dates, name="issue_date"),
        target_dates=pd.DatetimeIndex(target_dates, name="target_date"),
    )


def flatten_sequences(sequences: SequenceData) -> np.ndarray:
    return sequences.inputs.reshape(len(sequences.inputs), -1)
