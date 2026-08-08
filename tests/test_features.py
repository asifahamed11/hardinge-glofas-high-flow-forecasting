from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hardinge_high_flow.config import load_config
from hardinge_high_flow.features import create_sequences, prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", "2011-12-31", freq="D", name="date")
    annual = np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.2425)
    high_flow = dates.month.isin([7, 8]).astype(np.int8)
    event_ids = pd.Series(high_flow, index=dates).diff().eq(1).cumsum()
    event_ids = event_ids.where(high_flow == 1, 0).astype(int)
    split = np.select(
        [
            dates <= pd.Timestamp("2005-12-31"),
            dates <= pd.Timestamp("2008-12-31"),
        ],
        ["train", "validation"],
        default="test",
    )
    frame = pd.DataFrame(
        {
            "era5_temperature_c": 25 + 5 * annual,
            "era5_soil_moisture_m3m3": 0.3 + 0.05 * annual,
            "era5_precipitation_mm": np.maximum(0, 6 + 5 * annual),
            "era5_runoff_mm": np.maximum(0, 2 + 2 * annual),
            "nasa_temperature_c": 26 + 4 * annual,
            "nasa_precipitation_mm": np.maximum(0, 5 + 4 * annual),
            "nasa_relative_humidity_percent": 70 + 15 * annual,
            "glofas_discharge_m3s": 10_000 + 4_000 * annual,
            "target_value": 10_000 + 4_000 * annual,
            "target_high_flow": high_flow,
            "high_flow_event_id": event_ids.to_numpy(),
            "target_source": "glofas_proxy",
            "split": split,
        },
        index=dates,
    )
    frame.loc[frame["split"] == "test", "nasa_precipitation_mm"] += 1_000
    return frame


def test_training_only_feature_thresholds() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["splits"].update(
        {
            "train_end": "2005-12-31",
            "validation_end": "2008-12-31",
            "test_end": "2011-12-31",
        }
    )
    frame = synthetic_frame()
    features = prepare_features(frame, config)
    expected = frame.loc[
        frame["split"] == "train",
        "nasa_precipitation_mm",
    ].quantile(config["features"]["extreme_quantile"])
    assert np.isclose(
        features.extreme_thresholds["nasa_precipitation_mm"],
        expected,
    )
    assert "glofas_discharge_m3s" not in features.feature_names


def test_sequences_respect_split_and_horizon() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    frame = synthetic_frame()
    features = prepare_features(frame, config)
    sequences = create_sequences(
        features,
        "test",
        sequence_length=12,
        horizon=5,
    )
    assert sequences.inputs.shape[1] == 12
    assert (sequences.target_dates - sequences.issue_dates).days.min() == 5
    assert (sequences.target_dates - sequences.issue_dates).days.max() == 5
    assert features.context.loc[sequences.issue_dates, "split"].eq("test").all()
    assert features.context.loc[sequences.target_dates, "split"].eq("test").all()
