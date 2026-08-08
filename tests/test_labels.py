from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hardinge_high_flow.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
MODULE_PATH = PROJECT_ROOT / "scripts" / "create_high_flow_labels.py"
SPEC = importlib.util.spec_from_file_location("create_labels_script", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load scripts/create_high_flow_labels.py.")
CREATE_LABELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CREATE_LABELS)


def test_target_threshold_uses_training_period_only() -> None:
    dates = pd.date_range("2000-01-01", "2011-12-31", freq="D", name="date")
    values = np.linspace(1_000, 2_000, len(dates))
    frame = pd.DataFrame({"glofas_discharge_m3s": values}, index=dates)
    frame.loc["2009-01-01":, "glofas_discharge_m3s"] += 1_000_000
    config = load_config(DEFAULT_CONFIG)
    config["period"].update(
        {
            "start": "2000-01-01",
            "end": "2011-12-31",
        }
    )
    config["splits"].update(
        {
            "train_end": "2005-12-31",
            "validation_end": "2008-12-31",
            "test_end": "2011-12-31",
        }
    )
    labeled, metadata = CREATE_LABELS.create_labels(
        frame,
        config,
        "glofas_proxy",
    )
    expected = frame.loc[
        :"2005-12-31",
        "glofas_discharge_m3s",
    ].quantile(config["target"]["quantile"])
    assert np.isclose(metadata["threshold"], expected)
    assert labeled["target_source"].eq("glofas_proxy").all()


def test_missing_observed_values_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "observed.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "observed_discharge_m3s": [1_000.0, np.nan],
        }
    ).to_csv(path, index=False)
    config = load_config(DEFAULT_CONFIG)
    config["_project_root"] = str(tmp_path)
    config["target"]["observed"]["path"] = path.name
    with pytest.raises(ValueError, match="missing or non-finite"):
        CREATE_LABELS.read_observed_target(config)


def test_observed_quality_filter_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "observed.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "observed_discharge_m3s": [1_000.0, 2_000.0],
            "quality_flag": ["approved", "rejected"],
        }
    ).to_csv(path, index=False)
    config = load_config(DEFAULT_CONFIG)
    config["_project_root"] = str(tmp_path)
    observed_config = config["target"]["observed"]
    observed_config["path"] = path.name
    observed_config["quality_flag_column"] = "quality_flag"
    observed_config["accepted_quality_flags"] = ["APPROVED"]
    values, _, _ = CREATE_LABELS.read_observed_target(config)
    assert values.index.tolist() == [pd.Timestamp("2020-01-01")]
    assert values.iloc[0] == 1_000.0
