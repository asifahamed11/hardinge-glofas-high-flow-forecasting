from __future__ import annotations

import numpy as np
import pandas as pd

from hardinge_high_flow.config import load_config
from hardinge_high_flow.reporting import (
    calibration_diagnostic_table,
    canonicalize_metric_names,
    constant_decision_threshold,
    diagnostic_tables,
    event_definition_sensitivity,
    leave_one_year_out_sensitivity,
)


def test_metric_name_migration_is_explicit() -> None:
    legacy = pd.DataFrame(
        {
            "pr_auc": [0.7],
            "pr_auc_ci_low": [0.6],
            "metric": ["pr_auc"],
        }
    )
    migrated = canonicalize_metric_names(legacy)
    assert list(migrated.columns) == [
        "average_precision",
        "average_precision_ci_low",
        "metric",
    ]
    assert migrated.loc[0, "metric"] == "average_precision"


def test_threshold_is_not_averaged_across_prediction_rows() -> None:
    frame = pd.DataFrame({"threshold": np.repeat(0.999, 1_801)})
    assert frame["threshold"].mean() > 0.999
    assert constant_decision_threshold(frame) == 0.999


def test_persistence_events_at_threshold_remain_detected(project_root) -> None:
    config = load_config(project_root / "configs" / "default.yaml")
    dates = pd.date_range("2020-01-01", periods=1_801, freq="D")
    labels = np.zeros(len(dates), dtype=int)
    labels[100:103] = 1
    probabilities = np.full(len(dates), 0.001)
    probabilities[labels == 1] = 0.999
    predictions = pd.DataFrame(
        {
            "model": "persistence",
            "horizon_days": 1,
            "seed": 0,
            "issue_date": dates - pd.Timedelta(days=1),
            "target_date": dates,
            "target_high_flow": labels,
            "target_value": 10_000 + labels * 40_000,
            "probability": probabilities,
            "threshold": 0.999,
        }
    )
    _, _, magnitude, events = diagnostic_tables(predictions, config)
    assert events["detected"].all()
    assert (magnitude["detection_rate"] == 1.0).all()


def test_sensitivity_tables_preserve_declared_event_definitions(project_root) -> None:
    config = load_config(project_root / "configs" / "default.yaml")
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    labels = (dates.month == 8).astype(int)
    probabilities = np.where(labels == 1, 0.8, 0.05)
    predictions = pd.DataFrame(
        {
            "model": "random_forest",
            "horizon_days": 1,
            "seed": 42,
            "issue_date": dates - pd.Timedelta(days=1),
            "target_date": dates,
            "target_high_flow": labels,
            "target_value": 10_000 + labels * 40_000,
            "probability": probabilities,
            "threshold": 0.5,
        }
    )
    event_table = event_definition_sensitivity(predictions, config)
    assert set(event_table["allowed_intervening_negative_days"]) == {0, 1, 2}
    assert event_table.loc[event_table["primary_definition"], "allowed_intervening_negative_days"].item() == 0
    omitted = leave_one_year_out_sensitivity(predictions, config)
    assert set(omitted["omitted_year"]) == {2019, 2020, 2021, 2022, 2023}
    calibration = calibration_diagnostic_table(predictions, config)
    assert calibration["calibration_interval_clusters"].item() == 5
