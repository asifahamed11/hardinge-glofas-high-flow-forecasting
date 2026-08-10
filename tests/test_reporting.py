from __future__ import annotations

import numpy as np
import pandas as pd

from hardinge_high_flow.config import load_config
from hardinge_high_flow.reporting import (
    canonicalize_metric_names,
    constant_decision_threshold,
    diagnostic_tables,
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
