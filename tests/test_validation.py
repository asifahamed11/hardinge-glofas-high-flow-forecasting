from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hardinge_high_flow.config import load_config
from hardinge_high_flow.validation import (
    generate_rolling_origin_folds,
    relabel_for_fold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rolling_folds_expand_chronologically() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    dates = pd.date_range("1981-01-01", "2023-12-31", freq="D")
    folds = generate_rolling_origin_folds(
        dates,
        config["evaluation"]["rolling_origin"],
    )
    assert len(folds) == 4
    assert all(fold.train_end < fold.validation_end < fold.test_end for fold in folds)
    assert all(
        first.train_end < second.train_end
        for first, second in zip(folds, folds[1:], strict=False)
    )


def test_fold_threshold_uses_fold_training_only() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    dates = pd.date_range("1981-01-01", "2023-12-31", freq="D", name="date")
    values = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame(
        {
            "target_value": values,
            "target_source": "glofas_proxy",
        },
        index=dates,
    )
    fold = generate_rolling_origin_folds(
        dates,
        config["evaluation"]["rolling_origin"],
    )[0]
    labeled, threshold = relabel_for_fold(frame, fold, config)
    expected = frame.loc[: fold.train_end, "target_value"].quantile(
        config["target"]["quantile"]
    )
    assert np.isclose(threshold, expected)
    assert set(labeled["split"]) == {"train", "validation", "test"}
