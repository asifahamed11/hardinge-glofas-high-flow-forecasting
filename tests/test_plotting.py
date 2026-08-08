from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hardinge_high_flow.config import load_config
from hardinge_high_flow.plotting import generate_publication_figures

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_publication_figure_export(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["figures"]["formats"] = ["png"]
    config["figures"]["png_dpi"] = 120
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    labels = dates.month.isin([7, 8]).astype(int)
    prediction_frames = []
    metric_rows = []
    for horizon in (1, 3):
        for model, offset in (("persistence", 0.0), ("lstm", 0.05)):
            probabilities = np.clip(
                np.where(labels == 1, 0.75 + offset, 0.1 + offset),
                0,
                1,
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "horizon_days": horizon,
                        "seed": 42,
                        "issue_date": dates - pd.to_timedelta(horizon, unit="D"),
                        "target_date": dates,
                        "target_high_flow": labels,
                        "target_value": 10_000 + 3_000 * labels,
                        "probability": probabilities,
                        "threshold": 0.5,
                    }
                )
            )
            metric_rows.append(
                {
                    "model": model,
                    "horizon_days": horizon,
                    "seed": 42,
                    "f1": 0.8 - 0.05 * horizon,
                    "pr_auc": 0.85 - 0.04 * horizon,
                }
            )
    saved = generate_publication_figures(
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        tmp_path,
        config,
    )
    assert len(saved) == 5
    for path in saved:
        assert path.is_file()
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert min(image.size) > 200
