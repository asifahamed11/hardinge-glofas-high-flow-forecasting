from __future__ import annotations

import numpy as np

from hardinge_high_flow.interpretability import grouped_permutation_importance


def test_grouped_permutation_identifies_predictive_feature() -> None:
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 100)
    inputs = rng.normal(size=(len(labels), 3, 2)).astype(np.float32)
    inputs[:, :, 0] = labels[:, None] * 2.0 + rng.normal(
        scale=0.1,
        size=(len(labels), 3),
    )

    def predict(values: np.ndarray) -> np.ndarray:
        scores = values[:, :, 0].mean(axis=1)
        return 1 / (1 + np.exp(-(scores - 1.0)))

    importance = grouped_permutation_importance(
        predict,
        inputs,
        labels,
        ("signal", "noise"),
        repeats=3,
        seed=42,
    )
    grouped = importance.groupby("feature")["importance"].mean()
    assert grouped["signal"] > 0.3
    assert grouped["signal"] > grouped["noise"]
