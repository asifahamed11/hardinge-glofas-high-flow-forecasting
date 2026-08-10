"""Model-agnostic, sequence-aware predictor interpretation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def grouped_permutation_importance(
    predict_probabilities: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Permute one physical feature jointly across every sequence timestep.

    A sample permutation moves the complete feature trajectory from one
    sequence to another.  This retains within-window temporal structure while
    breaking the feature's association with the target and other predictors.
    """

    inputs = np.asarray(inputs, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    names = tuple(map(str, feature_names))
    if inputs.ndim != 3:
        raise ValueError("Permutation inputs must have [sample, time, feature] axes.")
    if len(inputs) != len(labels):
        raise ValueError("Permutation inputs and labels must have equal lengths.")
    if inputs.shape[2] != len(names):
        raise ValueError("Feature names do not match the input feature axis.")
    if repeats < 1:
        raise ValueError("Permutation repeats must be positive.")
    if np.unique(labels).size != 2:
        raise ValueError("Average precision requires both target classes.")

    baseline_probabilities = np.asarray(predict_probabilities(inputs), dtype=float)
    if baseline_probabilities.shape != labels.shape:
        raise ValueError("The predictor must return one probability per sample.")
    baseline = float(average_precision_score(labels, baseline_probabilities))
    rng = np.random.default_rng(seed)
    rows = []
    for feature_index, feature_name in enumerate(names):
        for repeat in range(1, repeats + 1):
            order = rng.permutation(len(inputs))
            permuted = inputs.copy()
            permuted[:, :, feature_index] = inputs[order, :, feature_index]
            probabilities = np.asarray(
                predict_probabilities(permuted),
                dtype=float,
            )
            score = float(average_precision_score(labels, probabilities))
            rows.append(
                {
                    "feature": feature_name,
                    "feature_index": feature_index,
                    "repeat": repeat,
                    "baseline_average_precision": baseline,
                    "permuted_average_precision": score,
                    "importance": baseline - score,
                }
            )
    return pd.DataFrame(rows)
