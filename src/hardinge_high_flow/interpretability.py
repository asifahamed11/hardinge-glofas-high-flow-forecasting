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
    feature_groups: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Permute a predictor or predictor family across every sequence timestep.

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
    groups = feature_groups or {
        feature_name: [feature_index]
        for feature_index, feature_name in enumerate(names)
    }
    assigned = sorted(index for indices in groups.values() for index in indices)
    if assigned != list(range(len(names))):
        raise ValueError("Feature groups must partition every feature index once.")
    rows = []
    for feature_name, feature_indices in groups.items():
        for repeat in range(1, repeats + 1):
            order = rng.permutation(len(inputs))
            permuted = inputs.copy()
            permuted[:, :, feature_indices] = inputs[order][:, :, feature_indices]
            probabilities = np.asarray(
                predict_probabilities(permuted),
                dtype=float,
            )
            score = float(average_precision_score(labels, probabilities))
            rows.append(
                {
                    "feature": feature_name,
                    "feature_indices": ";".join(map(str, feature_indices)),
                    "feature_count": int(len(feature_indices)),
                    "repeat": repeat,
                    "baseline_average_precision": baseline,
                    "permuted_average_precision": score,
                    "importance": baseline - score,
                }
            )
    return pd.DataFrame(rows)


def hydrometeorological_feature_groups(
    feature_names: tuple[str, ...] | list[str],
) -> dict[str, list[int]]:
    """Map correlated engineered predictors to physical predictor families."""

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(map(str, feature_names)):
        if name == "glofas_discharge_m3s" or name.startswith("glofas_discharge"):
            family = "Current GloFAS discharge"
        elif "day_of_year" in name:
            family = "Seasonality"
        elif "precipitation_product_difference" in name or name.startswith(
            "mean_precipitation"
        ):
            family = "Cross-product precipitation"
        elif name.startswith("era5_precipitation"):
            family = "ERA5-Land precipitation"
        elif name.startswith("nasa_precipitation"):
            family = "NASA POWER precipitation"
        elif "soil_moisture" in name:
            family = "ERA5-Land soil moisture"
        elif name.startswith("era5_runoff"):
            family = "ERA5-Land runoff"
        elif "temperature" in name:
            family = "Temperature"
        elif "humidity" in name:
            family = "NASA POWER humidity"
        else:
            family = "Other engineered predictors"
        groups.setdefault(family, []).append(index)
    return groups
