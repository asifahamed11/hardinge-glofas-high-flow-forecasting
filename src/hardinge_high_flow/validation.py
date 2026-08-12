"""Expanding-window rolling-origin validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path
from .evaluation import event_metrics
from .features import create_sequences, prepare_features
from .reporting import summarize_metrics


@dataclass(frozen=True)
class RollingOriginFold:
    name: str
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def generate_rolling_origin_folds(
    dates: pd.DatetimeIndex,
    settings: dict[str, Any],
) -> list[RollingOriginFold]:
    start_year = int(dates.min().year)
    final_year = int(dates.max().year)
    initial_years = int(settings["initial_training_years"])
    validation_years = int(settings["validation_years"])
    test_years = int(settings["test_years"])
    step_years = int(settings["step_years"])
    minimum_test_years = int(settings["minimum_test_years"])
    maximum_folds = int(settings["maximum_folds"])
    if (
        min(
            initial_years,
            validation_years,
            test_years,
            step_years,
            minimum_test_years,
        )
        < 1
    ):
        raise ValueError("Rolling-origin durations must be positive.")

    folds = []
    for fold_index in range(maximum_folds):
        train_end_year = start_year + initial_years - 1 + fold_index * step_years
        validation_end_year = train_end_year + validation_years
        test_end_year = min(validation_end_year + test_years, final_year)
        available_test_years = test_end_year - validation_end_year
        if available_test_years < minimum_test_years:
            break
        folds.append(
            RollingOriginFold(
                name=(
                    f"fold_{fold_index + 1}_"
                    f"test_{validation_end_year + 1}_{test_end_year}"
                ),
                train_end=pd.Timestamp(f"{train_end_year}-12-31"),
                validation_end=pd.Timestamp(f"{validation_end_year}-12-31"),
                test_end=pd.Timestamp(f"{test_end_year}-12-31"),
            )
        )
    if len(folds) < 2:
        raise ValueError("At least two rolling-origin folds are required.")
    return folds


def relabel_for_fold(
    frame: pd.DataFrame,
    fold: RollingOriginFold,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    subset = frame.loc[frame.index <= fold.test_end].copy()
    subset["split"] = np.select(
        [
            subset.index <= fold.train_end,
            subset.index <= fold.validation_end,
        ],
        ["train", "validation"],
        default="test",
    )
    source = str(subset["target_source"].iloc[0])
    fixed_threshold = config["target"][source].get("fixed_threshold")
    training_values = subset.loc[
        subset["split"] == "train",
        "target_value",
    ].dropna()
    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else float(
            training_values.quantile(
                float(config["target"]["quantile"]),
                interpolation="linear",
            )
        )
    )
    subset["target_high_flow"] = (subset["target_value"] >= threshold).astype(np.int8)
    positive = subset["target_high_flow"].astype(bool)
    event_starts = positive & ~positive.shift(fill_value=False)
    subset["high_flow_event_id"] = event_starts.cumsum().where(positive, 0).astype(int)
    positives = subset.groupby("split")["target_high_flow"].sum()
    if set(positives.index) != {"train", "validation", "test"}:
        raise ValueError(f"Incomplete split in {fold.name}.")
    if (positives == 0).any():
        raise ValueError(f"A split has no positive days in {fold.name}.")
    return subset, threshold


def fold_sequence_counts(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Count usable sequences, positive days, and events in every split."""

    features = prepare_features(frame, config)
    rows = []
    for split in ("train", "validation", "test"):
        for horizon in map(int, config["experiment"]["horizons"]):
            sequences = create_sequences(
                features,
                split,
                int(config["experiment"]["sequence_length"]),
                horizon,
            )
            events = event_metrics(
                sequences.labels,
                sequences.labels.astype(float),
                0.5,
                sequences.target_dates,
                int(config["evaluation"]["event_gap_days"]),
            )
            rows.append(
                {
                    "split": split,
                    "horizon_days": horizon,
                    "usable_sequences": int(len(sequences.labels)),
                    "positive_days": int(sequences.labels.sum()),
                    "prevalence": float(sequences.labels.mean()),
                    "events": int(events["actual_events"]),
                    "first_target_date": sequences.target_dates.min().date().isoformat(),
                    "last_target_date": sequences.target_dates.max().date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def rolling_origin_count_table(config: dict[str, Any]) -> pd.DataFrame:
    """Reconstruct manuscript-facing split counts for every temporal fold."""

    from .experiment import read_labeled_dataset

    base_frame = read_labeled_dataset(config)
    rows = []
    folds = generate_rolling_origin_folds(
        base_frame.index,
        config["evaluation"]["rolling_origin"],
    )
    for fold in folds:
        fold_frame, threshold = relabel_for_fold(base_frame, fold, config)
        counts = fold_sequence_counts(fold_frame, config)
        counts.insert(0, "target_threshold", threshold)
        counts.insert(0, "test_end", fold.test_end.date().isoformat())
        counts.insert(0, "validation_end", fold.validation_end.date().isoformat())
        counts.insert(0, "train_end", fold.train_end.date().isoformat())
        counts.insert(0, "fold", fold.name)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def run_rolling_origin(
    config: dict[str, Any],
    smoke_test: bool = False,
) -> tuple[Path, Path]:
    from .experiment import (
        read_labeled_dataset,
        run_experiment,
        smoke_test_config,
    )

    base_frame = read_labeled_dataset(config)
    folds = generate_rolling_origin_folds(
        base_frame.index,
        config["evaluation"]["rolling_origin"],
    )
    metric_frames = []
    threshold_rows = []
    count_frames = []
    for fold in folds:
        fold_frame, threshold = relabel_for_fold(base_frame, fold, config)
        fold_config = copy.deepcopy(config)
        fold_config["splits"] = {
            "train_end": fold.train_end.date().isoformat(),
            "validation_end": fold.validation_end.date().isoformat(),
            "test_end": fold.test_end.date().isoformat(),
        }
        if smoke_test:
            fold_config = smoke_test_config(fold_config)
        fold_counts = fold_sequence_counts(fold_frame, fold_config)
        fold_counts.insert(0, "fold", fold.name)
        count_frames.append(fold_counts)
        outputs = run_experiment(
            fold_config,
            frame=fold_frame,
            output_namespace=f"rolling_origin/{fold.name}",
            create_figures=False,
        )
        fold_metrics = pd.read_csv(outputs["metrics"])
        fold_metrics.insert(0, "fold", fold.name)
        metric_frames.append(fold_metrics)
        threshold_rows.append(
            {
                "fold": fold.name,
                "train_end": fold.train_end.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "target_threshold": threshold,
            }
        )

    table_directory = resolve_project_path(config, config["paths"]["tables"])
    table_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = table_directory / "rolling_origin_metrics.csv"
    summary_path = table_directory / "rolling_origin_summary.csv"
    threshold_path = table_directory / "rolling_origin_thresholds.csv"
    count_path = table_directory / "rolling_origin_counts.csv"
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(metrics_path, index=False)
    summary = summarize_metrics(all_metrics)
    fold_counts = (
        all_metrics.groupby(["model", "horizon_days"], observed=True)["fold"]
        .nunique()
        .rename("folds")
        .reset_index()
    )
    summary = fold_counts.merge(
        summary,
        on=["model", "horizon_days"],
        validate="one_to_one",
    )
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)
    pd.concat(count_frames, ignore_index=True).to_csv(count_path, index=False)
    return metrics_path, summary_path
