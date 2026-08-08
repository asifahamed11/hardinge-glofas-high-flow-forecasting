"""End-to-end multi-horizon experiment runner."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.utils.class_weight import compute_sample_weight

from .config import resolve_project_path
from .evaluation import (
    binary_metrics,
    block_bootstrap_intervals,
    cost_loss_analysis,
    event_catalog,
    event_metrics,
    fit_sigmoid_calibrator,
    paired_block_bootstrap_difference,
    select_threshold,
)
from .features import (
    FeatureData,
    SequenceData,
    create_sequences,
    flatten_sequences,
    prepare_features,
)
from .models import (
    make_classical_model,
    make_deep_model,
    predict_deep_probabilities,
    refit_deep_model,
    train_deep_model,
)
from .plotting import generate_publication_figures

LOGGER = logging.getLogger("experiment")
BASELINE_MODELS = ("climatology", "persistence", "glofas_signal")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_file(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def read_labeled_dataset(config: dict[str, Any]) -> pd.DataFrame:
    parquet_path = resolve_project_path(
        config,
        config["paths"]["labeled_dataset_parquet"],
    )
    csv_path = resolve_project_path(
        config,
        config["paths"]["labeled_dataset_csv"],
    )
    if parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
        loaded_path = parquet_path
        output_hash_key = "parquet_sha256"
    elif csv_path.is_file():
        frame = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
        loaded_path = csv_path
        output_hash_key = "csv_sha256"
    else:
        raise FileNotFoundError(
            "Labeled dataset missing. Run scripts/create_high_flow_labels.py first."
        )
    label_metadata_path = resolve_project_path(
        config,
        config["paths"]["label_metadata"],
    )
    if not label_metadata_path.is_file():
        raise FileNotFoundError(
            "Label metadata is missing. Recreate labels before training."
        )
    label_metadata = json.loads(label_metadata_path.read_text(encoding="utf-8"))
    expected_label_hash = label_metadata.get("outputs", {}).get(output_hash_key)
    if expected_label_hash != _sha256_file(loaded_path):
        raise ValueError(
            "Labeled dataset does not match its metadata. Recreate labels."
        )

    master_candidates = (
        (
            "master_dataset_parquet",
            resolve_project_path(config, config["paths"]["master_dataset_parquet"]),
        ),
        (
            "master_dataset_csv",
            resolve_project_path(config, config["paths"]["master_dataset_csv"]),
        ),
    )
    master_name, master_path = next(
        ((name, path) for name, path in master_candidates if path.is_file()),
        ("", Path()),
    )
    master_input = label_metadata.get("inputs", {}).get(master_name)
    if not master_name or not isinstance(master_input, dict):
        raise ValueError(
            "Labels do not record master-dataset provenance. Recreate labels."
        )
    if master_input.get("sha256") != _sha256_file(master_path):
        raise ValueError(
            "Master dataset changed after labels were created. Recreate labels."
        )

    base_columns = set(config["features"]["base_columns"])
    if {"era5_precipitation_mm", "era5_runoff_mm"} & base_columns:
        master_metadata_path = resolve_project_path(
            config, config["paths"]["master_dataset_metadata"]
        )
        if not master_metadata_path.is_file():
            raise ValueError("Master-dataset metadata is required for ERA5 inputs.")
        master_metadata = json.loads(
            master_metadata_path.read_text(encoding="utf-8")
        )
        accumulation_method = (
            master_metadata.get("sources", {})
            .get("era5_land", {})
            .get("accumulation_method")
        )
        if accumulation_method != (
            "ERA5-Land 00 UTC accumulation shifted to the preceding UTC day"
        ):
            raise ValueError(
                "ERA5-Land accumulation provenance is unverified. Re-download "
                "accumulations, rebuild the dataset, and recreate labels."
            )

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Labeled dataset dates are not unique.")
    return frame


def smoke_test_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    settings = smoke["experiment"]["smoke_test"]
    smoke["experiment"]["maximum_epochs"] = int(settings["maximum_epochs"])
    smoke["experiment"]["seeds"] = list(settings["seeds"])
    smoke["experiment"]["horizons"] = list(settings["horizons"])
    smoke["experiment"]["deep_models"] = list(settings["deep_models"])
    smoke["evaluation"]["bootstrap_iterations"] = 50
    return smoke


def _validation_partitions(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(labels)
    for fraction in np.linspace(0.35, 0.65, 13):
        boundary = int(count * fraction)
        calibration = np.arange(0, boundary)
        threshold = np.arange(boundary, count)
        if (
            np.unique(labels[calibration]).size == 2
            and np.unique(labels[threshold]).size == 2
        ):
            return calibration, threshold
    raise ValueError(
        "Validation period cannot support separate calibration "
        "and threshold-selection subsets."
    )


def _deep_training_partitions(
    sequences: SequenceData,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(sequences.labels)
    for fraction in np.linspace(0.75, 0.9, 7):
        boundary = int(count * fraction)
        training = np.arange(0, boundary)
        early_stopping = np.arange(boundary, count)
        if (
            np.unique(sequences.labels[training]).size == 2
            and np.unique(sequences.labels[early_stopping]).size == 2
        ):
            return training, early_stopping
    raise ValueError("Training period cannot support an early-stopping block.")


def _empirical_probability(
    training_values: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    sorted_training = np.sort(np.asarray(training_values, dtype=float))
    ranks = np.searchsorted(sorted_training, values, side="right")
    return (ranks + 0.5) / (len(sorted_training) + 1.0)


def _baseline_probabilities(
    name: str,
    sequences: SequenceData,
    training_sequences: SequenceData,
    features: FeatureData,
) -> np.ndarray:
    if name == "climatology":
        training_months = training_sequences.target_dates.month
        monthly = (
            pd.Series(
                training_sequences.labels,
                index=training_months,
            )
            .groupby(level=0)
            .mean()
        )
        overall = float(training_sequences.labels.mean())
        return np.asarray(
            [
                float(monthly.get(month, overall))
                for month in sequences.target_dates.month
            ]
        )

    if name == "persistence":
        current_labels = features.context.loc[
            sequences.issue_dates,
            "target_high_flow",
        ].to_numpy(dtype=float)
        return np.clip(current_labels, 0.001, 0.999)

    if name == "glofas_signal":
        training_values = features.context.loc[
            training_sequences.issue_dates,
            "glofas_discharge_m3s",
        ].to_numpy(dtype=float)
        issue_values = features.context.loc[
            sequences.issue_dates,
            "glofas_discharge_m3s",
        ].to_numpy(dtype=float)
        return _empirical_probability(training_values, issue_values)

    raise ValueError(f"Unsupported baseline: {name}")


def _calibrate_predictions(
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    calibration_indices: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if method == "none":
        return (
            validation_probabilities,
            test_probabilities,
            {},
        )
    if method != "sigmoid":
        raise ValueError(f"Unsupported calibration method: {method}")
    calibrator = fit_sigmoid_calibrator(
        validation_probabilities[calibration_indices],
        validation_labels[calibration_indices],
    )
    return (
        calibrator.transform(validation_probabilities),
        calibrator.transform(test_probabilities),
        asdict(calibrator),
    )


def _evaluate_run(
    model_name: str,
    horizon: int,
    seed: int,
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    validation_sequences: SequenceData,
    test_sequences: SequenceData,
    features: FeatureData,
    config: dict[str, Any],
    calibration_metadata: dict[str, float],
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _, threshold_indices = _validation_partitions(validation_sequences.labels)
    threshold, validation_score = select_threshold(
        validation_sequences.labels[threshold_indices],
        validation_probabilities[threshold_indices],
        metric=str(config["evaluation"]["threshold_metric"]),
    )
    metrics = binary_metrics(
        test_sequences.labels,
        test_probabilities,
        threshold,
        int(config["evaluation"]["reliability_bins"]),
    )
    metrics.update(
        event_metrics(
            test_sequences.labels,
            test_probabilities,
            threshold,
            test_sequences.target_dates,
            int(config["evaluation"]["event_gap_days"]),
        )
    )
    metrics.update(
        block_bootstrap_intervals(
            test_sequences.labels,
            test_probabilities,
            threshold,
            test_sequences.target_dates,
            int(config["evaluation"]["bootstrap_iterations"]),
            float(config["evaluation"]["confidence_level"]),
            seed + horizon * 10_000,
            int(config["evaluation"]["reliability_bins"]),
        )
    )
    metrics.update(
        {
            "model": model_name,
            "horizon_days": horizon,
            "seed": seed,
            "validation_threshold_score": validation_score,
            "calibration_coefficient": calibration_metadata.get(
                "coefficient",
                np.nan,
            ),
            "calibration_intercept": calibration_metadata.get(
                "intercept",
                np.nan,
            ),
        }
    )
    if extra_metadata:
        metrics.update(extra_metadata)

    target_values = features.context.loc[
        test_sequences.target_dates,
        "target_value",
    ].to_numpy(dtype=float)
    target_sources = (
        features.context.loc[
            test_sequences.target_dates,
            "target_source",
        ]
        .astype(str)
        .to_numpy()
    )
    predictions = pd.DataFrame(
        {
            "model": model_name,
            "horizon_days": horizon,
            "seed": seed,
            "issue_date": test_sequences.issue_dates,
            "target_date": test_sequences.target_dates,
            "target_high_flow": test_sequences.labels,
            "target_value": target_values,
            "target_source": target_sources,
            "probability": test_probabilities,
            "threshold": threshold,
            "prediction": (test_probabilities >= threshold).astype(np.int8),
        }
    )
    return metrics, predictions


def _save_deep_checkpoint(
    model_name: str,
    horizon: int,
    seed: int,
    state_dict: dict[str, torch.Tensor],
    training_metadata: dict[str, Any],
    feature_names: tuple[str, ...],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"{model_name}_h{horizon}_seed{seed}.pt"
    torch.save(
        {
            "model_name": model_name,
            "horizon_days": horizon,
            "seed": seed,
            "state_dict": state_dict,
            "feature_names": list(feature_names),
            **training_metadata,
        },
        destination,
    )


def _paired_comparisons(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Make date- and seed-aligned paired comparisons.

    Per-seed evaluation preserves each validation-selected decision threshold.
    Averaging probabilities across seeds and borrowing the first seed's
    threshold would not define a valid ensemble classifier.
    """
    rows = []
    comparison_index = 0
    for horizon in sorted(predictions["horizon_days"].unique()):
        horizon_data = predictions[predictions["horizon_days"] == horizon]
        models = set(horizon_data["model"])
        comparison_sets = {
            "persistence": models - {"persistence"},
            "logistic_regression": models
            - set(BASELINE_MODELS)
            - {"logistic_regression"},
        }
        for reference_name, candidate_models in comparison_sets.items():
            reference_all = horizon_data[
                horizon_data["model"] == reference_name
            ]
            if reference_all.empty:
                continue
            reference_seeds = set(reference_all["seed"].astype(int))
            for model in sorted(candidate_models):
                model_all = horizon_data[horizon_data["model"] == model]
                for seed in sorted(model_all["seed"].astype(int).unique()):
                    model_data = model_all[model_all["seed"] == seed].sort_values(
                        "target_date"
                    )
                    reference_seed = seed if seed in reference_seeds else min(
                        reference_seeds
                    )
                    reference_data = reference_all[
                        reference_all["seed"] == reference_seed
                    ].sort_values("target_date")
                    merged = model_data[
                        [
                            "target_date",
                            "target_high_flow",
                            "probability",
                            "threshold",
                        ]
                    ].merge(
                        reference_data[
                            [
                                "target_date",
                                "target_high_flow",
                                "probability",
                                "threshold",
                            ]
                        ],
                        on=["target_date", "target_high_flow"],
                        suffixes=("_model", "_reference"),
                        validate="one_to_one",
                    )
                    for metric in ("pr_auc", "f1"):
                        result = paired_block_bootstrap_difference(
                            merged["target_high_flow"].to_numpy(),
                            merged["probability_model"].to_numpy(),
                            merged["probability_reference"].to_numpy(),
                            float(merged["threshold_model"].iloc[0]),
                            float(merged["threshold_reference"].iloc[0]),
                            pd.DatetimeIndex(merged["target_date"]),
                            metric,
                            int(config["evaluation"]["bootstrap_iterations"]),
                            float(config["evaluation"]["confidence_level"]),
                            int(horizon) * 1_000_000 + comparison_index,
                        )
                        comparison_index += 1
                        rows.append(
                            {
                                "model": model,
                                "reference_model": reference_name,
                                "horizon_days": int(horizon),
                                "seed": int(seed),
                                "reference_seed": int(reference_seed),
                                **result,
                            }
                        )
    return pd.DataFrame(rows)


def _diagnostic_tables(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_rows = []
    annual_rows = []
    magnitude_rows = []
    event_frames = []
    grouping = ["model", "horizon_days", "seed"]
    for keys, group in predictions.groupby(grouping, observed=True):
        model, horizon, seed = keys
        group = group.sort_values("target_date")
        threshold = float(group["threshold"].mean())
        costs = cost_loss_analysis(
            group["target_high_flow"].to_numpy(),
            group["probability"].to_numpy(),
            list(map(float, config["evaluation"]["cost_loss_ratios"])),
        )
        costs.insert(0, "seed", int(seed))
        costs.insert(0, "horizon_days", int(horizon))
        costs.insert(0, "model", str(model))
        cost_rows.extend(costs.to_dict(orient="records"))

        events = event_catalog(
            group["target_high_flow"].to_numpy(),
            group["probability"].to_numpy(),
            threshold,
            pd.DatetimeIndex(group["target_date"]),
            pd.DatetimeIndex(group["issue_date"]),
            group["target_value"].to_numpy(),
            int(config["evaluation"]["event_gap_days"]),
        )
        if not events.empty:
            events.insert(0, "seed", int(seed))
            events.insert(0, "horizon_days", int(horizon))
            events.insert(0, "model", str(model))
            event_frames.append(events)

        for year, annual in group.groupby(
            group["target_date"].dt.year,
            observed=True,
        ):
            metrics = binary_metrics(
                annual["target_high_flow"].to_numpy(),
                annual["probability"].to_numpy(),
                threshold,
                int(config["evaluation"]["reliability_bins"]),
            )
            annual_rows.append(
                {
                    "model": model,
                    "horizon_days": int(horizon),
                    "seed": int(seed),
                    "year": int(year),
                    **metrics,
                }
            )

        positives = group[group["target_high_flow"] == 1].copy()
        if len(positives) >= 3:
            positives["magnitude_group"] = pd.qcut(
                positives["target_value"].rank(method="first"),
                q=3,
                labels=["lower", "middle", "upper"],
            )
            positives["detected"] = positives["probability"] >= threshold
            for magnitude, magnitude_data in positives.groupby(
                "magnitude_group",
                observed=True,
            ):
                magnitude_rows.append(
                    {
                        "model": model,
                        "horizon_days": int(horizon),
                        "seed": int(seed),
                        "magnitude_group": str(magnitude),
                        "positive_days": int(len(magnitude_data)),
                        "detection_rate": float(magnitude_data["detected"].mean()),
                        "target_value_median": float(
                            magnitude_data["target_value"].median()
                        ),
                    }
                )
    return (
        pd.DataFrame(cost_rows),
        pd.DataFrame(annual_rows),
        pd.DataFrame(magnitude_rows),
        (
            pd.concat(event_frames, ignore_index=True)
            if event_frames
            else pd.DataFrame()
        ),
    )


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_experiment(
    config: dict[str, Any],
    frame: pd.DataFrame | None = None,
    output_namespace: str | None = None,
    create_figures: bool = True,
) -> dict[str, Path]:
    if frame is None:
        frame = read_labeled_dataset(config)
    features = prepare_features(frame, config)
    experiment = config["experiment"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_directory = resolve_project_path(config, config["paths"]["models"])
    table_directory = resolve_project_path(config, config["paths"]["tables"])
    figure_directory = resolve_project_path(config, config["paths"]["figures"])
    prediction_directory = resolve_project_path(
        config,
        config["paths"]["predictions"],
    )
    metadata_path = resolve_project_path(
        config,
        config["paths"]["experiment_metadata"],
    )
    if output_namespace:
        namespace = Path(output_namespace)
        if namespace.is_absolute() or ".." in namespace.parts:
            raise ValueError("Output namespace must stay within outputs.")
        model_directory /= namespace
        table_directory /= namespace
        figure_directory /= namespace
        prediction_directory /= namespace
        metadata_path = metadata_path.parent / namespace / metadata_path.name
    for directory in (
        model_directory,
        table_directory,
        figure_directory,
        prediction_directory,
        metadata_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(features.scaler, model_directory / "feature_scaler.joblib")
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    loss_rows: list[dict[str, Any]] = []

    for horizon in map(int, experiment["horizons"]):
        LOGGER.info("Running horizon %d days.", horizon)
        train = create_sequences(
            features,
            "train",
            int(experiment["sequence_length"]),
            horizon,
        )
        validation = create_sequences(
            features,
            "validation",
            int(experiment["sequence_length"]),
            horizon,
        )
        test = create_sequences(
            features,
            "test",
            int(experiment["sequence_length"]),
            horizon,
        )
        calibration_indices, _ = _validation_partitions(validation.labels)

        for baseline in BASELINE_MODELS:
            validation_probabilities = _baseline_probabilities(
                baseline,
                validation,
                train,
                features,
            )
            test_probabilities = _baseline_probabilities(
                baseline,
                test,
                train,
                features,
            )
            metrics, predictions = _evaluate_run(
                baseline,
                horizon,
                0,
                validation_probabilities,
                test_probabilities,
                validation,
                test,
                features,
                config,
                {},
                {"parameter_count": 0, "best_epoch": 0},
            )
            metrics_rows.append(metrics)
            prediction_frames.append(predictions)

        flat_train = flatten_sequences(train)
        flat_validation = flatten_sequences(validation)
        flat_test = flatten_sequences(test)
        for seed in map(int, experiment["seeds"]):
            for model_name in experiment["classical_models"]:
                LOGGER.info(
                    "Training %s, horizon %d, seed %d.",
                    model_name,
                    horizon,
                    seed,
                )
                model = make_classical_model(str(model_name), seed)
                if model_name == "hist_gradient_boosting":
                    sample_weight = compute_sample_weight(
                        class_weight="balanced",
                        y=train.labels,
                    )
                    model.fit(
                        flat_train,
                        train.labels,
                        sample_weight=sample_weight,
                    )
                else:
                    model.fit(flat_train, train.labels)
                raw_validation = model.predict_proba(flat_validation)[:, 1]
                raw_test = model.predict_proba(flat_test)[:, 1]
                calibrated_validation, calibrated_test, calibration = (
                    _calibrate_predictions(
                        raw_validation,
                        raw_test,
                        validation.labels,
                        calibration_indices,
                        str(experiment["calibration"]),
                    )
                )
                metrics, predictions = _evaluate_run(
                    str(model_name),
                    horizon,
                    seed,
                    calibrated_validation,
                    calibrated_test,
                    validation,
                    test,
                    features,
                    config,
                    calibration,
                    {"parameter_count": np.nan, "best_epoch": 0},
                )
                metrics_rows.append(metrics)
                prediction_frames.append(predictions)
                joblib.dump(
                    {
                        "model": model,
                        "calibration": calibration,
                        "feature_names": features.feature_names,
                        "horizon_days": horizon,
                        "seed": seed,
                    },
                    model_directory / f"{model_name}_h{horizon}_seed{seed}.joblib",
                )

            train_indices, early_stopping_indices = _deep_training_partitions(train)
            for model_name in experiment["deep_models"]:
                LOGGER.info(
                    "Training %s, horizon %d, seed %d.",
                    model_name,
                    horizon,
                    seed,
                )
                model = make_deep_model(
                    str(model_name),
                    train.inputs.shape[2],
                    config,
                    seed,
                )
                training = train_deep_model(
                    model,
                    train.inputs[train_indices],
                    train.labels[train_indices],
                    train.inputs[early_stopping_indices],
                    train.labels[early_stopping_indices],
                    config,
                    seed,
                    device,
                )
                selected_learning_rates = training.learning_rates[
                    : training.best_epoch
                ]
                model = make_deep_model(
                    str(model_name),
                    train.inputs.shape[2],
                    config,
                    seed,
                )
                refit_state, refit_losses = refit_deep_model(
                    model,
                    train.inputs,
                    train.labels,
                    config,
                    seed,
                    device,
                    selected_learning_rates,
                )
                model.load_state_dict(refit_state)
                raw_validation = predict_deep_probabilities(
                    model,
                    validation.inputs,
                    device,
                )
                raw_test = predict_deep_probabilities(
                    model,
                    test.inputs,
                    device,
                )
                calibrated_validation, calibrated_test, calibration = (
                    _calibrate_predictions(
                        raw_validation,
                        raw_test,
                        validation.labels,
                        calibration_indices,
                        str(experiment["calibration"]),
                    )
                )
                metrics, predictions = _evaluate_run(
                    str(model_name),
                    horizon,
                    seed,
                    calibrated_validation,
                    calibrated_test,
                    validation,
                    test,
                    features,
                    config,
                    calibration,
                    {
                        "parameter_count": training.parameter_count,
                        "best_epoch": training.best_epoch,
                        "selection_training_samples": int(len(train_indices)),
                        "selection_validation_samples": int(
                            len(early_stopping_indices)
                        ),
                        "refit_training_samples": int(len(train.labels)),
                        "refit_epochs": int(len(refit_losses)),
                    },
                )
                metrics_rows.append(metrics)
                prediction_frames.append(predictions)
                for epoch, (
                    train_loss,
                    validation_loss,
                    validation_pr_auc,
                    learning_rate,
                ) in enumerate(
                    zip(
                        training.train_losses,
                        training.validation_losses,
                        training.validation_pr_auc,
                        training.learning_rates,
                        strict=False,
                    ),
                    start=1,
                ):
                    loss_rows.append(
                        {
                            "model": model_name,
                            "horizon_days": horizon,
                            "seed": seed,
                            "phase": "epoch_selection",
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "validation_loss": validation_loss,
                            "validation_pr_auc": validation_pr_auc,
                            "learning_rate": learning_rate,
                        }
                    )
                for epoch, (refit_loss, learning_rate) in enumerate(
                    zip(refit_losses, selected_learning_rates, strict=True),
                    start=1,
                ):
                    loss_rows.append(
                        {
                            "model": model_name,
                            "horizon_days": horizon,
                            "seed": seed,
                            "phase": "full_training_refit",
                            "epoch": epoch,
                            "train_loss": refit_loss,
                            "validation_loss": np.nan,
                            "validation_pr_auc": np.nan,
                            "learning_rate": learning_rate,
                        }
                    )
                _save_deep_checkpoint(
                    str(model_name),
                    horizon,
                    seed,
                    refit_state,
                    {
                        "selection_best_epoch": training.best_epoch,
                        "refit_epochs": len(refit_losses),
                        "refit_training_samples": len(train.labels),
                        "calibration": calibration,
                    },
                    features.feature_names,
                    model_directory,
                )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(
        ["horizon_days", "model", "seed"]
    )
    predictions_frame = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(["horizon_days", "model", "seed", "target_date"])
    losses_frame = pd.DataFrame(loss_rows)
    comparisons = _paired_comparisons(predictions_frame, config)
    cost_loss, annual_metrics, magnitude_metrics, event_performance = (
        _diagnostic_tables(
            predictions_frame,
            config,
        )
    )
    summary = (
        metrics_frame.groupby(["model", "horizon_days"], observed=True)
        .agg(
            runs=("seed", "count"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            brier_mean=("brier_score", "mean"),
            event_detection_rate_mean=("event_detection_rate", "mean"),
            false_alarm_ratio_mean=("false_alarm_ratio", "mean"),
        )
        .reset_index()
    )

    output_paths = {
        "metrics": table_directory / "metrics_detailed.csv",
        "summary": table_directory / "metrics_summary.csv",
        "comparisons": table_directory / "paired_bootstrap.csv",
        "losses": table_directory / "learning_curves.csv",
        "cost_loss": table_directory / "cost_loss_analysis.csv",
        "annual_metrics": table_directory / "metrics_by_year.csv",
        "magnitude_metrics": table_directory / "metrics_by_magnitude.csv",
        "event_performance": table_directory / "event_performance.csv",
        "predictions": prediction_directory / "test_predictions.parquet",
        "metadata": metadata_path,
    }
    metrics_frame.to_csv(output_paths["metrics"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    comparisons.to_csv(output_paths["comparisons"], index=False)
    losses_frame.to_csv(output_paths["losses"], index=False)
    cost_loss.to_csv(output_paths["cost_loss"], index=False)
    annual_metrics.to_csv(output_paths["annual_metrics"], index=False)
    magnitude_metrics.to_csv(output_paths["magnitude_metrics"], index=False)
    event_performance.to_csv(output_paths["event_performance"], index=False)
    predictions_frame.to_parquet(output_paths["predictions"], index=False)
    predictions_frame.to_csv(
        prediction_directory / "test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    figure_paths = (
        generate_publication_figures(
            metrics_frame,
            predictions_frame,
            figure_directory,
            config,
        )
        if create_figures
        else []
    )

    project_root = Path(config["_project_root"])
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    input_candidates = [
        resolve_project_path(config, config["paths"]["labeled_dataset_parquet"]),
        resolve_project_path(config, config["paths"]["labeled_dataset_csv"]),
        resolve_project_path(config, config["paths"]["label_metadata"]),
        Path(config["_config_path"]),
    ]
    source_candidates = [
        Path(__file__),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("models.py"),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("config.py"),
        project_root / "scripts" / "build_dataset.py",
        project_root / "scripts" / "create_high_flow_labels.py",
        project_root / "scripts" / "train_evaluate.py",
        project_root / "scripts" / "download_era5_accumulations.py",
    ]
    input_fingerprints = [
        _fingerprint_file(path, project_root)
        for path in input_candidates
        if path.is_file()
    ]
    source_fingerprints = [
        _fingerprint_file(path, project_root)
        for path in source_candidates
        if path.is_file()
    ]
    run_payload = json.dumps(
        {
            "configuration": public_config,
            "inputs": input_fingerprints,
            "source": source_fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_fingerprint = hashlib.sha256(run_payload).hexdigest()
    output_fingerprints = {
        name: _fingerprint_file(path, project_root)
        for name, path in output_paths.items()
        if name != "metadata" and path.is_file()
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint_sha256": run_fingerprint,
        "git_commit": _git_commit(project_root),
        "device": str(device),
        "feature_count": len(features.feature_names),
        "feature_names": list(features.feature_names),
        "feature_extreme_thresholds": features.extreme_thresholds,
        "horizons": list(map(int, experiment["horizons"])),
        "seeds": list(map(int, experiment["seeds"])),
        "imbalance_strategy": experiment["imbalance_strategy"],
        "calibration": experiment["calibration"],
        "bootstrap_block": config["evaluation"]["bootstrap_block"],
        "target_source": str(frame["target_source"].iloc[0]),
        "configuration": public_config,
        "input_fingerprints": input_fingerprints,
        "source_fingerprints": source_fingerprints,
        "output_fingerprints": output_fingerprints,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "figure_files": [path.name for path in figure_paths],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Experiment outputs saved under %s.", project_root / "outputs")
    return output_paths
