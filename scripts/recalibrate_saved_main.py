"""Recalibrate saved primary models without refitting their predictive weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config
from hardinge_high_flow.evaluation import (
    binary_metrics,
    block_bootstrap_intervals,
    calibration_slope_intercept,
    event_metrics,
    fit_sigmoid_calibrator,
    select_threshold,
)
from hardinge_high_flow.experiment import (
    BASELINE_MODELS,
    _baseline_probabilities,
    _configured_validation_partitions,
    read_labeled_dataset,
)
from hardinge_high_flow.features import (
    create_sequences,
    prepare_features,
)
from hardinge_high_flow.models import make_deep_model, predict_deep_probabilities
from hardinge_high_flow.reporting import canonicalize_metric_names

LOGGER = logging.getLogger("recalibrate_saved_main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a better-supported temporal calibration split to saved primary "
            "models, then rebuild primary probabilities and metrics."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override the configured year-block bootstrap iterations.",
    )
    parser.add_argument(
        "--namespace",
        default="",
        help=(
            "Saved-output namespace to recalibrate, for example "
            "rolling_origin/fold_1_test_2009_2012."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_learned_probabilities(
    model_name: str,
    horizon: int,
    seed: int,
    validation_inputs: np.ndarray,
    saved_test_probabilities: np.ndarray,
    previous_calibration: dict[str, float],
    config: dict[str, Any],
    model_directory: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    coefficient = float(previous_calibration["coefficient"])
    intercept = float(previous_calibration["intercept"])
    if np.isclose(coefficient, 0):
        raise ValueError("Cannot invert a zero-slope saved calibrator.")
    clipped = np.clip(saved_test_probabilities, 1e-6, 1 - 1e-6)
    calibrated_logits = np.log(clipped / (1 - clipped))
    raw_logits = (calibrated_logits - intercept) / coefficient
    raw_test = 1 / (1 + np.exp(-raw_logits))

    if model_name in set(map(str, config["experiment"]["classical_models"])):
        artifact = model_directory / f"{model_name}_h{horizon}_seed{seed}.joblib"
        bundle = joblib.load(artifact)
        model = bundle["model"]
        raw_validation = model.predict_proba(
            validation_inputs.reshape(len(validation_inputs), -1)
        )[:, 1]
        return raw_validation, raw_test, bundle, artifact

    artifact = model_directory / f"{model_name}_h{horizon}_seed{seed}.pt"
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    model = make_deep_model(
        model_name,
        validation_inputs.shape[2],
        config,
        seed,
    )
    model.load_state_dict(checkpoint["state_dict"])
    raw_validation = predict_deep_probabilities(
        model,
        validation_inputs,
        torch.device("cpu"),
    )
    return raw_validation, raw_test, checkpoint, artifact


def _save_updated_calibration(
    artifact: Path,
    payload: dict[str, Any],
    calibration: dict[str, float],
) -> None:
    payload["calibration"] = calibration
    if artifact.suffix == ".joblib":
        joblib.dump(payload, artifact)
    else:
        torch.save(payload, artifact)


def _raw_probability_frame(
    model: str,
    horizon: int,
    seed: int,
    split: str,
    sequences,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "horizon_days": horizon,
            "seed": seed,
            "split": split,
            "issue_date": sequences.issue_dates,
            "target_date": sequences.target_dates,
            "target_high_flow": sequences.labels,
            "raw_probability": probabilities,
        }
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    if args.bootstrap_iterations is not None:
        if args.bootstrap_iterations < 20:
            raise ValueError("Use at least 20 bootstrap iterations.")
        config["evaluation"]["bootstrap_iterations"] = args.bootstrap_iterations

    namespace = Path(args.namespace)
    if namespace.is_absolute() or ".." in namespace.parts:
        raise ValueError("Output namespace must stay within outputs.")
    if namespace.parts[:1] == ("ablations",):
        if namespace.name == "with_streamflow":
            config["features"]["include_streamflow"] = True
        elif namespace.name == "no_streamflow":
            config["features"]["include_streamflow"] = False
    frame = read_labeled_dataset(config)
    if namespace.parts and namespace.parts[0] == "rolling_origin":
        from hardinge_high_flow.validation import (
            generate_rolling_origin_folds,
            relabel_for_fold,
        )

        folds = generate_rolling_origin_folds(
            frame.index,
            config["evaluation"]["rolling_origin"],
        )
        matching = [fold for fold in folds if fold.name == namespace.name]
        if len(matching) != 1:
            raise ValueError(f"Unknown rolling-origin namespace: {namespace}")
        fold = matching[0]
        frame, _ = relabel_for_fold(frame, fold, config)
        config["splits"] = {
            "train_end": fold.train_end.date().isoformat(),
            "validation_end": fold.validation_end.date().isoformat(),
            "test_end": fold.test_end.date().isoformat(),
        }
    features = prepare_features(frame, config)
    table_directory = PROJECT_ROOT / config["paths"]["tables"] / namespace
    model_directory = PROJECT_ROOT / config["paths"]["models"] / namespace
    prediction_directory = (
        PROJECT_ROOT / config["paths"]["predictions"] / namespace
    )
    metrics_path = table_directory / "metrics_detailed.csv"
    predictions_path = prediction_directory / "test_predictions.parquet"
    old_metrics = canonicalize_metric_names(pd.read_csv(metrics_path))
    old_predictions = pd.read_parquet(predictions_path)
    old_predictions["target_date"] = pd.to_datetime(old_predictions["target_date"])

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, Any]] = []
    learned_models = {
        *map(str, config["experiment"]["classical_models"]),
        *map(str, config["experiment"]["deep_models"]),
    }
    recalibrated_models = {*BASELINE_MODELS, *learned_models}

    for horizon in map(int, config["experiment"]["horizons"]):
        train = create_sequences(
            features,
            "train",
            int(config["experiment"]["sequence_length"]),
            horizon,
        )
        validation = create_sequences(
            features,
            "validation",
            int(config["experiment"]["sequence_length"]),
            horizon,
        )
        test = create_sequences(
            features,
            "test",
            int(config["experiment"]["sequence_length"]),
            horizon,
        )
        calibration_indices, threshold_indices = _configured_validation_partitions(
            validation.labels,
            config,
        )
        run_keys = old_metrics.loc[
            (old_metrics["horizon_days"] == horizon)
            & old_metrics["model"].isin(recalibrated_models),
            ["model", "seed"],
        ].drop_duplicates()
        for run in run_keys.itertuples(index=False):
            model_name = str(run.model)
            seed = int(run.seed)
            old_row = old_metrics[
                (old_metrics["model"] == model_name)
                & (old_metrics["horizon_days"] == horizon)
                & (old_metrics["seed"] == seed)
            ].iloc[0].to_dict()

            if model_name in BASELINE_MODELS:
                raw_validation = _baseline_probabilities(
                    model_name,
                    validation,
                    train,
                    features,
                )
                raw_test = _baseline_probabilities(
                    model_name,
                    test,
                    train,
                    features,
                )
                calibrated_validation = raw_validation
                calibrated_test = raw_test
                calibration: dict[str, float] = {}
            elif model_name in learned_models:
                stored_for_inversion = old_predictions[
                    (old_predictions["model"] == model_name)
                    & (old_predictions["horizon_days"] == horizon)
                    & (old_predictions["seed"] == seed)
                ].sort_values("target_date")
                previous_calibration = {
                    "coefficient": float(old_row["calibration_coefficient"]),
                    "intercept": float(old_row["calibration_intercept"]),
                }
                raw_validation, raw_test, artifact_payload, artifact = (
                    _raw_learned_probabilities(
                        model_name,
                        horizon,
                        seed,
                        validation.inputs,
                        stored_for_inversion["probability"].to_numpy(dtype=float),
                        previous_calibration,
                        config,
                        model_directory,
                    )
                )
                calibrator = fit_sigmoid_calibrator(
                    raw_validation[calibration_indices],
                    validation.labels[calibration_indices],
                )
                calibration = asdict(calibrator)
                calibrated_validation = calibrator.transform(raw_validation)
                calibrated_test = calibrator.transform(raw_test)
                _save_updated_calibration(
                    artifact,
                    artifact_payload,
                    calibration,
                )
            else:
                raise ValueError(f"Unexpected model in saved metrics: {model_name}")

            threshold, threshold_score = select_threshold(
                validation.labels[threshold_indices],
                calibrated_validation[threshold_indices],
                str(config["evaluation"]["threshold_metric"]),
            )
            updated = binary_metrics(
                test.labels,
                calibrated_test,
                threshold,
                int(config["evaluation"]["reliability_bins"]),
            )
            updated.update(
                event_metrics(
                    test.labels,
                    calibrated_test,
                    threshold,
                    test.target_dates,
                    int(config["evaluation"]["event_gap_days"]),
                )
            )
            updated.update(
                block_bootstrap_intervals(
                    test.labels,
                    calibrated_test,
                    threshold,
                    test.target_dates,
                    int(config["evaluation"]["bootstrap_iterations"]),
                    float(config["evaluation"]["confidence_level"]),
                    seed + horizon * 10_000,
                    int(config["evaluation"]["reliability_bins"]),
                )
            )
            old_row.update(updated)
            old_row.update(
                {
                    "validation_threshold_score": threshold_score,
                    "calibration_coefficient": calibration.get(
                        "coefficient",
                        np.nan,
                    ),
                    "calibration_intercept": calibration.get("intercept", np.nan),
                    "calibration_samples": int(len(calibration_indices)),
                    "calibration_positive_days": int(
                        validation.labels[calibration_indices].sum()
                    ),
                    "threshold_selection_samples": int(len(threshold_indices)),
                    "threshold_selection_positive_days": int(
                        validation.labels[threshold_indices].sum()
                    ),
                }
            )
            metric_rows.append(old_row)

            stored = old_predictions[
                (old_predictions["model"] == model_name)
                & (old_predictions["horizon_days"] == horizon)
                & (old_predictions["seed"] == seed)
            ].sort_values("target_date").copy()
            if not np.array_equal(
                stored["target_date"].to_numpy(dtype="datetime64[ns]"),
                test.target_dates.to_numpy(dtype="datetime64[ns]"),
            ):
                raise ValueError("Saved predictions do not align with reconstructed test dates.")
            stored["probability"] = calibrated_test
            stored["threshold"] = threshold
            stored["prediction"] = (calibrated_test >= threshold).astype(np.int8)
            prediction_frames.append(stored)
            raw_frames.extend(
                [
                    _raw_probability_frame(
                        model_name,
                        horizon,
                        seed,
                        "validation",
                        validation,
                        raw_validation,
                    ),
                    _raw_probability_frame(
                        model_name,
                        horizon,
                        seed,
                        "test",
                        test,
                        raw_test,
                    ),
                ]
            )

            if model_name not in learned_models:
                continue
            for fraction in (0.55, 0.60, 0.65, 0.70):
                boundary = int(len(validation.labels) * fraction)
                calibration_block = np.arange(boundary)
                threshold_block = np.arange(boundary, len(validation.labels))
                sensitivity_calibrator = fit_sigmoid_calibrator(
                    raw_validation[calibration_block],
                    validation.labels[calibration_block],
                )
                sensitivity_validation = sensitivity_calibrator.transform(
                    raw_validation
                )
                sensitivity_test = sensitivity_calibrator.transform(raw_test)
                sensitivity_threshold, _ = select_threshold(
                    validation.labels[threshold_block],
                    sensitivity_validation[threshold_block],
                    str(config["evaluation"]["threshold_metric"]),
                )
                sensitivity_metrics = binary_metrics(
                    test.labels,
                    sensitivity_test,
                    sensitivity_threshold,
                    int(config["evaluation"]["reliability_bins"]),
                )
                sensitivity_calibration = calibration_slope_intercept(
                    test.labels,
                    sensitivity_test,
                    test.target_dates,
                    float(config["evaluation"]["confidence_level"]),
                )
                sensitivity_rows.append(
                    {
                        "model": model_name,
                        "horizon_days": horizon,
                        "seed": seed,
                        "calibration_fraction": fraction,
                        "calibration_samples": int(len(calibration_block)),
                        "calibration_positive_days": int(
                            validation.labels[calibration_block].sum()
                        ),
                        "threshold_selection_samples": int(len(threshold_block)),
                        "threshold_selection_positive_days": int(
                            validation.labels[threshold_block].sum()
                        ),
                        "calibrator_coefficient": sensitivity_calibrator.coefficient,
                        "calibrator_intercept": sensitivity_calibrator.intercept,
                        "decision_threshold": sensitivity_threshold,
                        **sensitivity_metrics,
                        **sensitivity_calibration,
                    }
                )
            LOGGER.info(
                "Recalibrated %s, horizon=%d, seed=%d.",
                model_name,
                horizon,
                seed,
            )

    untouched_metrics = old_metrics[
        ~old_metrics["model"].isin(recalibrated_models)
    ].copy()
    untouched_predictions = old_predictions[
        ~old_predictions["model"].isin(recalibrated_models)
    ].copy()
    for keys, group in untouched_predictions.groupby(
        ["model", "horizon_days", "seed"],
        observed=True,
    ):
        model_name, horizon, seed = keys
        group = group.sort_values("target_date")
        threshold = float(group["threshold"].iloc[0])
        updated_events = event_metrics(
            group["target_high_flow"].to_numpy(),
            group["probability"].to_numpy(),
            threshold,
            pd.DatetimeIndex(group["target_date"]),
            int(config["evaluation"]["event_gap_days"]),
        )
        mask = (
            (untouched_metrics["model"] == model_name)
            & (untouched_metrics["horizon_days"] == horizon)
            & (untouched_metrics["seed"] == seed)
        )
        for name, value in updated_events.items():
            untouched_metrics.loc[mask, name] = value
    metric_rows.extend(untouched_metrics.to_dict(orient="records"))
    prediction_frames.append(untouched_predictions)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["horizon_days", "model", "seed"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["horizon_days", "model", "seed", "target_date"]
    )
    raw_probabilities = pd.concat(raw_frames, ignore_index=True).sort_values(
        ["split", "horizon_days", "model", "seed", "target_date"]
    )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["horizon_days", "model", "seed", "calibration_fraction"]
    )

    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    predictions.to_csv(
        prediction_directory / "test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    raw_path = prediction_directory / "raw_validation_test_probabilities.parquet"
    raw_probabilities.to_parquet(raw_path, index=False)
    sensitivity_path = table_directory / "calibration_split_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    metadata_path = (
        PROJECT_ROOT
        / "outputs"
        / "metadata"
        / namespace
        / "recalibration.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Temporal recalibration of saved primary model weights; predictive "
            "weights were not refitted."
        ),
        "namespace": namespace.as_posix() if namespace.parts else "primary",
        "calibration_fraction": float(
            config["experiment"]["calibration_fraction"]
        ),
        "calibration_minimum_positives": int(
            config["experiment"]["calibration_minimum_positives"]
        ),
        "event_gap_days": int(config["evaluation"]["event_gap_days"]),
        "bootstrap_iterations": int(
            config["evaluation"]["bootstrap_iterations"]
        ),
        "generated": [
            {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": _sha256(path)}
            for path in (
                metrics_path,
                predictions_path,
                raw_path,
                sensitivity_path,
            )
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Recalibration metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
