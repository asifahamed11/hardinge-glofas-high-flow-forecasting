"""Regenerate corrected tables, interpretation, and figures without retraining."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config
from hardinge_high_flow.evaluation import SigmoidCalibrator
from hardinge_high_flow.experiment import read_labeled_dataset
from hardinge_high_flow.features import create_sequences, prepare_features
from hardinge_high_flow.interpretability import grouped_permutation_importance
from hardinge_high_flow.plotting import (
    configure_figure_style,
    generate_publication_figures,
    plot_permutation_importance,
)
from hardinge_high_flow.reporting import (
    canonicalize_metric_names,
    diagnostic_tables,
    summarize_metrics,
)

LOGGER = logging.getLogger("regenerate_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild corrected metrics, diagnostics, grouped permutation "
            "importance, and manuscript-facing figures from saved artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=3,
        help="Grouped permutations per feature, fitted seed, and horizon.",
    )
    parser.add_argument(
        "--skip-importance",
        action="store_true",
        help="Skip grouped Random Forest permutation importance.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_threshold(config: dict[str, Any]) -> float:
    metadata_path = PROJECT_ROOT / config["paths"]["label_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata["threshold"])


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _regenerate_namespace(
    prediction_path: Path,
    config: dict[str, Any],
    target_threshold: float,
) -> list[Path]:
    prediction_root = PROJECT_ROOT / config["paths"]["predictions"]
    relative = prediction_path.parent.relative_to(prediction_root)
    table_directory = PROJECT_ROOT / config["paths"]["tables"] / relative
    figure_directory = PROJECT_ROOT / config["paths"]["figures"] / relative
    metrics_path = table_directory / "metrics_detailed.csv"
    if not metrics_path.is_file():
        LOGGER.warning("No metrics found for %s; skipping.", relative or "main")
        return []

    predictions = pd.read_parquet(prediction_path)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions["issue_date"] = pd.to_datetime(predictions["issue_date"])
    if "target_threshold" not in predictions:
        predictions["target_threshold"] = target_threshold
    metrics = canonicalize_metric_names(pd.read_csv(metrics_path))
    written = [metrics_path, table_directory / "metrics_summary.csv"]
    _write_table(metrics, metrics_path)
    _write_table(summarize_metrics(metrics), written[-1])

    cost_loss, annual, magnitude, events = diagnostic_tables(predictions, config)
    diagnostic_paths = [
        table_directory / "cost_loss_analysis.csv",
        table_directory / "metrics_by_year.csv",
        table_directory / "metrics_by_magnitude.csv",
        table_directory / "event_performance.csv",
    ]
    for frame, path in zip(
        (cost_loss, annual, magnitude, events),
        diagnostic_paths,
        strict=True,
    ):
        _write_table(frame, path)
    written.extend(diagnostic_paths)

    for name in ("paired_bootstrap.csv", "learning_curves.csv"):
        path = table_directory / name
        if path.is_file():
            _write_table(canonicalize_metric_names(pd.read_csv(path)), path)
            written.append(path)

    create_figures = not relative.parts or relative.parts[0] == "ablations"
    if not create_figures:
        return written
    return [
        *written,
        *generate_publication_figures(
            metrics,
            predictions,
            figure_directory,
            config,
        ),
    ]


def _regenerate_rolling_summary(config: dict[str, Any]) -> list[Path]:
    table_directory = PROJECT_ROOT / config["paths"]["tables"]
    metrics_path = table_directory / "rolling_origin_metrics.csv"
    if not metrics_path.is_file():
        return []
    metrics = canonicalize_metric_names(pd.read_csv(metrics_path))
    _write_table(metrics, metrics_path)
    summary = summarize_metrics(metrics)
    fold_counts = (
        metrics.groupby(["model", "horizon_days"], observed=True)["fold"]
        .nunique()
        .rename("folds")
        .reset_index()
    )
    summary = fold_counts.merge(
        summary,
        on=["model", "horizon_days"],
        validate="one_to_one",
    )
    summary_path = table_directory / "rolling_origin_summary.csv"
    _write_table(summary, summary_path)
    return [metrics_path, summary_path]


def _calibrated_predictor(bundle: dict[str, Any]):
    model = bundle["model"]
    calibration = bundle.get("calibration", {})
    calibrator = (
        SigmoidCalibrator(
            coefficient=float(calibration["coefficient"]),
            intercept=float(calibration["intercept"]),
        )
        if {"coefficient", "intercept"}.issubset(calibration)
        else None
    )

    def predict(inputs: np.ndarray) -> np.ndarray:
        flattened = inputs.reshape(len(inputs), -1)
        probabilities = model.predict_proba(flattened)[:, 1]
        return calibrator.transform(probabilities) if calibrator else probabilities

    return predict


def _random_forest_importance(
    config: dict[str, Any],
    frame: pd.DataFrame,
    model_directory: Path,
    repeats: int,
    label: str,
) -> pd.DataFrame:
    features = prepare_features(frame, config)
    rows = []
    for horizon in map(int, config["experiment"]["horizons"]):
        sequences = create_sequences(
            features,
            "test",
            int(config["experiment"]["sequence_length"]),
            horizon,
        )
        for fitted_seed in map(int, config["experiment"]["seeds"]):
            path = model_directory / f"random_forest_h{horizon}_seed{fitted_seed}.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Saved Random Forest artifact missing: {path}")
            bundle = joblib.load(path)
            importance = grouped_permutation_importance(
                _calibrated_predictor(bundle),
                sequences.inputs,
                sequences.labels,
                features.feature_names,
                repeats=repeats,
                seed=horizon * 1_000_000 + fitted_seed,
            )
            importance.insert(0, "fitted_seed", fitted_seed)
            importance.insert(0, "horizon_days", horizon)
            importance.insert(0, "model", "random_forest")
            importance.insert(0, "feature_configuration", label)
            rows.append(importance)
            LOGGER.info(
                "Permutation importance complete: %s, horizon=%d, seed=%d",
                label,
                horizon,
                fitted_seed,
            )
    return pd.concat(rows, ignore_index=True)


def _regenerate_importance(
    config: dict[str, Any],
    repeats: int,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    if repeats < 1:
        raise ValueError("--permutation-repeats must be positive.")
    frame = read_labeled_dataset(config)
    table_root = PROJECT_ROOT / config["paths"]["tables"]
    model_root = PROJECT_ROOT / config["paths"]["models"]
    importance_frames = {}

    main = _random_forest_importance(
        config,
        frame,
        model_root,
        repeats,
        "Without current discharge",
    )
    main_path = table_root / "permutation_importance.csv"
    _write_table(main, main_path)
    importance_frames["Without current discharge"] = main
    table_paths = [main_path]

    streamflow_models = model_root / "ablations" / "with_streamflow"
    if streamflow_models.is_dir():
        streamflow_config = copy.deepcopy(config)
        streamflow_config["features"]["include_streamflow"] = True
        with_streamflow = _random_forest_importance(
            streamflow_config,
            frame,
            streamflow_models,
            repeats,
            "With current discharge",
        )
        streamflow_path = (
            table_root / "ablations" / "with_streamflow" / "permutation_importance.csv"
        )
        _write_table(with_streamflow, streamflow_path)
        table_paths.append(streamflow_path)
        importance_frames["With current discharge"] = with_streamflow

    configure_figure_style(config)
    figure_paths = plot_permutation_importance(
        importance_frames,
        PROJECT_ROOT / config["paths"]["figures"],
        config,
    )
    return importance_frames, [*table_paths, *figure_paths]


def _write_postprocessing_metadata(
    config: dict[str, Any],
    generated: list[Path],
    repeats: int | None,
) -> Path:
    metadata_path = PROJECT_ROOT / "outputs" / "metadata" / "postprocessing.json"
    unique_paths = sorted({path.resolve() for path in generated if path.is_file()})
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Correct Average Precision naming, rebuild threshold-sensitive "
            "diagnostics, add grouped permutation importance, and regenerate figures."
        ),
        "source_predictions": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _file_sha256(path),
            }
            for path in sorted(
                (PROJECT_ROOT / config["paths"]["predictions"]).rglob(
                    "test_predictions.parquet"
                )
            )
        ],
        "permutation_repeats": repeats,
        "scientific_interpretation": {
            "target": "GloFAS-modelled high-flow exceedance proxy",
            "study_type": "retrospective multi-horizon prediction",
            "not_supported": [
                "gauge-validated flood occurrence",
                "observed flood prediction skill",
                "operational warning-system performance",
            ],
            "spatial_limitation": (
                "The ERA5-Land predictors are cosine-latitude-weighted means "
                "over 20.7-26.6 N, 88.0-92.6 E, not full-upstream, "
                "catchment-weighted aggregates."
            ),
        },
        "postprocessing_source": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _file_sha256(path),
            }
            for path in (
                PROJECT_ROOT / "scripts" / "regenerate_analysis.py",
                PROJECT_ROOT / "src" / "hardinge_high_flow" / "reporting.py",
                PROJECT_ROOT / "src" / "hardinge_high_flow" / "plotting.py",
                PROJECT_ROOT / "src" / "hardinge_high_flow" / "interpretability.py",
            )
        ],
        "generated_files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _file_sha256(path),
            }
            for path in unique_paths
        ],
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = parse_args()
    config = load_config(args.config)
    threshold = _target_threshold(config)
    prediction_root = PROJECT_ROOT / config["paths"]["predictions"]
    prediction_paths = sorted(prediction_root.rglob("test_predictions.parquet"))
    if not prediction_paths:
        raise FileNotFoundError("No saved prediction artifacts were found.")

    generated = []
    for path in prediction_paths:
        generated.extend(_regenerate_namespace(path, config, threshold))
        LOGGER.info("Corrected namespace: %s", path.parent.relative_to(prediction_root))
    generated.extend(_regenerate_rolling_summary(config))

    repeats = None
    if not args.skip_importance:
        repeats = int(args.permutation_repeats)
        _, importance_outputs = _regenerate_importance(config, repeats)
        generated.extend(importance_outputs)
    else:
        table_root = PROJECT_ROOT / config["paths"]["tables"]
        saved_importance_paths = [
            table_root / "permutation_importance.csv",
            table_root
            / "ablations"
            / "with_streamflow"
            / "permutation_importance.csv",
        ]
        labels = ("Without current discharge", "With current discharge")
        saved_frames = {
            label: pd.read_csv(path)
            for label, path in zip(labels, saved_importance_paths, strict=True)
            if path.is_file()
        }
        if saved_frames:
            repeats = max(int(frame["repeat"].max()) for frame in saved_frames.values())
            configure_figure_style(config)
            generated.extend(path for path in saved_importance_paths if path.is_file())
            generated.extend(
                plot_permutation_importance(
                    saved_frames,
                    PROJECT_ROOT / config["paths"]["figures"],
                    config,
                )
            )
    metadata_path = _write_postprocessing_metadata(config, generated, repeats)
    LOGGER.info("Postprocessing metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
