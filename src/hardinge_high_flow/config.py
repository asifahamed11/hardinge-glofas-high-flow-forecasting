"""Configuration loading and path validation."""

from __future__ import annotations

import copy
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT_ENVIRONMENT_VARIABLE = "HIGHFLOW_PROJECT_ROOT"
REQUIRED_SECTIONS = {
    "period",
    "paths",
    "study_area",
    "datasets",
    "target",
    "splits",
    "features",
    "experiment",
    "evaluation",
    "figures",
}


def _validate_area(area: Any, name: str) -> None:
    if not isinstance(area, list) or len(area) != 4:
        raise ValueError(f"{name} must contain [north, west, south, east].")
    north, west, south, east = map(float, area)
    if not north > south or not east > west:
        raise ValueError(f"{name} has invalid geographic bounds.")


def _validate_relative_path(value: Any, name: str) -> None:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"Configured path must be relative: {name}")
    if ".." in path.parts:
        raise ValueError(f"Configured path cannot escape the project root: {name}")


def _validate_config(config: dict[str, Any]) -> None:
    missing_sections = REQUIRED_SECTIONS - set(config)
    if missing_sections:
        raise ValueError(f"Configuration sections missing: {sorted(missing_sections)}")

    for name, value in config["paths"].items():
        _validate_relative_path(value, f"paths.{name}")
    _validate_relative_path(
        config["target"]["observed"]["path"],
        "target.observed.path",
    )

    start = date.fromisoformat(str(config["period"]["start"]))
    end = date.fromisoformat(str(config["period"]["end"]))
    if start > end:
        raise ValueError("period.start must not exceed period.end.")

    train_end = date.fromisoformat(str(config["splits"]["train_end"]))
    validation_end = date.fromisoformat(str(config["splits"]["validation_end"]))
    test_end = date.fromisoformat(str(config["splits"]["test_end"]))
    if not train_end < validation_end < test_end:
        raise ValueError("Split dates must increase chronologically.")
    if train_end < start:
        raise ValueError("splits.train_end precedes period.start.")
    if test_end > end:
        raise ValueError("splits.test_end exceeds period.end.")

    _validate_area(config["study_area"]["era5_area"], "era5_area")
    _validate_area(config["study_area"]["glofas_area"], "glofas_area")
    for index, domain in enumerate(config["study_area"]["domains"]):
        _validate_area(domain["area"], f"study_area.domains[{index}].area")
    for index, point in enumerate(config["study_area"]["points"]):
        latitude = float(point["latitude"])
        longitude = float(point["longitude"])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"study_area.points[{index}] has invalid coordinates.")

    target_source = config["target"]["source"]
    if target_source not in {"glofas_proxy", "observed"}:
        raise ValueError("Unsupported target.source.")
    quantile = float(config["target"]["quantile"])
    if not 0 < quantile < 1:
        raise ValueError("target.quantile must be between zero and one.")

    maximum_missing_days = int(config["datasets"]["maximum_missing_days"])
    causal_fill_days = int(config["datasets"].get("causal_fill_days", 0))
    if maximum_missing_days < 0 or causal_fill_days < 0:
        raise ValueError("Missing-data limits cannot be negative.")
    if causal_fill_days > maximum_missing_days:
        raise ValueError(
            "datasets.causal_fill_days cannot exceed maximum_missing_days."
        )

    horizons = [int(value) for value in config["experiment"]["horizons"]]
    if not horizons or min(horizons) < 1 or len(horizons) != len(set(horizons)):
        raise ValueError("Experiment horizons must be unique positive integers.")
    if int(config["experiment"]["sequence_length"]) < 2:
        raise ValueError("sequence_length must be at least two.")
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Experiment seeds must be non-empty and unique.")

    strategy = config["experiment"]["imbalance_strategy"]
    if strategy not in {"none", "pos_weight", "focal", "sampler"}:
        raise ValueError("Unsupported imbalance strategy.")
    early_stopping_metric = str(
        config["experiment"].get("early_stopping_metric", "average_precision")
    )
    if early_stopping_metric not in {"average_precision", "loss"}:
        raise ValueError("Unsupported early_stopping_metric.")

    if str(config["evaluation"]["bootstrap_block"]) != "year":
        raise ValueError("Only year-block bootstrap is supported.")
    iterations = int(config["evaluation"]["bootstrap_iterations"])
    confidence = float(config["evaluation"]["confidence_level"])
    if iterations < 20:
        raise ValueError("Use at least 20 bootstrap iterations.")
    if not 0 < confidence < 1:
        raise ValueError("evaluation.confidence_level must be between zero and one.")

    if str(config["figures"]["style"]) != "research":
        raise ValueError("Unsupported research figure style.")
    if int(config["figures"]["png_dpi"]) < 300:
        raise ValueError("PNG resolution must be at least 300 dpi.")
    if int(config["figures"]["tiff_dpi"]) < 300:
        raise ValueError("TIFF resolution must be at least 300 dpi.")
    supported_formats = {"pdf", "eps", "png", "tif", "tiff"}
    formats = [str(value).lower() for value in config["figures"]["formats"]]
    if not formats or len(formats) != len(set(formats)):
        raise ValueError("Figure formats must be non-empty and unique.")
    if not set(formats).issubset(supported_formats):
        raise ValueError("Unsupported research figure format.")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a mapping.")

    config = copy.deepcopy(loaded)
    _validate_config(config)
    configured_root = os.environ.get(ROOT_ENVIRONMENT_VARIABLE)
    project_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else config_path.parent.parent.resolve()
    )
    config["_project_root"] = str(project_root)
    config["_config_path"] = str(config_path)
    return config


def resolve_project_path(
    config: dict[str, Any],
    relative_path: str | Path,
) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("Project paths must be relative.")
    root = Path(config["_project_root"]).resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Project paths cannot escape the project root.") from exc
    return resolved
