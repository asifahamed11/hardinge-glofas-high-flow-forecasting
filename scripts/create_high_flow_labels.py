"""Create leakage-safe high-flow labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config, resolve_project_path

LOGGER = logging.getLogger("create_labels")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_master_dataset(config: dict[str, Any]) -> pd.DataFrame:
    parquet_path = resolve_project_path(
        config,
        config["paths"]["master_dataset_parquet"],
    )
    csv_path = resolve_project_path(
        config,
        config["paths"]["master_dataset_csv"],
    )
    if parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.is_file():
        frame = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    else:
        raise FileNotFoundError(
            "Master dataset missing. Run scripts/build_dataset.py first."
        )

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Master dataset dates are not unique.")
    return frame


def read_observed_target(
    config: dict[str, Any],
) -> tuple[pd.Series, str, str]:
    target_config = config["target"]["observed"]
    path = resolve_project_path(config, target_config["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Observed target file not found: {path}. No proxy fallback is performed."
        )

    frame = pd.read_csv(path)
    date_column = str(target_config["date_column"])
    value_column = str(target_config["value_column"])
    missing = {date_column, value_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Observed target columns missing: {sorted(missing)}")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    if frame[date_column].duplicated().any():
        raise ValueError("Observed target dates are not unique.")

    quality_column = target_config.get("quality_flag_column")
    accepted_flags = {
        str(value).strip().casefold()
        for value in target_config.get("accepted_quality_flags", [])
    }
    if quality_column is not None:
        quality_column = str(quality_column)
        if quality_column not in frame.columns:
            raise ValueError(f"Observed quality-flag column missing: {quality_column}")
        if accepted_flags:
            quality_values = (
                frame[quality_column].astype(str).str.strip().str.casefold()
            )
            frame = frame.loc[quality_values.isin(accepted_flags)].copy()
            if frame.empty:
                raise ValueError("No observations passed the quality-flag filter.")

    numeric_values = pd.to_numeric(frame[value_column], errors="raise")
    if numeric_values.isna().any() or not np.isfinite(numeric_values).all():
        raise ValueError("Observed target contains missing or non-finite values.")
    values = pd.Series(
        numeric_values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame[date_column], name="date"),
        name="target_value",
    )
    return (
        values.sort_index().rename("target_value"),
        str(target_config["name"]),
        str(target_config["unit"]),
    )


def target_series(
    frame: pd.DataFrame,
    config: dict[str, Any],
    source: str,
) -> tuple[pd.DataFrame, str, str]:
    if source == "glofas_proxy":
        column = "glofas_discharge_m3s"
        if column not in frame:
            raise ValueError(f"Required target column missing: {column}")
        result = frame.copy()
        result["target_value"] = result[column]
        return result, "GloFAS modelled discharge", "m3_s-1"

    if source == "observed":
        observed, name, unit = read_observed_target(config)
        result = frame.join(observed, how="inner")
        if result.empty:
            raise ValueError("Observed target has no overlapping dates.")
        return result, name, unit

    raise ValueError(f"Unsupported target source: {source}")


def resolve_threshold(
    values: pd.Series,
    config: dict[str, Any],
    source: str,
) -> tuple[float, str]:
    source_config = config["target"][source]
    fixed_threshold = source_config.get("fixed_threshold")
    if fixed_threshold is not None:
        return float(fixed_threshold), "configured_fixed_threshold"

    quantile = float(config["target"]["quantile"])
    threshold = float(values.quantile(quantile, interpolation="linear"))
    return threshold, f"training_quantile_{quantile:.4f}"


def assign_event_ids(labels: pd.Series) -> pd.Series:
    positive = labels.astype(bool)
    starts = positive & ~positive.shift(fill_value=False)
    event_ids = starts.cumsum().where(positive, 0).astype(int)
    return event_ids.rename("high_flow_event_id")


def split_name(
    dates: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.Series:
    conditions = [
        dates <= train_end,
        (dates > train_end) & (dates <= validation_end),
        (dates > validation_end) & (dates <= test_end),
    ]
    labels = np.select(
        conditions,
        ["train", "validation", "test"],
        default="excluded",
    )
    return pd.Series(labels, index=dates, name="split")


def create_labels(
    frame: pd.DataFrame,
    config: dict[str, Any],
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labeled, target_name, target_unit = target_series(frame, config, source)
    expected_dates = pd.date_range(
        config["period"]["start"],
        config["period"]["end"],
        freq="D",
        name="date",
    )
    if not labeled.index.equals(expected_dates):
        missing = expected_dates.difference(labeled.index)
        extra = labeled.index.difference(expected_dates)
        raise ValueError(
            "Target coverage must exactly match the configured period "
            f"(missing={len(missing)}, extra={len(extra)}). "
            "Update period and split dates rather than treating missing "
            "targets as non-events."
        )
    split_config = config["splits"]
    train_end = pd.Timestamp(split_config["train_end"])
    validation_end = pd.Timestamp(split_config["validation_end"])
    test_end = pd.Timestamp(split_config["test_end"])
    if not train_end < validation_end < test_end:
        raise ValueError("Split dates must increase chronologically.")

    labeled["split"] = split_name(
        labeled.index,
        train_end,
        validation_end,
        test_end,
    )
    labeled = labeled[labeled["split"] != "excluded"].copy()
    training_values = labeled.loc[
        labeled["split"] == "train",
        "target_value",
    ].dropna()
    if training_values.empty:
        raise ValueError("Training target values are empty.")

    threshold, threshold_method = resolve_threshold(
        training_values,
        config,
        source,
    )
    labeled["target_high_flow"] = (labeled["target_value"] >= threshold).astype(np.int8)
    labeled["high_flow_event_id"] = assign_event_ids(labeled["target_high_flow"])
    labeled["target_source"] = source

    counts = (
        labeled.groupby("split", observed=True)["target_high_flow"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "days",
                "sum": "positive_days",
                "mean": "positive_fraction",
            }
        )
    )
    required_splits = {"train", "validation", "test"}
    if set(counts.index) != required_splits:
        raise ValueError("All chronological splits must be non-empty.")
    if (counts["positive_days"] == 0).any():
        raise ValueError("Every split must contain positive target days.")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_source": source,
        "target_name": target_name,
        "target_unit": target_unit,
        "label_name": "target_high_flow",
        "threshold": threshold,
        "threshold_method": threshold_method,
        "threshold_fit_period": {
            "start": training_values.index.min().date().isoformat(),
            "end": training_values.index.max().date().isoformat(),
            "observations": int(len(training_values)),
        },
        "splits": {
            split: {
                key: (int(value) if key in {"days", "positive_days"} else float(value))
                for key, value in row.items()
            }
            for split, row in counts.to_dict(orient="index").items()
        },
        "interpretation": (
            "Proxy high-flow exceedance in modelled GloFAS discharge"
            if source == "glofas_proxy"
            else "High-flow exceedance in independently observed data"
        ),
    }
    return labeled, metadata


def save_labeled_dataset(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> None:
    csv_path = resolve_project_path(
        config,
        config["paths"]["labeled_dataset_csv"],
    )
    parquet_path = resolve_project_path(
        config,
        config["paths"]["labeled_dataset_parquet"],
    )
    metadata_path = resolve_project_path(
        config,
        config["paths"]["label_metadata"],
    )
    for path in (csv_path, parquet_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(csv_path, index=True, date_format="%Y-%m-%d")
    frame.to_parquet(parquet_path, index=True)
    metadata["outputs"] = {
        "csv_sha256": sha256_file(csv_path),
        "parquet_sha256": sha256_file(parquet_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved leakage-safe labels to %s", csv_path)


def add_master_input_fingerprints(
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> None:
    project_root = Path(config["_project_root"]).resolve()
    candidates = {
        "master_dataset_parquet": resolve_project_path(
            config, config["paths"]["master_dataset_parquet"]
        ),
        "master_dataset_csv": resolve_project_path(
            config, config["paths"]["master_dataset_csv"]
        ),
        "master_dataset_metadata": resolve_project_path(
            config, config["paths"]["master_dataset_metadata"]
        ),
    }
    metadata["inputs"] = {
        name: {
            "path": path.resolve().relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in candidates.items()
        if path.is_file()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create training-only high-flow thresholds and labels."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--target-source",
        choices=("glofas_proxy", "observed"),
        help="Override the configured target source.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    source = args.target_source or str(config["target"]["source"])
    frame = read_master_dataset(config)
    labeled, metadata = create_labels(frame, config, source)
    add_master_input_fingerprints(metadata, config)
    save_labeled_dataset(labeled, metadata, config)


if __name__ == "__main__":
    main()
