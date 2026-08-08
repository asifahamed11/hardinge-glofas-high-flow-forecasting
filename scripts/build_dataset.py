"""Build the validated daily hydro-meteorological dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config, resolve_project_path

LOGGER = logging.getLogger("build_dataset")
TIME_NAMES = ("valid_time", "time", "date")
LATITUDE_NAMES = ("latitude", "lat")
LONGITUDE_NAMES = ("longitude", "lon")
GLOFAS_DAILY_TIME_MAPPING = (
    "End-of-24-hour averaging timestamp shifted to the preceding UTC day"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


@contextmanager
def materialized_netcdf(path: Path) -> Iterator[Path]:
    if not zipfile.is_zipfile(path):
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="highflow_netcdf_") as temp_name:
        temporary_directory = Path(temp_name)
        with zipfile.ZipFile(path) as archive:
            candidates = [
                member
                for member in archive.infolist()
                if not member.is_dir()
                and Path(member.filename).suffix.lower() in {".nc", ".netcdf"}
            ]
            if len(candidates) != 1:
                raise ValueError(f"{path} must contain exactly one NetCDF file.")
            member = candidates[0]
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member in {path}.")
            destination = temporary_directory / member_path.name
            with archive.open(member) as source, destination.open("wb") as target:
                while block := source.read(1_048_576):
                    target.write(block)
        yield destination


def find_name(candidates: Sequence[str], available: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(f"Expected one of {tuple(candidates)}; found {tuple(available)}")


def find_variable(dataset: xr.Dataset, aliases: Sequence[str]) -> str:
    return find_name(aliases, list(dataset.data_vars))


def extract_spatial_mean(
    dataset: xr.Dataset,
    aliases: Sequence[str],
    output_name: str,
) -> pd.Series:
    variable = find_variable(dataset, aliases)
    data = dataset[variable].squeeze(drop=True)
    time_name = find_name(TIME_NAMES, list(data.coords))
    time_dimension = dataset[time_name].dims[0]
    reduce_dimensions = [
        dimension for dimension in data.dims if dimension != time_dimension
    ]
    if reduce_dimensions:
        latitude_name = next(
            (
                name
                for name in LATITUDE_NAMES
                if name in data.coords and name in reduce_dimensions
            ),
            None,
        )
        if latitude_name is None:
            data = data.mean(dim=reduce_dimensions, skipna=True)
        else:
            latitude_weights = np.cos(np.deg2rad(data[latitude_name].astype(float)))
            data = data.weighted(latitude_weights).mean(
                dim=reduce_dimensions,
                skipna=True,
            )
    dates = pd.to_datetime(dataset[time_name].values)
    values = np.asarray(data.values, dtype=float).reshape(-1)
    if len(dates) != len(values):
        raise ValueError(f"Unexpected dimensions for {variable}.")
    return pd.Series(values, index=dates, name=output_name)


def extract_nearest_point(
    dataset: xr.Dataset,
    aliases: Sequence[str],
    output_name: str,
    latitude: float,
    longitude: float,
) -> tuple[pd.Series, tuple[float, float]]:
    variable = find_variable(dataset, aliases)
    latitude_name = find_name(LATITUDE_NAMES, list(dataset.coords))
    longitude_name = find_name(LONGITUDE_NAMES, list(dataset.coords))
    selected = dataset[variable].sel(
        {
            latitude_name: latitude,
            longitude_name: longitude,
        },
        method="nearest",
    )
    selected = selected.squeeze(drop=True)
    time_name = find_name(TIME_NAMES, list(selected.coords))
    time_dimension = dataset[time_name].dims[0]
    remaining = [
        dimension for dimension in selected.dims if dimension != time_dimension
    ]
    if remaining:
        raise ValueError(f"Unexpected non-time dimensions for {variable}: {remaining}")

    dates = pd.to_datetime(dataset[time_name].values)
    values = np.asarray(selected.values, dtype=float).reshape(-1)
    selected_latitude = float(selected[latitude_name].values)
    selected_longitude = float(selected[longitude_name].values)
    return (
        pd.Series(values, index=dates, name=output_name),
        (selected_latitude, selected_longitude),
    )


def normalize_temperature(
    series: pd.Series,
    units: str | None,
) -> pd.Series:
    unit_text = (units or "").lower()
    if "kelvin" in unit_text or unit_text.strip() == "k":
        return series - 273.15
    if series.dropna().median() > 150:
        return series - 273.15
    return series


def normalize_water_depth(
    series: pd.Series,
    units: str | None,
) -> pd.Series:
    unit_text = (units or "").lower().strip()
    if unit_text in {"m", "metre", "meter", "m of water equivalent"}:
        return series * 1_000.0
    return series


def read_gridded_series(
    files: Sequence[Path],
    aliases: Sequence[str],
    output_name: str,
    conversion: str | None = None,
    required_dataset_attribute: tuple[str, str] | None = None,
) -> pd.Series:
    if not files:
        raise FileNotFoundError(f"No files found for {output_name}.")

    pieces: list[pd.Series] = []
    for path in files:
        with (
            materialized_netcdf(path) as netcdf_path,
            xr.open_dataset(netcdf_path) as dataset,
        ):
            if required_dataset_attribute is not None:
                attribute_name, expected_value = required_dataset_attribute
                actual_value = dataset.attrs.get(attribute_name)
                if actual_value != expected_value:
                    raise ValueError(
                        f"{path} was not produced by the verified ERA5-Land "
                        "daily-accumulation method. Re-run the accumulation "
                        "downloader before rebuilding the dataset."
                    )
            variable = find_variable(dataset, aliases)
            units = dataset[variable].attrs.get("units")
            series = extract_spatial_mean(dataset, aliases, output_name)
            if conversion == "temperature":
                series = normalize_temperature(series, units)
            elif conversion == "water_depth":
                series = normalize_water_depth(series, units)
            pieces.append(series)

    combined = pd.concat(pieces).sort_index()
    combined.index = combined.index.normalize()
    if combined.index.has_duplicates:
        duplicate_spread = combined.groupby(level=0).agg(
            lambda values: float(np.nanmax(values) - np.nanmin(values))
        )
        if (duplicate_spread.fillna(0) > 1e-8).any():
            raise ValueError(f"Conflicting duplicate dates for {output_name}.")
        combined = combined.groupby(level=0).mean()
    return combined.rename(output_name)


def read_glofas_series(
    files: Sequence[Path],
    latitude: float,
    longitude: float,
) -> tuple[pd.Series, tuple[float, float]]:
    if not files:
        raise FileNotFoundError("No GloFAS files found.")

    aliases = ("dis24", "river_discharge_in_the_last_24_hours", "avg_dis")
    pieces: list[pd.Series] = []
    selected_coordinates: tuple[float, float] | None = None
    for path in files:
        with (
            materialized_netcdf(path) as netcdf_path,
            xr.open_dataset(netcdf_path) as dataset,
        ):
            series, coordinates = extract_nearest_point(
                dataset,
                aliases,
                "glofas_discharge_m3s",
                latitude,
                longitude,
            )
            if selected_coordinates is None:
                selected_coordinates = coordinates
            elif not np.allclose(
                selected_coordinates,
                coordinates,
                atol=1e-8,
            ):
                raise ValueError("GloFAS grid coordinates changed.")
            pieces.append(series)

    combined = pd.concat(pieces).sort_index()
    # GloFAS daily discharge is a mean over the preceding 24-hour model step.
    # Its timestamp marks the end of that averaging period, so map it to the
    # calendar day represented by the interval before joining daily predictors.
    combined.index = combined.index.normalize() - pd.Timedelta(days=1)
    if combined.index.has_duplicates:
        duplicate_spread = combined.groupby(level=0).agg(
            lambda values: float(np.nanmax(values) - np.nanmin(values))
        )
        if (duplicate_spread.fillna(0) > 1e-8).any():
            raise ValueError("Conflicting duplicate GloFAS dates.")
        combined = combined.groupby(level=0).mean()
    if (combined.dropna() < 0).any():
        raise ValueError("GloFAS discharge contains negative values.")
    if selected_coordinates is None:
        raise RuntimeError("No GloFAS coordinates were selected.")
    return combined.rename("glofas_discharge_m3s"), selected_coordinates


def nasa_power_url(config: dict[str, Any]) -> str:
    point = config["study_area"]["hardinge_bridge"]
    parameters = ",".join(config["datasets"]["nasa_power"]["parameters"])
    start = str(config["period"]["start"]).replace("-", "")
    end = str(config["period"]["end"]).replace("-", "")
    return (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={parameters}"
        "&community=AG"
        f"&longitude={float(point['longitude']):.5f}"
        f"&latitude={float(point['latitude']):.5f}"
        f"&start={start}&end={end}"
        "&format=CSV&time-standard=UTC"
    )


def download_nasa_power(
    destination: Path,
    config: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    response = requests.get(nasa_power_url(config), timeout=180)
    response.raise_for_status()
    temporary.write_bytes(response.content)
    if temporary.stat().st_size < 1_000:
        temporary.unlink(missing_ok=True)
        raise ValueError("NASA POWER download was unexpectedly small.")
    temporary.replace(destination)


def read_nasa_power(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as file_handle:
        header_index = next(
            (index for index, line in enumerate(file_handle) if "-END HEADER-" in line),
            None,
        )
    if header_index is None:
        raise ValueError(f"NASA POWER header marker missing in {path}.")

    frame = pd.read_csv(path, skiprows=header_index + 1)
    frame.columns = frame.columns.str.strip()
    required = {"YEAR", "DOY", "T2M", "PRECTOTCORR", "RH2M"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"NASA POWER columns missing: {sorted(missing)}")
    frame = frame.replace(-999.0, np.nan)
    frame["date"] = pd.to_datetime(
        frame["YEAR"].astype(str) + frame["DOY"].astype(int).astype(str).str.zfill(3),
        format="%Y%j",
    )
    frame = frame.rename(
        columns={
            "T2M": "nasa_temperature_c",
            "PRECTOTCORR": "nasa_precipitation_mm",
            "RH2M": "nasa_relative_humidity_percent",
        }
    )
    columns = [
        "date",
        "nasa_temperature_c",
        "nasa_precipitation_mm",
        "nasa_relative_humidity_percent",
    ]
    return frame[columns].set_index("date").sort_index()


def collect_files(directory: Path) -> list[Path]:
    files = [*directory.glob("*.nc"), *directory.glob("*.zip")]
    return sorted(path for path in files if path.is_file())


def validate_dataset(
    frame: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    maximum_missing_days: int,
    causal_fill_days: int = 0,
) -> pd.DataFrame:
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Dataset dates are not sorted.")
    if frame.index.has_duplicates:
        raise ValueError("Dataset dates are not unique.")

    aligned = frame.reindex(expected_index)
    missing_before_fill = aligned.isna().sum()
    LOGGER.info("Missing values before causal fill: %s", missing_before_fill.to_dict())
    if int(missing_before_fill.max()) > maximum_missing_days:
        raise ValueError(
            f"Missing-data limit exceeded: {missing_before_fill.to_dict()}"
        )

    if causal_fill_days:
        # Forward fill is causal: every replacement uses only information that
        # was already available before the filled date.  Backward fill and
        # bidirectional interpolation are intentionally forbidden.
        aligned = aligned.ffill(limit=causal_fill_days)
    missing_after_fill = aligned.isna().sum()
    if int(missing_after_fill.max()) > 0:
        raise ValueError(
            "Unresolved missing values remain after causal fill: "
            f"{missing_after_fill.to_dict()}"
        )

    complete = aligned.copy()
    if not complete.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Complete dataset contains date gaps.")
    if (complete["era5_precipitation_mm"] < 0).any():
        raise ValueError("ERA5 precipitation contains negative values.")
    if (complete["nasa_precipitation_mm"] < 0).any():
        raise ValueError("NASA precipitation contains negative values.")
    return complete


def build_dataset(
    config: dict[str, Any],
    offline: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_directory = resolve_project_path(
        config,
        config["paths"]["raw_era5_daily"],
    )
    accumulation_directory = resolve_project_path(
        config,
        config["paths"]["raw_era5_accumulations"],
    )
    glofas_directory = resolve_project_path(
        config,
        config["paths"]["raw_glofas"],
    )
    nasa_path = resolve_project_path(
        config,
        config["paths"]["raw_nasa_power"],
    )
    if not nasa_path.exists():
        if offline:
            raise FileNotFoundError(
                f"NASA POWER input missing in offline mode: {nasa_path}"
            )
        LOGGER.info("Downloading NASA POWER data.")
        download_nasa_power(nasa_path, config)

    daily_files = collect_files(daily_directory)
    accumulation_files = collect_files(accumulation_directory)
    glofas_files = collect_files(glofas_directory)
    station = config["study_area"]["hardinge_bridge"]

    series = [
        read_gridded_series(
            daily_files,
            ("t2m", "2m_temperature"),
            "era5_temperature_c",
            conversion="temperature",
        ),
        read_gridded_series(
            daily_files,
            ("swvl1", "volumetric_soil_water_layer_1"),
            "era5_soil_moisture_m3m3",
        ),
        read_gridded_series(
            accumulation_files,
            ("tp", "total_precipitation"),
            "era5_precipitation_mm",
            conversion="water_depth",
            required_dataset_attribute=(
                "daily_accumulation_method",
                "ERA5-Land 00 UTC accumulation shifted to the preceding UTC day",
            ),
        ),
        read_gridded_series(
            accumulation_files,
            ("ro", "runoff"),
            "era5_runoff_mm",
            conversion="water_depth",
            required_dataset_attribute=(
                "daily_accumulation_method",
                "ERA5-Land 00 UTC accumulation shifted to the preceding UTC day",
            ),
        ),
    ]
    glofas, selected_coordinates = read_glofas_series(
        glofas_files,
        float(station["latitude"]),
        float(station["longitude"]),
    )
    series.append(glofas)
    frame = pd.concat(series, axis=1)
    frame = frame.join(read_nasa_power(nasa_path), how="outer")

    expected_index = pd.date_range(
        start=pd.Timestamp(config["period"]["start"]),
        end=pd.Timestamp(config["period"]["end"]),
        freq="D",
        name="date",
    )
    frame = validate_dataset(
        frame,
        expected_index,
        int(config["datasets"]["maximum_missing_days"]),
        int(config["datasets"].get("causal_fill_days", 0)),
    )
    frame.index.name = "date"

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Daily hydro-meteorological predictors at Hardinge Bridge",
        "target_status": "No labels are created by this script",
        "missing_data_policy": {
            "maximum_missing_days": int(
                config["datasets"]["maximum_missing_days"]
            ),
            "causal_fill_days": int(config["datasets"].get("causal_fill_days", 0)),
            "method": "past-only forward fill; no future-value interpolation",
        },
        "sources": {
            "era5_land": {
                "dataset": "ERA5-Land post-processed daily statistics",
                "doi": "10.24381/cds.e9c9c792",
                "accumulation_method": (
                    "ERA5-Land 00 UTC accumulation shifted to the preceding UTC day"
                ),
            },
            "glofas": {
                "dataset": "CEMS GloFAS historical river discharge",
                "doi": "10.24381/cds.a4fdd6b9",
                "system_version": config["datasets"]["glofas"]["system_version"],
                "daily_time_mapping": GLOFAS_DAILY_TIME_MAPPING,
            },
            "nasa_power": {
                "dataset": "NASA POWER daily point API",
                "endpoint": "https://power.larc.nasa.gov/api/temporal/daily/point",
                "time_standard": "UTC",
            },
        },
        "date_start": frame.index.min().date().isoformat(),
        "date_end": frame.index.max().date().isoformat(),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "units": {
            "era5_temperature_c": "degree_Celsius",
            "era5_soil_moisture_m3m3": "m3_m-3",
            "era5_precipitation_mm": "mm_day-1",
            "era5_runoff_mm": "mm_day-1",
            "glofas_discharge_m3s": "m3_s-1",
            "nasa_temperature_c": "degree_Celsius",
            "nasa_precipitation_mm": "mm_day-1",
            "nasa_relative_humidity_percent": "percent",
        },
        "spatial_processing": {
            "era5": "Cosine-latitude-weighted grid mean over configured domain",
            "glofas": "Nearest grid cell to configured station",
            "glofas_selected_latitude": selected_coordinates[0],
            "glofas_selected_longitude": selected_coordinates[1],
            "nasa_power": "Configured point",
        },
        "input_counts": {
            "era5_daily": len(daily_files),
            "era5_accumulations": len(accumulation_files),
            "glofas": len(glofas_files),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
        },
    }
    return frame, metadata


def save_dataset(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> None:
    csv_path = resolve_project_path(
        config,
        config["paths"]["master_dataset_csv"],
    )
    parquet_path = resolve_project_path(
        config,
        config["paths"]["master_dataset_parquet"],
    )
    metadata_path = resolve_project_path(
        config,
        config["paths"]["master_dataset_metadata"],
    )
    for path in (csv_path, parquet_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(csv_path, index=True, date_format="%Y-%m-%d")
    frame.to_parquet(parquet_path, index=True)
    project_root = Path(config["_project_root"])
    config_path = Path(config["_config_path"])
    metadata["configuration"] = {
        "path": relative_path(config_path, project_root),
        "sha256": sha256_file(config_path),
    }
    metadata["outputs"] = {
        "csv": relative_path(csv_path, project_root),
        "parquet": relative_path(parquet_path, project_root),
        "csv_sha256": sha256_file(csv_path),
        "parquet_sha256": sha256_file(parquet_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved %d rows to %s", len(frame), csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the validated daily master dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading missing NASA POWER data.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    frame, metadata = build_dataset(config, args.offline)
    save_dataset(frame, metadata, config)


if __name__ == "__main__":
    main()
