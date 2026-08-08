"""Download ERA5-Land daily precipitation and runoff sums."""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config, resolve_project_path
from hardinge_high_flow.download_utils import (
    DownloadResult,
    cds_retrieve,
    iter_months,
    load_cds_keys,
    unpack_zip_to_netcdf,
    valid_netcdf,
)

LOGGER = logging.getLogger("era5_accumulations")

DATASET_NAME = "reanalysis-era5-land"
REQUIRED_VARIABLES = {"tp", "ro"}
DAILY_ACCUMULATION_METHOD = (
    "ERA5-Land 00 UTC accumulation shifted to the preceding UTC day"
)


def valid_daily_accumulation_file(path: Path) -> bool:
    if not valid_netcdf(path, REQUIRED_VARIABLES):
        return False
    try:
        from hardinge_high_flow.download_utils import _HDF5_LOCK

        with _HDF5_LOCK, xr.open_dataset(path, decode_times=True) as dataset:
            normalized = _rename_time_coordinate(dataset)
            dates = pd.DatetimeIndex(normalized["time"].values)
            return (
                dataset.attrs.get("daily_accumulation_method")
                == DAILY_ACCUMULATION_METHOD
                and not dates.empty
                and dates.is_normalized
                and not dates.has_duplicates
            )
    except (OSError, ValueError, KeyError):
        return False


def _rename_time_coordinate(dataset: xr.Dataset) -> xr.Dataset:
    """Normalize the CDS v1/v2 time-coordinate name."""
    if "time" in dataset.coords:
        return dataset
    if "valid_time" in dataset.coords:
        return dataset.rename({"valid_time": "time"})
    raise ValueError("ERA5-Land response has no time or valid_time coordinate.")


def daily_accumulations_from_midnight(
    dataset: xr.Dataset,
    year: int,
    month: int,
) -> xr.Dataset:
    """Extract daily totals from ERA5-Land 00 UTC accumulations.

    ERA5-Land accumulated variables at 00 UTC represent the complete
    accumulation for the preceding UTC day.  Hourly values must therefore
    not be summed.  The caller supplies snapshots through 00 UTC on the first
    day of the following month so every target day is available.
    """
    normalized = _rename_time_coordinate(dataset)
    timestamps = pd.DatetimeIndex(normalized["time"].values)
    if timestamps.empty:
        raise ValueError("ERA5-Land response contains no timestamps.")
    midnight_mask = timestamps.hour == 0
    if not midnight_mask.any():
        raise ValueError("ERA5-Land response contains no 00 UTC accumulations.")

    midnight = normalized.isel(time=np.flatnonzero(midnight_mask))
    target_dates = pd.DatetimeIndex(midnight["time"].values) - pd.Timedelta(days=1)
    midnight = midnight.assign_coords(time=target_dates)
    expected_dates = pd.date_range(
        start=pd.Timestamp(year=year, month=month, day=1),
        end=pd.Timestamp(
            year=year,
            month=month,
            day=calendar.monthrange(year, month)[1],
        ),
        freq="D",
    )
    selected = midnight.sel(time=expected_dates)
    actual_dates = pd.DatetimeIndex(selected["time"].values)
    if not actual_dates.equals(expected_dates):
        raise ValueError(
            f"Daily accumulation coverage is incomplete for {year:04d}-{month:02d}."
        )
    if not REQUIRED_VARIABLES.issubset(set(selected.data_vars)):
        raise ValueError("Daily ERA5-Land output is missing precipitation or runoff.")
    valid_cell_fractions: dict[str, float] = {}
    for variable in REQUIRED_VARIABLES:
        values = np.asarray(selected[variable].values, dtype=float)
        finite = np.isfinite(values)
        if values.ndim < 1 or values.shape[0] != len(expected_dates):
            raise ValueError(f"Daily ERA5-Land {variable} has unexpected dimensions.")
        per_day = finite.reshape(len(expected_dates), -1)
        if not per_day.any(axis=1).all():
            raise ValueError(
                f"Daily ERA5-Land {variable} has a day with no valid grid cells."
            )
        # ERA5-Land contains a fixed NaN mask over ocean cells.  It is safe to
        # retain that mask because the dataset builder computes a skip-NaN,
        # latitude-weighted land-cell mean.  Time-varying missingness is not a
        # land/sea mask and is rejected.
        if not np.all(per_day == per_day[0]):
            raise ValueError(
                f"Daily ERA5-Land {variable} has time-varying missing grid cells."
            )
        if (values[finite] < -1e-10).any():
            raise ValueError(f"Daily ERA5-Land {variable} contains negative values.")
        valid_cell_fractions[variable] = float(per_day[0].mean())
    selected.attrs["daily_accumulation_method"] = DAILY_ACCUMULATION_METHOD
    selected.attrs["valid_cell_fraction_tp"] = valid_cell_fractions["tp"]
    selected.attrs["valid_cell_fraction_ro"] = valid_cell_fractions["ro"]
    selected.attrs["missing_grid_policy"] = (
        "fixed ERA5-Land ocean mask allowed; time-varying missing cells rejected"
    )
    return selected


def _request_payload(
    year: int,
    month: int,
    days: list[str],
    area: list[float],
) -> dict[str, object]:
    return {
        "variable": ["total_precipitation", "runoff"],
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": ["00:00"],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _retrieve_validated_netcdf(
    request: dict[str, object],
    destination: Path,
    keys: list[str],
    description: str,
    validation_attempts: int = 3,
) -> None:
    """Download and validate a CDS response, retrying incomplete payloads."""
    if validation_attempts < 1:
        raise ValueError("validation_attempts must be at least 1.")
    last_problem = "downloaded file did not contain tp and ro"
    for attempt in range(1, validation_attempts + 1):
        destination.unlink(missing_ok=True)
        cds_retrieve(DATASET_NAME, request, destination, keys)
        try:
            unpack_zip_to_netcdf(destination)
        except (OSError, ValueError) as exc:
            last_problem = str(exc)
        else:
            if valid_netcdf(destination, REQUIRED_VARIABLES):
                return
            last_problem = (
                f"invalid NetCDF payload ({destination.stat().st_size} bytes)"
                if destination.is_file()
                else "CDS response file is missing"
            )
        if attempt < validation_attempts:
            LOGGER.warning(
                "%s response failed validation on attempt %d/%d (%s); "
                "downloading it again.",
                description,
                attempt,
                validation_attempts,
                last_problem,
            )
    destination.unlink(missing_ok=True)
    raise ValueError(
        f"{description} ERA5-Land response failed validation after "
        f"{validation_attempts} attempts: {last_problem}."
    )


def retrieve_month(
    year: int,
    month: int,
    output_directory: Path,
    area: list[float],
    keys: list[str],
    overwrite: bool,
    dry_run: bool,
) -> DownloadResult:
    destination = output_directory / f"era5_land_daily_sum_{year}_{month:02d}.nc"

    if destination.exists() and not overwrite:
        if valid_daily_accumulation_file(destination):
            return DownloadResult(year, month, "skipped")
        LOGGER.warning(
            "Existing %04d-%02d file is legacy or invalid; it will be replaced "
            "only after a corrected download passes validation: %s",
            year,
            month,
            destination,
        )

    if dry_run:
        message = str(destination)
        if destination.exists() and not overwrite:
            message = f"repair invalid existing file: {destination}"
        return DownloadResult(year, month, "planned", message)

    request_id = uuid.uuid4().hex
    current_part = destination.with_name(
        f"{destination.name}.{request_id}.current.part"
    )
    boundary_part = destination.with_name(
        f"{destination.name}.{request_id}.boundary.part"
    )
    daily_part = destination.with_name(f"{destination.name}.{request_id}.daily.part")
    for temporary_path in (current_part, boundary_part, daily_part):
        temporary_path.unlink(missing_ok=True)
    valid_days = [
        f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]
    next_month = date(year, month, calendar.monthrange(year, month)[1]) + timedelta(
        days=1
    )
    current_request = _request_payload(year, month, valid_days, area)
    boundary_request = _request_payload(
        next_month.year,
        next_month.month,
        ["01"],
        area,
    )

    try:
        _retrieve_validated_netcdf(
            current_request,
            current_part,
            keys,
            "Current-month",
        )
        _retrieve_validated_netcdf(
            boundary_request,
            boundary_part,
            keys,
            "Next-month boundary",
        )

        from hardinge_high_flow.download_utils import _HDF5_LOCK

        with _HDF5_LOCK:
            with xr.open_dataset(current_part, decode_times=True) as current_dataset:
                current_loaded = _rename_time_coordinate(current_dataset).load()
            with xr.open_dataset(boundary_part, decode_times=True) as boundary_dataset:
                boundary_loaded = _rename_time_coordinate(boundary_dataset).load()
            combined = xr.concat(
                [current_loaded, boundary_loaded],
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            ).sortby("time")
            daily = daily_accumulations_from_midnight(combined, year, month)
            daily.to_netcdf(daily_part)

        if not valid_daily_accumulation_file(daily_part):
            raise ValueError("Processed daily ERA5-Land NetCDF failed validation.")
        daily_part.replace(destination)
        current_part.unlink(missing_ok=True)
        boundary_part.unlink(missing_ok=True)
        return DownloadResult(year, month, "downloaded")
    except Exception as exc:
        for temporary_path in (current_part, boundary_part, daily_part):
            temporary_path.unlink(missing_ok=True)
        return DownloadResult(year, month, "failed", str(exc))


def run_downloads(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    keys = load_cds_keys(PROJECT_ROOT)
    period = config["period"]
    start_year = args.start_year or int(str(period["start"])[:4])
    end_year = args.end_year or int(str(period["end"])[:4])
    if start_year > end_year:
        raise ValueError("start_year cannot exceed end_year.")

    output_directory = resolve_project_path(
        config,
        config["paths"]["raw_era5_accumulations"],
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    area = list(map(float, config["study_area"]["era5_area"]))
    tasks = list(iter_months(start_year, end_year))
    counts = {
        "downloaded": 0,
        "skipped": 0,
        "planned": 0,
        "failed": 0,
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                retrieve_month,
                year,
                month,
                output_directory,
                area,
                keys,
                args.overwrite,
                args.dry_run,
            ): (year, month)
            for year, month in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result.status] += 1
            level = logging.ERROR if result.status == "failed" else logging.INFO
            LOGGER.log(
                level,
                "%s %04d-%02d (%d/%d)%s",
                result.status.upper(),
                result.year,
                result.month,
                index,
                len(tasks),
                f": {result.message}" if result.message else "",
            )

    LOGGER.info("Download summary: %s", counts)
    return 1 if counts["failed"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land daily accumulated variables."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    raise SystemExit(run_downloads(parse_args()))


if __name__ == "__main__":
    main()
