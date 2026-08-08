"""Download monthly ERA5-Land daily-mean inputs."""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

LOGGER = logging.getLogger("era5_daily")

DATASET_NAME = "derived-era5-land-daily-statistics"
REQUIRED_VARIABLES = {"t2m", "swvl1"}


def retrieve_month(
    year: int,
    month: int,
    output_directory: Path,
    area: list[float],
    keys: list[str],
    overwrite: bool,
    dry_run: bool,
) -> DownloadResult:
    destination = output_directory / f"era5_land_daily_mean_{year}_{month:02d}.nc"

    if destination.exists() and not overwrite:
        if valid_netcdf(destination, REQUIRED_VARIABLES):
            return DownloadResult(year, month, "skipped")
        return DownloadResult(
            year,
            month,
            "failed",
            f"Existing file is invalid: {destination}",
        )

    if dry_run:
        return DownloadResult(year, month, "planned", str(destination))

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    valid_days = [
        f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]
    request = {
        "variable": [
            "2m_temperature",
            "volumetric_soil_water_layer_1",
        ],
        "year": str(year),
        "month": f"{month:02d}",
        "day": valid_days,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    try:
        cds_retrieve(DATASET_NAME, request, temporary, keys)
        unpack_zip_to_netcdf(temporary)
        if not valid_netcdf(temporary, REQUIRED_VARIABLES):
            raise ValueError("Downloaded NetCDF failed validation.")
        temporary.replace(destination)
        return DownloadResult(year, month, "downloaded")
    except Exception as exc:
        temporary.unlink(missing_ok=True)
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
        config["paths"]["raw_era5_daily"],
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
        description="Download monthly ERA5-Land daily means."
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
