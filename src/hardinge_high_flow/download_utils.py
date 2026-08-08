"""Shared utilities for CDS API download scripts."""

from __future__ import annotations

import logging
import random
import shutil
import threading
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# HDF5 is not thread-safe; serialize all NetCDF I/O to avoid
# spurious HDF5-DIAG warnings when multiple threads validate files.
_HDF5_LOCK = threading.Lock()

CDS_API_URL = "https://cds.climate.copernicus.eu/api"
EWDS_API_URL = "https://ewds.climate.copernicus.eu/api"
MAX_RETRIES = 10

RETRYABLE_PATTERNS = [
    "temporarily limited",
    "rejected",
    "status code 400",
    "status code 429",
    "status code 500",
    "status code 502",
    "status code 503",
    "status code 504",
    "request limit",
    "too many requests",
]


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a single month download attempt."""

    year: int
    month: int
    status: str
    message: str = ""


def load_cds_keys(project_root: Path) -> list[str]:
    """Load CDS API keys from ``cds_keys.txt`` or ``.cdsapirc``.

    Keys are never hardcoded in scripts.  This function reads them
    from one of two file-based sources in order of priority.
    """
    keys_file = project_root / "cds_keys.txt"
    if keys_file.is_file():
        keys = [
            line.strip()
            for line in keys_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if keys:
            LOGGER.info("Loaded %d CDS API key(s) from %s", len(keys), keys_file)
            return keys

    rc_file = project_root / ".cdsapirc"
    if rc_file.is_file():
        for line in rc_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("key:"):
                key = line.split(":", 1)[1].strip()
                if key:
                    LOGGER.info("Loaded 1 CDS API key from %s", rc_file)
                    return [key]

    raise FileNotFoundError(
        "No CDS API keys found.  "
        "Create cds_keys.txt or .cdsapirc in the project root."
    )


def valid_netcdf(
    path: Path,
    required_variables: set[str],
    *,
    require_all: bool = True,
) -> bool:
    """Check that *path* is a readable NetCDF with expected variables.

    Parameters
    ----------
    required_variables:
        Variable names to look for in the file.
    require_all:
        If ``True`` (default), **all** variables must be present.
        If ``False``, **any** match is sufficient (useful for GloFAS
        where the variable name may differ between API versions).
    """
    if not path.is_file() or path.stat().st_size < 2_000:
        return False

    try:
        import xarray as xr

        with _HDF5_LOCK, xr.open_dataset(path, decode_times=False) as dataset:
            available = set(dataset.data_vars)
            if require_all:
                return required_variables.issubset(available)
            return bool(required_variables & available)
    except (OSError, ValueError):
        return False


def iter_months(
    start_year: int,
    end_year: int,
) -> Iterable[tuple[int, int]]:
    """Yield ``(year, month)`` pairs for a closed year range."""
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def _is_retryable(error: Exception) -> bool:
    """Return ``True`` if the CDS API error looks transient."""
    error_text = str(error).lower()
    return any(pattern in error_text for pattern in RETRYABLE_PATTERNS)


def cds_retrieve(
    dataset_name: str,
    request: dict,
    output_path: Path,
    keys: list[str],
    *,
    api_url: str = CDS_API_URL,
    max_retries: int = MAX_RETRIES,
) -> None:
    """Retrieve a dataset from the CDS API with retry and key rotation.

    Parameters
    ----------
    api_url:
        Override the default CDS API URL.  GloFAS datasets live on the
        EWDS endpoint (``https://ewds.climate.copernicus.eu/api``).

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.
    """
    import cdsapi

    # Each worker gets an independent pool.  Mutating the shared list supplied
    # to concurrent monthly downloads creates a race and can exhaust valid keys
    # in unrelated workers.
    available_keys = list(keys)
    for attempt in range(1, max_retries + 1):
        if not available_keys:
            raise RuntimeError(f"All CDS API keys have been exhausted for {dataset_name}.")
        key = random.choice(available_keys)
        client = cdsapi.Client(
            url=api_url,
            key=key,
            progress=False,
            quiet=True,
        )
        try:
            client.retrieve(dataset_name, request, str(output_path))
            return
        except Exception as exc:
            if _is_retryable(exc) and attempt < max_retries:
                sleep_time = min(60 * attempt, 600)
                LOGGER.warning(
                    "Attempt %d/%d failed (%s), retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
            elif _is_retryable(exc):
                raise RuntimeError(
                    f"CDS API request for {dataset_name} failed "
                    f"after {max_retries} attempts: {exc}"
                ) from exc
            else:
                # For non-retryable errors (like 403 Forbidden / missing licenses),
                # drop the key so it's not used again and retry with remaining keys.
                LOGGER.warning(
                    "A CDS API key failed with a non-retryable error (%s); "
                    "removing it from this worker's retry pool.",
                    exc,
                )
                available_keys.remove(key)
                if not available_keys:
                    raise RuntimeError(f"All CDS API keys exhausted. Last error: {exc}") from exc


def unpack_zip_to_netcdf(temporary: Path) -> None:
    """If *temporary* is a ZIP archive, extract and merge its NetCDF files.

    After this call the file at *temporary* is guaranteed to be a plain
    NetCDF file (or the call raises).
    """
    if not zipfile.is_zipfile(temporary):
        return

    import xarray as xr

    with zipfile.ZipFile(temporary, "r") as archive:
        nc_files = [f for f in archive.namelist() if f.lower().endswith(".nc")]
        if not nc_files:
            raise ValueError("Downloaded ZIP contains no .nc file.")
        extracted_paths: list[Path] = []
        for index, name in enumerate(nc_files):
            member_path = Path(name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("Downloaded ZIP contains an unsafe member path.")
            destination = temporary.parent / f"{temporary.name}.{index}.nc"
            with archive.open(name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_paths.append(destination)

    if len(extracted_paths) == 1:
        extracted_paths[0].replace(temporary)
    else:
        datasets = [
            xr.open_dataset(p, decode_times=False) for p in extracted_paths
        ]
        merged = xr.merge(datasets)
        merged.to_netcdf(temporary)
        for ds in datasets:
            ds.close()
        for p in extracted_paths:
            p.unlink()
