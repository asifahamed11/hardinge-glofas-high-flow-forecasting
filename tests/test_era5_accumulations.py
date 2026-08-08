from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "download_era5_accumulations.py"
SPEC = importlib.util.spec_from_file_location("era5_accumulation_script", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the ERA5-Land accumulation downloader.")
ERA5_ACCUMULATIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ERA5_ACCUMULATIONS)


def test_midnight_accumulations_are_shifted_to_the_previous_day() -> None:
    timestamps = pd.date_range("2020-01-01", "2020-02-01", freq="D")
    values = np.arange(len(timestamps), dtype=float).reshape(-1, 1, 1) / 1_000
    dataset = xr.Dataset(
        {
            "tp": (("valid_time", "latitude", "longitude"), values),
            "ro": (("valid_time", "latitude", "longitude"), values / 2),
        },
        coords={
            "valid_time": timestamps,
            "latitude": [24.0],
            "longitude": [89.0],
        },
    )
    daily = ERA5_ACCUMULATIONS.daily_accumulations_from_midnight(
        dataset,
        2020,
        1,
    )
    expected_dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
    assert pd.DatetimeIndex(daily["time"].values).equals(expected_dates)
    np.testing.assert_allclose(daily["tp"].to_numpy().reshape(-1), values[1:, 0, 0])


def test_fixed_ocean_mask_is_allowed_but_varying_missingness_is_rejected() -> None:
    timestamps = pd.date_range("2020-01-01", "2020-02-01", freq="D")
    values = np.ones((len(timestamps), 1, 3), dtype=float)
    values[:, :, 2] = np.nan
    dataset = xr.Dataset(
        {
            "tp": (("time", "latitude", "longitude"), values),
            "ro": (("time", "latitude", "longitude"), values),
        },
        coords={
            "time": timestamps,
            "latitude": [24.0],
            "longitude": [89.0, 90.0, 91.0],
        },
    )

    daily = ERA5_ACCUMULATIONS.daily_accumulations_from_midnight(dataset, 2020, 1)
    assert daily.attrs["valid_cell_fraction_tp"] == 2 / 3

    varying = dataset.copy(deep=True)
    varying["tp"].values[2, 0, 0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "time-varying missing"):
        ERA5_ACCUMULATIONS.daily_accumulations_from_midnight(varying, 2020, 1)


def test_hourly_or_unmarked_files_are_not_treated_as_valid_daily_totals(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2020-01-01", periods=24, freq="h")
    values = np.arange(24, dtype=float).reshape(-1, 1, 1)
    dataset = xr.Dataset(
        {
            "tp": (("valid_time", "latitude", "longitude"), values),
            "ro": (("valid_time", "latitude", "longitude"), values),
        },
        coords={
            "valid_time": timestamps,
            "latitude": [24.0],
            "longitude": [89.0],
        },
    )
    path = tmp_path / "old_hourly_sum.nc"
    dataset.to_netcdf(path)
    assert not ERA5_ACCUMULATIONS.valid_daily_accumulation_file(path)


def test_invalid_existing_file_is_planned_for_atomic_repair(tmp_path: Path) -> None:
    timestamps = pd.date_range("2020-01-01", periods=2, freq="D")
    values = np.ones((2, 1, 1), dtype=float)
    dataset = xr.Dataset(
        {
            "tp": (("time", "latitude", "longitude"), values),
            "ro": (("time", "latitude", "longitude"), values),
        },
        coords={
            "time": timestamps,
            "latitude": [24.0],
            "longitude": [89.0],
        },
    )
    destination = tmp_path / "era5_land_daily_sum_2020_01.nc"
    dataset.to_netcdf(destination)

    result = ERA5_ACCUMULATIONS.retrieve_month(
        year=2020,
        month=1,
        output_directory=tmp_path,
        area=[27.0, 85.0, 22.0, 92.0],
        keys=[],
        overwrite=False,
        dry_run=True,
    )

    assert result.status == "planned"
    assert "repair invalid existing file" in result.message


def test_invalid_download_payload_is_retried(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "response.part"
    attempts = 0

    def fake_retrieve(_dataset_name, _request, output_path, _keys) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            output_path.write_bytes(b"incomplete")
            return
        values = np.ones((2, 1, 1), dtype=float)
        xr.Dataset(
            {
                "tp": (("time", "latitude", "longitude"), values),
                "ro": (("time", "latitude", "longitude"), values),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=2, freq="D"),
                "latitude": [24.0],
                "longitude": [89.0],
            },
        ).to_netcdf(output_path)

    monkeypatch.setattr(ERA5_ACCUMULATIONS, "cds_retrieve", fake_retrieve)
    ERA5_ACCUMULATIONS._retrieve_validated_netcdf(
        request={},
        destination=destination,
        keys=[],
        description="test",
    )

    assert attempts == 2
    assert ERA5_ACCUMULATIONS.valid_daily_accumulation_file(destination) is False
