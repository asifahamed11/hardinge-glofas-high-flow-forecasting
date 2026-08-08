from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "build_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_dataset_script", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load scripts/build_dataset.py for testing.")
BUILD_DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_DATASET)


def synthetic_dataset() -> xr.Dataset:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    latitude = [24.10, 24.05]
    longitude = [89.00, 89.05]
    shape = (len(dates), len(latitude), len(longitude))
    return xr.Dataset(
        {
            "t2m": (
                ("valid_time", "latitude", "longitude"),
                np.arange(np.prod(shape), dtype=float).reshape(shape) + 273.15,
                {"units": "K"},
            ),
            "dis24": (
                ("valid_time", "latitude", "longitude"),
                np.arange(np.prod(shape), dtype=float).reshape(shape) + 100.0,
                {"units": "m3 s-1"},
            ),
        },
        coords={
            "valid_time": dates,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def test_spatial_mean_and_unit_conversion() -> None:
    dataset = synthetic_dataset()
    series = BUILD_DATASET.extract_spatial_mean(
        dataset,
        ("t2m",),
        "temperature",
    )
    converted = BUILD_DATASET.normalize_temperature(
        series,
        dataset["t2m"].attrs["units"],
    )
    weights = np.cos(np.deg2rad(dataset["latitude"]))
    expected = (
        dataset["t2m"].weighted(weights).mean(("latitude", "longitude")).values - 273.15
    )
    np.testing.assert_allclose(converted.to_numpy(), expected)


def test_nearest_station_cell_is_selected() -> None:
    dataset = synthetic_dataset()
    series, coordinates = BUILD_DATASET.extract_nearest_point(
        dataset,
        ("dis24",),
        "discharge",
        latitude=24.07,
        longitude=89.04,
    )
    assert coordinates == (24.05, 89.05)
    expected = dataset["dis24"].sel(latitude=24.05, longitude=89.05).values
    np.testing.assert_allclose(series.to_numpy(), expected)


def test_glofas_end_timestamp_maps_to_preceding_day(tmp_path: Path) -> None:
    dataset = synthetic_dataset().drop_vars("t2m")
    dataset = dataset.assign_coords(
        valid_time=pd.date_range("2020-01-02", periods=3, freq="D")
    )
    path = tmp_path / "glofas.nc"
    dataset.to_netcdf(path, engine="scipy")

    series, coordinates = BUILD_DATASET.read_glofas_series(
        [path],
        latitude=24.07,
        longitude=89.04,
    )

    assert coordinates == (24.05, 89.05)
    assert series.index.equals(pd.date_range("2020-01-01", periods=3, freq="D"))
    expected = dataset["dis24"].sel(latitude=24.05, longitude=89.05).values
    np.testing.assert_allclose(series.to_numpy(), expected)


def test_water_depth_conversion() -> None:
    series = pd.Series([0.001, 0.002])
    np.testing.assert_allclose(
        BUILD_DATASET.normalize_water_depth(series, "m").to_numpy(),
        [1.0, 2.0],
    )


def test_missing_values_are_filled_only_from_the_past() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D", name="date")
    frame = pd.DataFrame(
        {
            "era5_precipitation_mm": [1.0, np.nan, 100.0],
            "nasa_precipitation_mm": [2.0, 2.0, 2.0],
        },
        index=dates,
    )
    validated = BUILD_DATASET.validate_dataset(
        frame,
        dates,
        maximum_missing_days=1,
        causal_fill_days=1,
    )
    assert validated.loc["2020-01-02", "era5_precipitation_mm"] == 1.0


def test_unresolved_missing_values_are_rejected() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D", name="date")
    frame = pd.DataFrame(
        {
            "era5_precipitation_mm": [np.nan, 1.0, 2.0],
            "nasa_precipitation_mm": [2.0, 2.0, 2.0],
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="Unresolved missing"):
        BUILD_DATASET.validate_dataset(
            frame,
            dates,
            maximum_missing_days=1,
            causal_fill_days=1,
        )


def test_unsafe_archive_member_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.nc", b"not a NetCDF")
    with (
        pytest.raises(ValueError, match="Unsafe archive"),
        BUILD_DATASET.materialized_netcdf(archive_path),
    ):
        pass
