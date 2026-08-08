# Dataset acquisition and organization

The configured study period is 1 January 1981 through 31 December 2023. All paths are project-relative and can be changed in `configs/default.yaml`.

## 1. Copernicus account and API access

ERA5-Land and GloFAS require a free Copernicus account.

1. Register or sign in at the [Climate Data Store](https://cds.climate.copernicus.eu/).
2. Open each dataset page listed below and accept its terms of use.
3. Follow the official [CDS API setup](https://cds.climate.copernicus.eu/how-to-api).
4. Install `cdsapi>=0.7.7`.
5. Store the personal access token in the standard `.cdsapirc` file displayed by the portal.
6. Never place the token in this repository, a notebook, a script, a screenshot, or a public issue.

The download scripts use `cdsapi.Client()` without embedding a URL or token. This allows the official client configuration to work on Windows, macOS, and Linux.

## 2. ERA5-Land daily inputs

Official source: [ERA5-Land post-processed daily statistics](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics), DOI [10.24381/cds.e9c9c792](https://doi.org/10.24381/cds.e9c9c792).

The project requests:

| Statistic | Variables | Output directory |
|---|---|---|
| Daily mean | 2 m temperature; volumetric soil water layer 1 | `data/raw/era5_land/daily_mean/` |
| Daily sum | Total precipitation; runoff | `data/raw/era5_land/daily_sum/` |

The configured domain is `[26.6, 88.0, 20.7, 92.6]` in
north-west-south-east order. The processing script calculates a
cosine-latitude-weighted grid mean, converts Kelvin to degrees Celsius, and
converts metres of water equivalent to millimetres.

Run:

```bash
python scripts/download_era5_daily.py --config configs/default.yaml
python scripts/download_era5_accumulations.py --config configs/default.yaml
```

ERA5-Land precipitation and runoff are forecast accumulations, not independent
hourly increments.  The downloader requests only the 00 UTC snapshots, which
represent the complete accumulation for the preceding UTC day, shifts their
timestamps back by one day, and verifies exact calendar coverage.  It never
sums the cumulative hourly fields.

ERA5-Land is a land-only product, so ocean cells inside the configured box are
stored as a fixed missing-value mask. The downloader records the valid-cell
fraction, permits only a time-invariant land/sea mask, and rejects a response
if an entire day is missing or the mask changes over time. Dataset aggregation
uses a skip-missing, cosine-latitude-weighted mean over valid land cells.

Before a large request, inspect the planned filenames:

```bash
python scripts/download_era5_daily.py --start-year 1981 --end-year 1981 --dry-run
python scripts/download_era5_accumulations.py --start-year 1981 --end-year 1981 --dry-run
```

Each completed monthly NetCDF is validated and moved atomically from a unique
temporary file. Invalid legacy files are automatically downloaded again and
are replaced only after the corrected response passes validation.

## 3. GloFAS historical discharge

Official source: [River discharge and related historical data from GloFAS](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical), DOI [10.24381/cds.a4fdd6b9](https://doi.org/10.24381/cds.a4fdd6b9).

Run:

```bash
python scripts/download_glofas.py --config configs/default.yaml
```

Files are stored in `data/raw/glofas/`. The request uses:

- GloFAS version 4.0;
- LISFLOOD;
- consolidated historical data;
- river discharge averaged over the previous 24 hours;
- NetCDF output;
- a small configured domain around Hardinge Bridge.

The processing script selects the GloFAS grid cell nearest the configured Hardinge Bridge coordinates. It does not average every cell in a one-degree box.

Important scientific limitation: GloFAS historical discharge is a gridded LISFLOOD simulation forced by ERA5, not a BWDB gauge observation and not a medium-range forecast. When this series supplies the label, use the terms **GloFAS modelled discharge** and **high-flow proxy**, not “observed flood.”

## 4. NASA POWER daily point data

Official source: [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/).

No account or API key is required. If the file is absent, `scripts/build_dataset.py` requests:

- `T2M`: 2 m air temperature;
- `PRECTOTCORR`: corrected precipitation;
- `RH2M`: 2 m relative humidity;
- the configured Hardinge Bridge point;
- UTC daily values;
- CSV output.

The file is stored as:

```text
data/raw/nasa_power/nasa_power_daily.csv
```

NASA POWER daily data begin on 1 January 1981. The script explicitly requests `time-standard=UTC`; the API otherwise defaults to local solar time.

Use `python scripts/build_dataset.py --offline` to prohibit this automatic download.

## 5. Independent BWDB/FFWC observations

For the scientifically preferred journal experiment, obtain daily observed discharge or water level from the Bangladesh Water Development Board.

Official starting points:

- [BWDB Hydroinformatics and Flood Forecasting Circle](https://www.hydrology.bwdb.gov.bd/)
- [BWDB water-level data viewer](https://www.hydrology.bwdb.gov.bd/index.php?pagetitle=water_level_data_view)
- [BWDB data availability and rates](https://www.hydrology.bwdb.gov.bd/index.php?id=225&pagetitle=rate_of_data&subid=131)
- [Flood Forecasting and Warning Centre](https://ffwc.gov.bd/)

Recommended request procedure:

1. Confirm the official station identifier for Hardinge Bridge.
2. Request the longest quality-controlled daily series available, preferably 1981–2023.
3. Request station coordinates, measurement type, unit, datum, rating-curve history, danger level, missing-value codes, and quality flags.
4. Confirm whether the supplied values are instantaneous, daily mean, or daily maximum.
5. Confirm the data licence and whether redistribution is permitted.
6. Save the licensed file locally as `data/external/observations/hardinge_observations.csv`.
7. Do not commit restricted observations to GitHub.

If the observed record covers a different continuous period, update
`period.start`, `period.end`, and all three split end dates in
`configs/default.yaml` before downloading and processing the public inputs.
The label script requires exact daily target coverage for the configured
period; it will not convert missing target dates into non-events.

The default schema is:

```csv
date,observed_discharge_m3s,quality_flag
2019-01-01,12345.6,approved
```

If BWDB supplies water level instead of discharge, change `target.observed.value_column`, `target.observed.name`, and `target.observed.unit` in the configuration. If an official danger level is available, set `target.observed.fixed_threshold`; otherwise the code fits the configured quantile on the training period only.

If the licensed file contains a formal quality field, set
`target.observed.quality_flag_column` and list the permitted values under
`target.observed.accepted_quality_flags`. Matching is case-insensitive. The
loader rejects duplicate dates, missing values, non-numeric values, and a
quality filter that leaves no observations. It does not silently reinterpret
missing observations as non-events.

Create observed labels with:

```bash
python scripts/create_high_flow_labels.py \
  --config configs/default.yaml \
  --target-source observed
```

The code never silently falls back to GloFAS when an observed file is missing.

## 6. Build the merged dataset

After all required inputs are present:

```bash
python scripts/build_dataset.py --config configs/default.yaml --offline
python scripts/create_high_flow_labels.py --config configs/default.yaml
```

The builder:

- sorts input files deterministically;
- validates variables and dates;
- safely handles a legacy archive containing one NetCDF;
- converts units explicitly;
- selects one consistent GloFAS grid cell;
- enforces unique, consecutive daily dates;
- refuses missing values beyond the configured limit;
- writes CSV and Parquet versions;
- records row counts, units, extraction coordinates, software versions, and SHA-256 hashes.

## Expected data tree

```text
data/
├── external/
│   └── observations/
│       └── hardinge_observations.csv
├── processed/
│   ├── labeled_daily.csv
│   ├── labeled_daily.parquet
│   ├── master_daily.csv
│   └── master_daily.parquet
└── raw/
    ├── era5_land/
    │   ├── daily_mean/
    │   │   └── era5_land_daily_mean_YYYY_MM.nc
    │   └── daily_sum/
    │       └── era5_land_daily_sum_YYYY_MM.nc
    ├── glofas/
    │   └── glofas_historical_YYYY_MM.nc
    └── nasa_power/
        └── nasa_power_daily.csv
```

Raw, processed, licensed, and generated files are excluded from Git. Only instructions and example schemas are versioned.
