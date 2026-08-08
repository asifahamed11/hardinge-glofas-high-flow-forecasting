# Data guide

The configured study period is 1 January 1981 through 31 December 2023. Paths,
coordinates, variables, and date splits are defined in `configs/default.yaml`.

## Access

ERA5-Land and GloFAS require a free Copernicus account. Accept the terms on each
dataset page and configure the official CDS API client using the instructions at
[CDS API setup](https://cds.climate.copernicus.eu/how-to-api). Keep `.cdsapirc`
and any local credential file out of version control.

## Sources

| Dataset | Project use | Local path |
|---|---|---|
| [ERA5-Land daily statistics](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics) | daily temperature and top-layer soil moisture | `data/raw/era5_land/daily_mean/` |
| [ERA5-Land hourly data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | daily precipitation and runoff accumulations | `data/raw/era5_land/daily_sum/` |
| [GloFAS historical discharge](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical) | modelled discharge proxy | `data/raw/glofas/` |
| [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) | point temperature, precipitation, and humidity | `data/raw/nasa_power/` |

Download the three Copernicus products with:

```bash
python scripts/download_era5_daily.py
python scripts/download_era5_accumulations.py
python scripts/download_glofas.py
```

NASA POWER is downloaded automatically when the merged dataset is built. Use
`--offline` to require an existing local file.

## ERA5-Land accumulations

Precipitation and runoff are accumulated forecast fields, not independent
hourly increments. The downloader requests 00 UTC values, shifts them to the
preceding day, and requests the next-month boundary needed to retain the final
day of each month. It never sums cumulative hourly values.

ERA5-Land is land-only. A fixed ocean mask inside the configured box is allowed;
time-varying missing cells or a completely missing day are rejected. Spatial
aggregation uses a cosine-latitude-weighted mean over valid land cells.

## Target definition

The default target is `glofas_proxy`: exceedance of a threshold fitted from
training-period GloFAS-modelled discharge. GloFAS historical discharge is a
LISFLOOD simulation forced by meteorological inputs. It is neither a gauge
observation nor an issued medium-range forecast.

Create the default labels with:

```bash
python scripts/create_high_flow_labels.py --target-source glofas_proxy
```

The code also supports independently measured daily discharge or water level.
This is optional and is not required for the default proxy workflow. The local
schema is documented in `data/external/observations/README.md`; restricted data
must not be committed.

## Build the dataset

After the required inputs are present:

```bash
python scripts/build_dataset.py --offline
python scripts/create_high_flow_labels.py --target-source glofas_proxy
```

The builder validates variables and dates, converts units, selects a consistent
GloFAS grid cell, rejects unresolved missing values, and records SHA-256 hashes
for the generated CSV, Parquet, and metadata files.

## Local layout

```text
data/
├── external/observations/     optional measured series
├── processed/                 merged and labelled datasets
└── raw/
    ├── era5_land/daily_mean/
    ├── era5_land/daily_sum/
    ├── glofas/
    └── nasa_power/
```

Raw, processed, licensed, and generated files are excluded from Git. Only this
guide and small example schemas are versioned.
