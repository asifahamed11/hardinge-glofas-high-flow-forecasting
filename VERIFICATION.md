# Verification record

Verification date: 8 August 2026

This record distinguishes checks completed on the supplied refactor from
data-dependent or environment-specific work that must be performed by the
authors.

## Completed checks

| Area | Result |
|---|---|
| Python syntax | `compileall` passed for `src/`, `scripts/`, and `tests/`. |
| Static quality | Ruff formatting and all configured lint rules passed. |
| Automated tests | 30 tests passed, including corrected ERA5-Land accumulation decoding, stable land/sea-mask handling, invalid-response retries, atomic legacy-file repair, causal missing-data rejection, full-training neural refitting, and valid paired bootstrap resampling. One NetCDF test emitted a local binary-compatibility warning because the active environment does not match the pinned requirements. |
| Configuration | YAML and TOML parsed successfully; required sections, chronological splits, geographic bounds, target mode, horizons, imbalance strategy, absolute paths, and path traversal are validated. |
| Packaging | A standards-based Python wheel built successfully without dependency resolution. |
| Dependencies | All 14 runtime imports are covered by exact pins, and every pin satisfies the corresponding `pyproject.toml` version range. |
| Notebook | Valid notebook format 4.5; no stored outputs, credentials, or duplicated model definitions. |
| Repository hygiene | Source-code credential, machine-specific-path, stale-name, and unfinished-marker scans passed. The local `cds_keys.txt` is non-empty and must be removed from the release and its keys rotated. |
| Acquisition interfaces | The ERA5-Land accumulation downloader planned 12 valid monthly products for a one-year dry run. Each product uses the 00 UTC snapshot for every following day, including a next-month boundary request, and maps it to the preceding UTC day. A live repair of December 2023 produced 31 verified days with a stable 0.7975 valid-land-cell fraction. |
| Dataset helpers | Synthetic tests passed for Kelvin-to-Celsius conversion, metre-to-millimetre conversion, latitude-weighted spatial averaging, nearest GloFAS cell selection, and unsafe archive rejection. |
| Leakage safeguards | Tests confirmed training-only feature thresholds, split-isolated multi-horizon sequences, past-only missing-data handling, and rejection of unresolved missing values. |
| Target construction | Tests confirmed training-only label thresholds, explicit observed-data quality filtering, and rejection of missing observed targets. |
| Evaluation | Tests passed for threshold selection, binary metrics, sigmoid calibration, aggregate and per-event metrics, warning lead time, year-block intervals, cost–loss analysis, fold-specific training thresholds, and seed-matched paired comparisons. |
| Temporal validation | Four expanding rolling-origin folds were generated chronologically and independently relabelled. |
| Figures | Five synthetic publication figures exported to PDF, EPS, PNG, and TIFF: 20 files in total. Raster files were RGB, approximately 600 dpi, and larger than 1,000 pixels in each dimension. EPS export produced no transparency warning. |

## Checks requiring the authors' environment

The following cannot be honestly marked as completed without credentials,
licensed data, specialist software, or a full numerical run:

1. Download all ERA5-Land and GloFAS files after accepting the official
   Copernicus terms and configuring `.cdsapirc`.
2. Build the 1981–2023 dataset from the complete remote products and inspect
   the recorded grid coordinates, missing-value counts, checksums, and date
   range.
3. Obtain and quality-control the BWDB/FFWC observations. The repository does
   not fabricate or redistribute them.
4. Recreate the labels after rebuilding the master dataset. The experiment now
   refuses stale labels whose recorded input fingerprints do not match the
   current master data.
5. Run the full multi-seed, multi-horizon experiment and rolling-origin
   validation. These are intentionally not replaced by synthetic scientific
   results.
6. Execute `scripts/create_qgis_layers.py` inside QGIS and inspect the exported
   GeoJSON layers.
7. Create a clean environment from `requirements.txt`. The machine used for
   this verification currently has newer NumPy, pandas, scikit-learn, PyTorch,
   and Xarray packages plus unrelated dependency conflicts, so its successful
   test run is not a substitute for the pinned reproducibility run.

## Final pre-publication checks

Before a manuscript submission or public release:

1. Use the independently observed target if licensing and coverage permit.
2. Run every ablation and rolling-origin experiment documented in `README.md`.
3. Inspect all output metadata and plots, then compare reported manuscript
   values directly with the generated tables.
4. Justify or replace the configured latitude-weighted ERA5-Land box average with an
   upstream catchment mask.
5. Restrict claims to retrospective high-flow hindcasting unless predictors
   available at forecast issue time are substituted.
6. Choose and add the authors' intended software licence, author metadata,
   repository URL, and archived-release DOI.
