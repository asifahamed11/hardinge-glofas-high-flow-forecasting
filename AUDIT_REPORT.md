# File-by-file code audit

## Executive finding

All eight supplied code files required material updates. None was suitable for a public reproducibility repository in its original form.

The most urgent issue was security: `dataset.ipynb`, `code_1.text`, `code_2.text`, and `code_3.text` contained the same personal CDS access token. It has been removed from every revised file. Because it appeared in uploaded source code, the owner should revoke or rotate that token in the Copernicus account before using the project.

The most important scientific issue was target interpretation. The original label represented the 95th-percentile exceedance of GloFAS modelled discharge. The revised project calls this a GloFAS high-flow proxy and supports an independent BWDB/FFWC target without silently mixing the two.

## 1. `dataset.ipynb`

### Original findings

- Seven large stateful code cells duplicated every standalone script.
- Stored outputs made the notebook approximately 2.6 MB and obscured which cells produced the displayed results.
- A personal API token was embedded in three acquisition cells.
- Download years extended to 2026 while the chapter described a 1980–2023 analysis.
- Downloading, extraction, merging, labelling, modelling, and mapping were mixed in one execution state.
- Global warning suppression could hide invalid variables, coordinate mismatches, and numerical problems.
- Paths and output filenames depended on the current working directory.
- The notebook labelled a modelled GloFAS percentile as a flood.
- The modelling cell duplicated `code_6.text` and used 150-dpi figures.
- The map depended on live third-party tiles and contained a thesis caption inside the code.

### Implemented revision

Replaced by `notebooks/workflow.ipynb`.

- Removed all stored outputs and credentials.
- Reduced the notebook to an explicit interface around tested scripts.
- Added project-root discovery using `pyproject.toml`.
- Disabled downloads, preprocessing, and training by default behind visible switches.
- Added return-code checking through `subprocess.run(..., check=True)`.
- Documented the default target as a GloFAS high-flow proxy.
- Pointed every stage to the shared configuration.
- Moved model, data, and figure logic into importable source modules.

### Verification

- Valid notebook JSON.
- Notebook format 4.5.
- No stored outputs.
- No secret or machine-specific path pattern.

## 2. `Untitled-0.py`

### Original findings

- The filename did not explain the script’s purpose.
- Imports assumed one specific PyQt installation rather than QGIS’s compatibility namespace.
- Coordinates and geometries were hard-coded independently from the data scripts.
- Re-running the script created duplicate in-memory layers.
- Layers contained only a name field and no source, role, bounds, or coordinate metadata.
- No symbology or persistent export was provided.
- The script executed immediately on import and gave no useful error outside QGIS.

### Implemented revision

Replaced by `scripts/create_qgis_layers.py`.

- Loads the same configured study points, domains, and journal palette as the analysis.
- Uses `qgis.PyQt`.
- Validates that QGIS is available and reports a clear failure otherwise.
- Replaces previously generated layers by name.
- Adds source, role, coordinates, and bounding values as attributes.
- Applies categorized point and polygon symbology.
- Exports reproducible GeoJSON layers.
- Provides a command-line interface and main guard.

### Verification

- Python syntax compilation passed.
- QGIS execution is documented as an environment-specific check.

## 3. `code_1.text`

### Original findings

- Personal CDS token embedded in source.
- `.text` extension obscured that the file was Python.
- All months requested days 01–31, including impossible dates.
- A file was skipped based only on existence, so interrupted files could be treated as complete.
- Downloads wrote directly to final paths.
- Years, area, worker count, and directory were hard-coded.
- No command-line interface, main guard, structured result, or non-zero failure status.
- Broad exceptions returned only a short console message.

### Implemented revision

Replaced by `scripts/download_era5_daily.py`.

- Uses standard CDS client authentication.
- Creates calendar-valid day lists.
- Reads years, domain, and paths from configuration with optional CLI overrides.
- Writes to `.part`, validates variables and file size, then performs an atomic rename.
- Refuses to skip an invalid existing file.
- Adds dry-run, overwrite, worker, year, and configuration arguments.
- Returns a failing process status when any month fails.
- Uses deterministic, meaningful filenames.

### Verification

- Syntax compilation passed.
- Credential-pattern scan passed.
- Dry-run behavior is included in final integration checks.

## 4. `code_2.text`

### Original findings

- Repeated the credential and most of `code_1.text`.
- Imported but did not use `Semaphore`.
- Downloaded 24 hourly fields per day only to calculate daily totals later.
- Used invalid 31-day requests for short months.
- Blindly skipped any existing file.
- Created a very large transfer and storage burden for four decades of data.

### Implemented revision

Replaced by `scripts/download_era5_accumulations.py`.

- Uses a descriptive filename and downloads the official ERA5-Land 00 UTC
  accumulation snapshot for each following day.
- Shifts every 00 UTC timestamp to the preceding UTC day and separately
  requests the first day of the next month so the final day is not lost.
- Rejects legacy files that do not record the verified accumulation method,
  preventing the former sum-of-cumulative-values error from being reused.
- Uses valid calendar days, configured paths, atomic files, and NetCDF validation.
- Removes the unused synchronization object and embedded credential.
- Adds the same resumable CLI and failure handling as the daily-mean downloader.

### Verification

- Syntax compilation passed.
- Credential-pattern scan passed.

## 5. `code_3.text`

### Original findings

- Personal token and service URL embedded in code.
- Warnings were globally suppressed.
- A one-degree box was later averaged into a discharge value, blending river and non-river cells.
- Existing-file validity was determined only by a 2 KB size threshold.
- No NetCDF variable check or atomic completion.
- Fixed years, GloFAS version, area, and output directory.

### Implemented revision

Replaced by `scripts/download_glofas.py`.

- Uses official CDS client configuration.
- Reads the version, compact station domain, years, and paths from one configuration.
- Validates the discharge variable inside NetCDF.
- Uses calendar-valid dates and atomic completion.
- Adds dry-run, overwrite, worker, and year controls.
- Separates acquisition from station-grid extraction.

### Verification

- Syntax compilation passed.
- Credential-pattern scan passed.

## 6. `code_4.text`

### Original findings

- Unsorted glob order made ingestion order platform-dependent.
- Zip extraction could overwrite colliding paths and did not check traversal.
- Entire gridded NetCDFs were converted to tabular data, causing unnecessary memory use.
- Variable selection could silently choose the first unknown column.
- ERA5 temperature remained in Kelvin while NASA temperature was Celsius.
- ERA5 water depths remained in metres while NASA precipitation was millimetres.
- GloFAS was averaged over the whole download box instead of extracting a river cell.
- NASA filename and location were hard-coded.
- Missing dates and duplicate dates were not validated.
- Errors were printed and ignored, allowing a partial dataset to be called successful.
- Inner joins silently changed the study period.
- No checksums, units, grid coordinates, software versions, or provenance were saved.

### Implemented revision

Replaced by `scripts/build_dataset.py`.

- Sorts every input deterministically.
- Safely materializes a legacy archive containing exactly one NetCDF.
- Reduces Xarray objects spatially before conversion using
  cosine-latitude-weighted ERA5-Land means.
- Requires explicit variable aliases and coordinate names.
- Converts Kelvin to Celsius and metres of water depth to millimetres.
- Selects the nearest GloFAS grid cell consistently and records its coordinates.
- Downloads NASA POWER from the official API at the configured point when permitted.
- Explicitly requests UTC daily NASA values.
- Enforces unique, sorted, consecutive dates and a configured missing-data limit.
- Writes CSV and Parquet.
- Writes units, extraction method, row counts, date coverage, software versions, and SHA-256 hashes.

### Verification

- Syntax compilation passed.
- Pure preprocessing helpers are statically checked.
- NetCDF execution requires the pinned Xarray and NetCDF dependencies.

## 7. `code_5.text`

### Original findings

- The discharge threshold was fitted on the full 44-year record, leaking validation and test distribution information.
- The label was called `is_flood` even though the source was modelled GloFAS discharge.
- No observed target, official threshold, explicit split, event identifier, or metadata was supported.
- The output did not record the numerical threshold or fitting period.

### Implemented revision

Replaced by `scripts/create_high_flow_labels.py`.

- Fits a quantile threshold on the training period only.
- Supports a configured fixed danger threshold.
- Uses the neutral model input column `target_high_flow`.
- Records `target_source`, `target_value`, and contiguous event identifiers.
- Supports `glofas_proxy` and `observed` modes.
- Refuses a missing observed file instead of falling back.
- Verifies that train, validation, and test blocks all contain positive examples.
- Writes CSV, Parquet, split distributions, threshold method, fitting period, and hashes.

### Verification

- Syntax compilation passed.
- Training-only threshold behavior is covered by tests.

## 8. `code_6.text`

### Original strengths retained

- Chronological outer holdout.
- Scaler fitted on training observations.
- Threshold chosen from validation rather than test predictions.
- Rare-event metrics and confusion matrices.
- Early stopping, gradient clipping, and checkpointing.

### Original findings

- More than 800 lines executed at import time.
- Feature extremes used full-record quantiles.
- Early rolling and lag values were replaced by zero.
- One horizon, one split, and one seed did not quantify temporal or stochastic uncertainty.
- No persistence, climatology, classical ML, or raw-GloFAS benchmark.
- Focal loss, positive weighting, and weighted sampling were applied simultaneously.
- The resulting scores were treated as probabilities without calibration checks.
- The “ConvLSTM” was actually Conv1D followed by LSTM.
- Threshold and early-stopping roles shared one validation block.
- Only day-wise metrics were reported for serially correlated events.
- No Brier score, calibration curve, event detection, confidence interval, paired comparison, or cost–loss analysis.
- Test figures were 150 dpi, oversized, title-heavy, color-only, and raster-only.
- Outputs were written into the working directory with inconsistent names.

### Implemented revision

Replaced by `scripts/train_evaluate.py` and modules in `src/hardinge_high_flow/`.

- Causal feature engineering is fitted on training data and reset at split boundaries.
- Missing rolling history is dropped rather than imputed as zero.
- Standardization is fitted on training rows only.
- Sequences remain within train, validation, or test blocks.
- Horizons are 1, 3, 5, and 7 days.
- Models include seasonal climatology, persistence, a current-day historical
  GloFAS signal, logistic regression, random forest, histogram gradient
  boosting, LSTM, GRU, CNN–LSTM, and attention-LSTM.
- Five seeds are configured.
- Neural early stopping uses a final chronological training block.
- Validation is divided into calibration and threshold-selection blocks.
- Imbalance methods are mutually exclusive and runnable as named ablations.
- Streamflow inclusion is a named, non-overwriting ablation.
- Outputs include PR-AUC, ROC-AUC, Brier score, expected calibration error,
  aggregate event scores, a per-event peak and warning-lead table, yearly
  scores, magnitude groups, cost–loss value, and year-block confidence
  intervals.
- Paired block-bootstrap comparisons use persistence as the reference.
- Expanding-window rolling-origin validation is available separately.
- Checkpoints include feature names, horizon, seed, calibration, and best epoch.
- Figures use the shared journal style, accessible encodings, vector output, and 600-dpi raster output.

### Verification

- Syntax compilation passed for every script and source module.
- Feature, sequence, metric, calibration, event, bootstrap, rolling-fold, and figure tests passed in the available runtime.
- Deep-learning smoke execution passed in the available runtime. A final
  submission run must still be repeated in a clean environment matching the
  pinned requirements.

## Cross-project consistency

| Requirement | Implementation |
|---|---|
| Portable paths | `pathlib`, project-relative YAML, optional `HIGHFLOW_PROJECT_ROOT` |
| One source of settings | `configs/default.yaml` |
| Secret handling | Standard `.cdsapirc`; secret patterns excluded and scanned |
| Reproducible environment | `pyproject.toml`, pinned runtime and development requirements |
| Leakage-safe labels | Training-only threshold or fixed observed threshold |
| Leakage-safe features | Training-only extremes/scaler; causal windows; split isolation |
| Target honesty | Explicit `glofas_proxy` versus `observed` modes |
| Multi-horizon evaluation | 1, 3, 5, and 7 days |
| Baselines | Climatology, persistence, current-day GloFAS signal, and three classical models |
| Uncertainty | Five seeds, yearly blocks, confidence intervals, paired bootstrap |
| Event usefulness | Event detection, misses, false alarms, onset delay, cost–loss |
| Sensitivity | Streamflow and four imbalance strategies |
| Temporal robustness | Final holdout plus four rolling-origin folds |
| Publication figures | Numbered vector and 600-dpi raster outputs |
| Provenance | Units, dates, versions, hashes, configuration, grid coordinates |
| Release hygiene | `.gitignore`, tests, docs, example schema, and a fail-fast publication-readiness audit; version control and a software licence remain author actions |

## Scientific limits code alone cannot remove

1. The repository cannot supply licensed BWDB/FFWC observations. The authors must obtain, verify, and document them.
2. A GloFAS high-flow proxy remains a model-to-model assessment, even when evaluated perfectly.
3. ERA5-Land and NASA POWER are retrospective products. A genuinely operational multi-day warning claim requires forecast or reforecast predictors available at issue time.
4. The configured latitude-weighted ERA5 spatial mean is not a physical
   upstream-catchment aggregation. A journal revision should justify the
   domain or replace it with catchment-masked upstream predictors.
5. A second observed station would materially strengthen external validation.

These are empirical requirements, not software defects, and should remain explicit in the manuscript.

## Files requiring no update

None. Every supplied code file required changes.
