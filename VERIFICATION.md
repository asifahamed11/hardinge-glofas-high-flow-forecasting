# Verification record

Last updated: 10 August 2026.

## Completed

- Python compilation and Ruff checks passed.
- 36 automated tests passed.
- Tests cover configuration, unit conversion, spatial extraction, ERA5-Land
  accumulation decoding, stable ocean masks, download retries, missing-data
  safeguards, target construction, sequence isolation, model refitting,
  calibration, event metrics, exact threshold handling, Average Precision
  naming, grouped permutation importance, uncertainty intervals, rolling folds,
  and figures.
- The complete 1981-2023 master and labelled datasets were built from the
  configured public products.
- The main five-seed experiment, all streamflow and imbalance ablations, and
  four expanding-window rolling-origin folds completed.
- Threshold-sensitive diagnostics were regenerated from saved predictions; the
  persistence event results now use each run's exact stored threshold.
- Tables consistently report Average Precision (AP), matching the implemented
  `average_precision_score` calculation.
- Random Forest grouped permutation importance was generated with and without
  current GloFAS discharge.
- The tracked release contains no CDS credentials, raw data, local environment,
  generated models, or machine-specific paths.
- The `main` branch is linked to the public GitHub repository.

One NetCDF test emits a binary-compatibility warning in the current local
environment. The test passes, but final numerical runs should use a clean
environment created from `requirements.txt`.

## Remaining release decisions

- choose a software licence and archive the exact analysis release;
- independently validate the GloFAS proxy against suitable gauge observations
  if such data become available;
- retain the proxy-target limitation in every scientific interpretation.
- retain the regional ERA5-Land box limitation; the present analysis is not a
  full-upstream, catchment-weighted rainfall-runoff experiment.

Run the automated release audit with:

```bash
python scripts/check_publication_readiness.py
```
