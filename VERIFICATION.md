# Verification record

Last updated: 8 August 2026.

## Completed

- Python compilation and Ruff checks passed.
- 30 automated tests passed.
- Tests cover configuration, unit conversion, spatial extraction, ERA5-Land
  accumulation decoding, stable ocean masks, download retries, missing-data
  safeguards, target construction, sequence isolation, model refitting,
  calibration, event metrics, uncertainty intervals, rolling folds, and figures.
- A live December 2023 accumulation repair produced 31 verified days with the
  expected previous-day timestamps.
- The tracked release contains no CDS credentials, raw data, local environment,
  generated models, or machine-specific paths.
- The `main` branch is linked to the public GitHub repository.

One NetCDF test emits a binary-compatibility warning in the current local
environment. The test passes, but final numerical runs should use a clean
environment created from `requirements.txt`.

## Still data-dependent

- complete the corrected ERA5-Land accumulation download;
- rebuild the 1981–2023 master and labelled datasets;
- run the full multi-seed experiment and all ablations;
- run expanding-window validation;
- inspect final figures, tables, metadata, and spatial layers;
- choose a software licence and archive the exact analysis release.

Run the automated release audit with:

```bash
python scripts/check_publication_readiness.py
```
