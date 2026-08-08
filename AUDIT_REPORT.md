# Technical audit

This project was reviewed for data leakage, target interpretation, acquisition
errors, reproducibility, and release safety.

## Corrections in the current workflow

- Credentials and machine-specific paths are excluded from tracked source.
- ERA5-Land precipitation and runoff use verified 00 UTC previous-day
  accumulations instead of sums of cumulative hourly values.
- GloFAS discharge is taken from one consistent grid cell near Hardinge Bridge.
- Missing-data handling is past-only and unresolved gaps stop the build.
- Feature scaling, extreme thresholds, and target thresholds are fitted on
  training data only.
- Sequences cannot cross training, validation, or test boundaries.
- Neural early stopping uses an inner temporal block, followed by a fresh fit on
  the complete training period.
- Comparisons are date-aligned and seed-matched against persistence and logistic
  regression.
- Inputs, configuration, labels, predictions, and results carry checksums or run
  fingerprints where appropriate.
- Download archives are validated, extracted safely, and moved atomically.

## Scientific boundaries

The default label is a GloFAS-modelled high-flow proxy. It should not be
described as an independently observed flood record. The experiment is also
retrospective because its predictors are historical products rather than
archived forecasts available at issue time.

The configured ERA5-Land domain is a weighted geographic average, not an
upstream catchment mask. A catchment-based aggregation or external station would
strengthen spatial validation when those data are available.

## Release checks

Before archiving a result set:

1. rebuild the dataset from verified accumulation files;
2. recreate labels after every master-dataset change;
3. run fixed-holdout, ablation, and rolling-origin experiments;
4. compare reported values directly with generated tables;
5. keep credentials, restricted data, and local environments outside Git;
6. tag the exact commit used for the final analysis.
