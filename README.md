# Hardinge Bridge high-flow forecasting

This repository rebuilds the original CRC Press chapter code as a portable, leakage-aware research workflow for a possible *Natural Hazards* journal paper.

The default experiment predicts exceedance of a training-period high-flow threshold in **GloFAS modelled discharge**. It does not claim to predict an independently observed flood. The preferred journal analysis uses quality-controlled BWDB/FFWC observations through the supported `observed` target mode.

## What changed

- Removed the exposed CDS credential and all machine-specific paths.
- Replaced stateful notebook code with configuration-driven scripts and modules.
- Changed the spatial GloFAS operation from a one-degree mean to the nearest configured station grid cell.
- Normalized temperature, precipitation, runoff, humidity, and discharge units.
- Fitted every threshold, scaler, and extreme indicator using training data only.
- Kept train, validation, and test sequences inside their own chronological blocks.
- Added 1-, 3-, 5-, and 7-day horizons.
- Added climatology, persistence, current-day GloFAS-signal,
  logistic-regression, random-forest, and gradient-boosting baselines.
- Renamed the former “ConvLSTM” correctly as CNN–LSTM.
- Prevented focal loss, class weighting, and weighted sampling from being combined unintentionally.
- Added multiple seeds, calibration, Brier score, reliability error, event metrics, year-block confidence intervals, and paired bootstrap comparisons.
- Added expanding-window rolling-origin validation.
- Decode ERA5-Land accumulated variables from the official 00 UTC
  previous-day totals instead of summing cumulative hourly fields.
- Refit neural models on the complete training block after temporal
  early-stopping selection, matching the data available to classical models.
- Added publication figures in PDF, EPS, 600-dpi PNG, and 600-dpi TIFF.
- Added tests, pinned dependencies, provenance metadata, and resumable downloads.

The detailed file audit is in `AUDIT_REPORT.md`; exact data instructions are
in `DATASETS.md`; completed and environment-dependent checks are separated in
`VERIFICATION.md`.

## Installation

Use Python 3.10, 3.11, or 3.12.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the pinned environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For Jupyter and development checks:

```bash
python -m pip install -r requirements-dev.txt
```

For a CUDA-enabled GPU, follow the official PyTorch installation selector for the required CUDA build, then install the remaining requirements. CPU execution is supported but the full neural experiment will be slow.

## Reproduce the default proxy experiment

First configure CDS access and accept the dataset terms as described in `DATASETS.md`.

Inspect a one-year download plan:

```bash
python scripts/download_era5_daily.py --start-year 1981 --end-year 1981 --dry-run
python scripts/download_era5_accumulations.py --start-year 1981 --end-year 1981 --dry-run
python scripts/download_glofas.py --start-year 1981 --end-year 1981 --dry-run
```

Download and preprocess:

```bash
python scripts/download_era5_daily.py
python scripts/download_era5_accumulations.py
python scripts/download_glofas.py
python scripts/build_dataset.py --offline
python scripts/create_high_flow_labels.py --target-source glofas_proxy
```

Run the fast integration experiment before the full study:

```bash
python scripts/train_evaluate.py --smoke-test
```

Run the configured final holdout experiment:

```bash
python scripts/train_evaluate.py
```

The same preprocessing sequence can be run with:

```bash
python scripts/run_pipeline.py --download --smoke-test
```

## Use independent observations

Obtain and organize the BWDB/FFWC series exactly as described in `DATASETS.md`, then run:

```bash
python scripts/create_high_flow_labels.py --target-source observed
python scripts/train_evaluate.py --output-namespace observed_target
```

No proxy fallback occurs if the observed file is absent or malformed.

## Required sensitivity analyses

Run the streamflow-input ablation without overwriting either result:

```bash
python scripts/train_evaluate.py \
  --exclude-streamflow \
  --output-namespace ablations/no_streamflow

python scripts/train_evaluate.py \
  --include-streamflow \
  --output-namespace ablations/with_streamflow
```

Run each neural-model imbalance strategy separately. Classical baselines retain
their documented native class-balancing settings in every comparison:

```bash
python scripts/train_evaluate.py \
  --imbalance-strategy none \
  --output-namespace ablations/imbalance_none

python scripts/train_evaluate.py \
  --imbalance-strategy pos_weight \
  --output-namespace ablations/imbalance_pos_weight

python scripts/train_evaluate.py \
  --imbalance-strategy focal \
  --output-namespace ablations/imbalance_focal

python scripts/train_evaluate.py \
  --imbalance-strategy sampler \
  --output-namespace ablations/imbalance_sampler
```

Run expanding-window validation:

```bash
python scripts/rolling_origin_validate.py --smoke-test
python scripts/rolling_origin_validate.py
```

Before treating any artifacts as submission-ready, run the fail-fast release
audit.  It checks corrected accumulation provenance, causal missing-data
handling, an independently observed target, every required ablation,
rolling-origin outputs, immutable version control, licensing, and credential
hygiene:

```bash
python scripts/check_publication_readiness.py
```

The audit is intentionally expected to fail until all scientific analyses and
release tasks have actually been completed.

If independently observed BWDB/FFWC discharge cannot be obtained, the project
can instead support a strictly declared GloFAS-proxy paper:

```bash
python scripts/check_publication_readiness.py --allow-proxy-target
```

Passing this mode does not validate claims about observed floods. The title,
abstract, methods, results, and conclusions must consistently describe the
target as GloFAS-modelled high-flow exceedance and the experiment as a
retrospective proxy study.

The rolling folds refit the high-flow threshold using each fold’s expanding training period and store fold-specific thresholds.

## Evaluation design

The fixed holdout design is:

| Block | Dates | Use |
|---|---|---|
| Training | 1981-01-01 to 2013-12-31 | Feature fitting and model estimation |
| Validation | 2014-01-01 to 2018-12-31 | Probability calibration and threshold selection |
| Test | 2019-01-01 to 2023-12-31 | Final untouched evaluation |

Neural training reserves the latest part of the training block only to select the learning-rate schedule and stopping epoch. A fresh network is then refit for that selected number of epochs on the complete training block, so neural and conventional models receive the same training period. The validation block is divided chronologically into separate calibration and threshold-selection portions. The test set is not used to fit preprocessing, labels, models, calibration, or decision thresholds.

Model settings are pre-specified in `configs/default.yaml` and copied into the
experiment metadata. The repository does not claim an undocumented
hyperparameter search. If the authors add tuning, candidate ranges must be
declared in advance and selected inside the training/validation structure of
each temporal fold, never against the final test block.

Reported outputs include:

- accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, critical success index, and false-alarm ratio;
- PR-AUC and ROC-AUC;
- Brier score and expected calibration error;
- event detection, missed events, false-alarm events, and onset delay;
- per-event peak detection and warning-issue lead time;
- cost–loss relative value across configured decision ratios;
- performance by test year and positive-flow magnitude group;
- year-block bootstrap confidence intervals;
- seed-matched paired block-bootstrap differences against persistence and
  logistic regression;
- date-aligned predictions for independent checking.

The `glofas_signal` baseline converts current-day historical GloFAS discharge
to an empirical training-distribution score. It is a retrospective signal
baseline, not a benchmark against an issued GloFAS forecast. A genuinely
operational comparison requires archived GloFAS forecasts or reforecasts.

## Journal figures

Figures follow the current
[*Discover Hazards* submission guidance](https://link.springer.com/journal/44475/submission-guidelines):

- 84 mm single-column and 174 mm double-column widths;
- 8-point sans-serif lettering;
- no embedded figure title or caption;
- color-vision-safe colors plus distinct line styles and markers;
- vector PDF and EPS;
- RGB PNG and LZW-compressed TIFF at 600 dpi;
- numbered filenames beginning with `Fig`.

The code creates:

1. skill versus forecast horizon;
2. precision–recall curves;
3. reliability diagrams;
4. a representative event-year probability timeline;
5. confusion matrices.

## QGIS study layers

Run `scripts/create_qgis_layers.py` with the QGIS Python interpreter. It:

- reads the same study coordinates and color palette;
- replaces previously generated layers instead of duplicating them;
- creates categorized point and domain layers;
- exports portable GeoJSON files to `outputs/gis/`.

QGIS itself is intentionally not listed as a pip dependency.

## Project structure

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── external/observations/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── workflow.ipynb
├── outputs/
├── scripts/
│   ├── build_dataset.py
│   ├── create_high_flow_labels.py
│   ├── create_qgis_layers.py
│   ├── download_era5_daily.py
│   ├── download_era5_accumulations.py
│   ├── download_glofas.py
│   ├── rolling_origin_validate.py
│   ├── run_pipeline.py
│   └── train_evaluate.py
├── src/hardinge_high_flow/
│   ├── config.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── features.py
│   ├── models.py
│   ├── plotting.py
│   └── validation.py
├── tests/
├── DATASETS.md
├── requirements.txt
└── pyproject.toml
```

## Mapping from the supplied files

| Supplied file | Revised location |
|---|---|
| `dataset.ipynb` | `notebooks/workflow.ipynb` |
| `Untitled-0.py` | `scripts/create_qgis_layers.py` |
| `code_1.text` | `scripts/download_era5_daily.py` |
| `code_2.text` | `scripts/download_era5_accumulations.py` |
| `code_3.text` | `scripts/download_glofas.py` |
| `code_4.text` | `scripts/build_dataset.py` |
| `code_5.text` | `scripts/create_high_flow_labels.py` |
| `code_6.text` | `scripts/train_evaluate.py` plus tested package modules |

## Verification

After installing the development dependencies:

```bash
python -m compileall -q src scripts
python -m pytest
python scripts/download_era5_daily.py --start-year 1981 --end-year 1981 --dry-run
python scripts/download_era5_accumulations.py --start-year 1981 --end-year 1981 --dry-run
python scripts/download_glofas.py --start-year 1981 --end-year 1981 --dry-run
python scripts/train_evaluate.py --smoke-test
```

The first four checks do not download scientific data. The last check requires
the processed labeled dataset. See `VERIFICATION.md` for the exact checks
completed on this revision and the checks that still require the real data,
PyTorch, or QGIS.

## Interpretation limits

Even with correct code, a GloFAS-derived label does not independently validate real flooding. Reanalysis variables are also retrospective products; a genuinely operational multi-day system should replace them with predictors available at forecast issue time, such as meteorological forecasts or reforecasts. Claims in a manuscript must match the selected target, station coverage, and actual lead time.

Before making the repository public, the authors should add the agreed licence, complete author metadata, repository URL, and archived release DOI.
