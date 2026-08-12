# GloFAS-modelled high-flow prediction at Hardinge Bridge

This repository contains a reproducible retrospective workflow for predicting
GloFAS-modelled high-flow exceedance at Hardinge Bridge 1, 3, 5, and 7 days
ahead. It compares statistical, machine-learning, and sequence models under a
strictly chronological evaluation.

The default target is exceedance of a training-period threshold in
**GloFAS-modelled discharge**. This is a retrospective proxy study: it does not
claim independent validation against gauge-observed floods or performance in an
operational forecasting system.

## What the pipeline does

- combines ERA5-Land, NASA POWER, and GloFAS historical inputs;
- builds lagged, rolling, interaction, and seasonal features without using
  future information;
- compares climatology, persistence, GloFAS signal, logistic regression,
  random forest, gradient boosting, LSTM, GRU, CNN–LSTM, and attention-LSTM;
- calibrates probabilities and selects decision thresholds outside the test set;
- reports discrimination, calibration, event, uncertainty, and cost–loss metrics;
- supports fixed-holdout and expanding-window validation;
- records configuration, checksums, software versions, and run fingerprints.

## Data

| Source | Variables used | Role |
|---|---|---|
| ERA5-Land | temperature, soil moisture, precipitation, runoff | gridded hydro-meteorological inputs |
| NASA POWER | temperature, precipitation, humidity | point-scale atmospheric inputs |
| GloFAS historical | modelled river discharge | default proxy target and optional predictor |

Raw and processed data stay outside version control. See [DATASETS.md](DATASETS.md)
for access, provenance, and directory details.

## Setup

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Development tools are installed separately:

```bash
python -m pip install -r requirements-dev.txt
```

## Run the workflow

Configure CDS access first, then download the public inputs:

```bash
python scripts/download_era5_daily.py
python scripts/download_era5_accumulations.py
python scripts/download_glofas.py
```

The accumulation downloader decodes ERA5-Land 00 UTC values as totals for the
preceding day. It validates calendar coverage and repairs legacy files only
after a corrected replacement has passed validation.

Build the merged dataset and create the proxy labels:

```bash
python scripts/build_dataset.py --offline
python scripts/create_high_flow_labels.py --target-source glofas_proxy
```

Start with a short integration run:

```bash
python scripts/train_evaluate.py --smoke-test
```

Then run the full fixed-holdout and rolling-origin experiments:

```bash
python scripts/train_evaluate.py
python scripts/rolling_origin_validate.py
```

The main sensitivity checks are available through explicit output namespaces:

```bash
python scripts/train_evaluate.py --exclude-streamflow --output-namespace ablations/no_streamflow
python scripts/train_evaluate.py --include-streamflow --output-namespace ablations/with_streamflow
python scripts/train_evaluate.py --imbalance-strategy none --output-namespace ablations/imbalance_none
python scripts/train_evaluate.py --imbalance-strategy pos_weight --output-namespace ablations/imbalance_pos_weight
python scripts/train_evaluate.py --imbalance-strategy focal --output-namespace ablations/imbalance_focal
python scripts/train_evaluate.py --imbalance-strategy sampler --output-namespace ablations/imbalance_sampler
```

Check whether the code, data products, and analyses needed for a reproducible
release are present:

```bash
python scripts/check_publication_readiness.py
```

If fitted models and test predictions already exist, corrected diagnostics,
grouped Random Forest permutation importance, and all publication figures can
be rebuilt without training the models again:

```bash
python scripts/regenerate_analysis.py
```

For a saved run created with the earlier calibration policy, recalibrate its
stored probabilities and thresholds before regeneration:

```bash
python scripts/recalibrate_saved_main.py --bootstrap-iterations 1000
```

## Evaluation design

| Block | Dates | Purpose |
|---|---|---|
| Training | 1981-01-01 to 2013-12-31 | feature fitting and model estimation |
| Validation | 2014-01-01 to 2018-12-31 | calibration and threshold selection |
| Test | 2019-01-01 to 2023-12-31 | final untouched evaluation |

The validation sequences are divided chronologically: the first 65% fits the
sigmoid calibrator and the final 35% selects the F1 operating threshold, with a
minimum of 30 positives in each block when feasible. Primary events are
uninterrupted exceedance runs; bridging one or two negative days is reported as
a sensitivity analysis. Uncertainty uses calendar-year blocks, paired
comparisons include Holm adjustment, and leave-one-test-year-out ranges are
reported. Predictor-family permutation importance is evaluated on the late
validation block, never on the test set.

Neural models use an inner temporal holdout to select the stopping epoch and
learning-rate schedule. A fresh model is then fitted on the complete training
block for the selected number of epochs, so classical and neural models receive
the same training period.

The GloFAS-signal baseline uses current-day historical discharge. It is a
retrospective reference, not an issued GloFAS forecast.

Average precision (AP) is the primary ranking metric for the rare-event target.
Threshold-sensitive results use the validation-selected threshold stored for
each fitted run; thresholds are never averaged across prediction rows.

Throughout this repository, a forecast means a retrospective multi-horizon
prediction of the GloFAS-modelled proxy. It must not be described as a
gauge-validated flood forecast, an observed flood prediction, or an operational
warning-system result. The optional current-discharge experiment is a
same-product information-availability ablation, not independent validation.

## Rerun guide

| Change | Required rerun |
|---|---|
| Documentation only | tests and publication-readiness audit |
| Metric reporting, diagnostics, or figures | `python scripts/regenerate_analysis.py` |
| Features, target, split dates, model architecture, or training settings | main training, all affected ablations, rolling-origin validation, then analysis regeneration |
| Raw-data domain, variables, or temporal alignment | dataset build, labels, all experiments, then analysis regeneration |

## Outputs

Generated figures, tables, predictions, checkpoints, and metadata are written
under `outputs/`. They remain local by default; [outputs/README.md](outputs/README.md)
describes the layout.

## Repository layout

```text
configs/                     experiment configuration
data/                        local raw, processed, and optional observed data
outputs/                     generated results and provenance
scripts/                     command-line workflow stages
src/hardinge_high_flow/      reusable modelling and evaluation code
tests/                       automated regression tests
DATASETS.md                  data acquisition and provenance notes
```

## Limits

- GloFAS-derived labels do not independently establish real flood occurrence.
- ERA5-Land, NASA POWER, and GloFAS historical data are retrospective products.
- The configured ERA5-Land box (20.7-26.6° N, 88.0-92.6° E) represents a
  regional lower-basin domain. It neither covers the full upstream Ganges
  catchment nor uses catchment-area weighting. Consequently, the results cannot
  be interpreted as a complete upstream rainfall-runoff model.
- Operational claims require archived predictors that were available at each
  forecast issue time.

Code is available at
[github.com/asifahamed11/hardinge-glofas-high-flow-forecasting](https://github.com/asifahamed11/hardinge-glofas-high-flow-forecasting).
Dataset access restrictions and redistribution terms remain those of the
original providers.
