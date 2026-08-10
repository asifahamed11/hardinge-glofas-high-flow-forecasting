# Generated outputs

The pipeline creates:

- `figures/`: PDF, EPS, 600-dpi PNG, and 600-dpi TIFF figures.
- `gis/`: study points and data-domain GeoJSON files.
- `metadata/`: provenance, thresholds, software versions, and experiment settings.
- `models/`: fitted scalers, classical estimators, and neural checkpoints.
- `predictions/`: date-aligned test predictions.
- `tables/`: detailed metrics, event performance, summaries, uncertainty
  intervals, paired comparisons, and grouped permutation importance.

`metadata/postprocessing.json` records the prediction hashes and regenerated
files produced by `scripts/regenerate_analysis.py`.

All default outputs evaluate a GloFAS-modelled high-flow proxy. They are not
evidence of gauge-validated flood occurrence or operational warning skill. The
configured ERA5-Land predictors are regional box averages rather than
full-upstream, catchment-weighted aggregates.

Generated outputs stay local by default. A final, immutable result set can be
archived separately and linked to the exact Git commit that produced it.
