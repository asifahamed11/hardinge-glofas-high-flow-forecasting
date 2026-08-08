# Generated outputs

The pipeline creates:

- `figures/`: PDF, EPS, 600-dpi PNG, and 600-dpi TIFF figures.
- `gis/`: study points and data-domain GeoJSON files.
- `metadata/`: provenance, thresholds, software versions, and experiment settings.
- `models/`: fitted scalers, classical estimators, and neural checkpoints.
- `predictions/`: date-aligned test predictions.
- `tables/`: detailed metrics, event performance, summaries, uncertainty
  intervals, and paired comparisons.

Generated outputs stay local by default. A final, immutable result set can be
archived separately and linked to the exact Git commit that produced it.
