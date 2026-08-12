# Study-area map sources

The files in this directory are downloaded locally and are not committed to
Git. Recreate them with:

```powershell
python scripts/download_study_area_map_data.py
```

The publication map uses:

- Natural Earth 1:10m Admin-0 countries, version 5.1.1 (public domain);
- HydroBASINS Asia level 6, version 1c;
- HydroRIVERS Asia, version 1.0.

HydroBASINS and HydroRIVERS require acknowledgement and citation of Lehner and
Grill (2013), https://doi.org/10.1002/hyp.9740.

The map-generation script derives the upstream catchment that drains through
the selected GloFAS grid point near Hardinge Bridge. It does not use older
locally held GADM files whose licence restricts commercial use and
redistribution.
