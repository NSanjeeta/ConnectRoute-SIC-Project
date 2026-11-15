# ConnectRoute

**Signal-aware journey planning for route-level LTE analysis.**

ConnectRoute compares driving routes using estimated LTE signal strength,
travel time, distance, outage risk, and terrain context. It combines a
Streamlit route-planning demo with a reproducible Salzburg research pipeline
for LTE RSRP prediction.

> **Demo status:** the Karnataka/India route experience is an illustrative
> transfer. The route estimator uses Indian LTE tower records and a
> Salzburg-fitted distance model; it is not locally validated Karnataka
> coverage.

## What the demo includes

- Geocoded origin and destination search with OSRM driving alternatives.
- Interactive OpenStreetMap/Folium route map with signal-colored segments.
- Route priorities: **Balanced**, **Best signal**, **Lowest outage risk**,
  **Fastest**, and **Shortest**.
- Configurable weak-signal and outage thresholds.
- Estimated RSRP profile with optional uncertainty band.
- Terrain elevation and diffraction context.
- Coverage summary, weak-signal hotspots, and heuristic tower-site candidates.
- CSV route-profile and GeoJSON route-alternative exports.
- A research benchmark view for saved Salzburg spatial-validation outputs.

## Quick demo guide

### 1. Install

Python 3.11 or newer is required. From PowerShell:

```powershell
cd C:\Users\rmegh\OneDrive\Desktop\ConnectRoute_SIC
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
```

### 2. Prepare the India LTE subset

The raw tower snapshot is large and is intentionally not committed to Git.
After placing `cell_towers_2021.csv.xz` in `data\raw`, run:

```powershell
.\.venv\Scripts\python.exe -m connectroute.tower_ingest `
  --country-code 404 405 `
  --output data\processed\opencellid_india_lte_2021.parquet `
  --summary results\india_tower_ingest_summary.json
```

### 3. Start the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py `
  --server.headless true --server.port 8501
```

Open **http://localhost:8501**.

### 4. Try the demo

1. Keep the default `Bengaluru, Karnataka, India` and `Mysuru, Karnataka,
   India`, or enter two other places.
2. Select profile detail (`50 m`, `100 m`, or `200 m`).
3. Choose a route priority in **Map controls**.
4. Adjust weak and outage thresholds if needed.
5. Click **Explore routes**.
6. Select a route to inspect its map, metrics, signal profile, terrain tab,
   coverage intelligence, hotspots, planning suggestions, and downloads.

The first geocoding/route request needs internet access. Terrain context is
optional and may take longer the first time while a DEM tile is downloaded
and cached locally.

## Methodology in brief

```text
Raw measurements/towers/terrain
        ↓
Cleaning and spatial feature engineering
        ↓
Nearest-tower matching and propagation baselines
        ↓
Spatially buffered model training and validation
        ↓
Route sampling and RSRP estimation
        ↓
Route scoring, hotspot detection, and dashboard visualization
```

The research pipeline uses Salzburg drive-test measurements as the measured
RSRP target. It builds tower-distance, propagation, and terrain features,
trains comparison models, and evaluates them with contiguous spatial folds and
a 500 m buffer. The dashboard samples OSRM routes, matches points to eligible
Indian LTE records, and applies the illustrative route estimator.

## Where AI/ML is used

The AI/ML implementation is in
[`src/connectroute/models.py`](src/connectroute/models.py):

- **LightGBM** predicts RSRP directly and learns hybrid residual corrections.
- **XGBoost** provides a hybrid residual comparison model.
- **Optuna** tunes model hyperparameters inside training folds.
- **LightGBM quantile models** estimate lower, median, and upper signal ranges.
- **Conformal calibration** adjusts uncertainty intervals using held-out
  calibration folds.

The modeling stage writes predictions and summaries to
`data\processed\spatial_model_predictions.parquet` and
`results\model_summary.json`. The live Karnataka route demo primarily uses
the fitted Salzburg log-distance baseline plus Indian tower locations, rather
than claiming a fully India-trained validated ML model.

## Repository map

| Path | Purpose |
|---|---|
| `app\streamlit_app.py` | Streamlit dashboard and interactive route demo |
| `src\connectroute\dashboard.py` | Testable route scoring and coverage analytics |
| `src\connectroute\routing.py` | Geocoding, OSRM alternatives, and route sampling |
| `src\connectroute\route_predict.py` | Route signal estimation |
| `src\connectroute\models.py` | LightGBM/XGBoost/quantile modeling |
| `src\connectroute\route_terrain.py` | DEM preparation and terrain profiles |
| `experiments\run_pipeline.ps1` | Full reproducible research pipeline |
| `experiments\compare_terrain_features.py` | Buffered terrain-feature comparison |
| `tests\` | Automated unit and regression tests |

## Validation

Run the focused automated suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The dashboard analytics and route-scoring tests should pass before publishing
changes. Use `MPLBACKEND=Agg` in environments where Matplotlib otherwise
selects an unavailable GUI backend.

## Publishing checklist

Before pushing to GitHub:

1. Confirm `.env`, `.streamlit\secrets.toml`, raw downloads, processed data,
   and results are ignored.
2. Never commit API tokens, passwords, or personal credentials.
3. Run the tests and a Streamlit HTTP smoke check.
4. Review `git diff` and `git status`.
5. Commit source code, tests, documentation, and small metadata only.
6. Push to a repository you own or have permission to update.

The large datasets are excluded by `.gitignore`; a fresh clone must follow the
data preparation instructions above.

---

The detailed research scope, data provenance, modeling assumptions, and
pipeline commands follow below.

## Research scope and guardrails

- Predict measured LTE RSRP in dBm; do not train on classes derived from the
  same inputs.
- Validate research claims only against the Salzburg highway drive-test data.
- Use spatially grouped validation with a buffer; never use random point splits
  for reported results.
- Treat the Karnataka route as an illustrative, unvalidated app demonstration.
- Keep original downloads unchanged. Generate processed data and results with
  reproducible code, configuration, and recorded seeds.

## Project phases

1. Verify and document data sources, schemas, licenses, and capstone requirements.
2. Ingest measurements, calculate route chainage and spatial blocks, and match
   measurements to nearby LTE towers (S1-S3).
3. Add terrain profiles and propagation-model baselines (S4-S5).
4. Train hybrid and comparison models; evaluate spatially and calibrate
   uncertainty (S6-S8).
5. Build the route-ranking map and dashboard, labelled according to validation
   status (S9).
6. Produce reproducible paper figures, reports, and a documented release.

## Workspace setup

Python 3.11 or newer is required. From PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The active development setup uses the Python interpreter available on this
machine (Python 3.13). If PowerShell blocks activation, use the interpreter
directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### Get the data on a fresh clone

Large raw downloads, generated Parquet files, and analysis outputs are not
stored in Git. They are excluded by `.gitignore`; download the required source
files into `data\raw` on each machine. Review `data\raw\README.md` for dataset
attribution and license notes before redistribution.

For the full research pipeline, download these three inputs from their
documented sources:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/mherlich/wireless-data-set/main/SRFG-v1.csv" `
  -OutFile "data\raw\SRFG-v1.csv"
Invoke-WebRequest `
  -Uri "https://datasets.clickhouse.com/cell_towers.csv.xz" `
  -OutFile "data\raw\cell_towers_2021.csv.xz"
Invoke-WebRequest `
  -Uri "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E013_00_DEM/Copernicus_DSM_COG_10_N47_00_E013_00_DEM.tif" `
  -OutFile "data\raw\Copernicus_DSM_COG_10_N47_00_E013_00_DEM.tif"
```

The ClickHouse tower snapshot is about 719 MB; the Salzburg measurements are
about 47 MB and the DEM is about 42 MB. The route demo needs only the public
tower snapshot to create its India LTE subset. After downloading it, prepare
the subset with:

```powershell
.\.venv\Scripts\python.exe -m connectroute.tower_ingest `
  --country-code 404 405 `
  --output data\processed\opencellid_india_lte_2021.parquet `
  --summary results\india_tower_ingest_summary.json
```

After that, launch the app as described below. The Salzburg model benchmark
appears after the research pipeline has generated its processed prediction
files; it is not included in the clone.

Run the S1 cleaner and its tests:

```powershell
python -m connectroute.ingest
python -m connectroute.tower_ingest
python -m connectroute.geometry
python -m connectroute.towers
python -m connectroute.terrain
python -m connectroute.propagation
python -m connectroute.models --trials 2
python -m connectroute.evaluate
python -m pytest
```

To rerun the complete research pipeline from raw local inputs, use
`.\experiments\run_pipeline.ps1`. It runs the configured 50-trial S6/S7 search
and 2,000-replicate S8 block bootstrap by default. For a faster development
run, use `.\experiments\run_pipeline.ps1 -TuningTrials 2
-BootstrapReplicates 200`. Each stage stops the pipeline on failure.

Install and launch the route comparison prototype:

```powershell
python -m pip install -e ".[app]"
streamlit run app\streamlit_app.py
```

The app compares Nominatim-geocoded OSRM driving alternatives (Bengaluru–
Mysuru by default), samples routes at the selected spacing, and automatically
estimates RSRP at each route point from the nearest eligible LTE record in the
local India tower subset. It displays the signal profile and route map as soon
as signal estimates are ready. Optional elevation and terrain profiles can be
calculated from the Terrain tab; first use may take longer while route DEM data
is prepared. Generate the India subset from the downloaded snapshot with:

The dashboard also includes a route-priority scorecard (balanced, signal-first,
outage-first, fastest, or shortest), configurable weak/outage thresholds,
distance-weighted coverage metrics, signal-colored route segments, matched LTE
tower overlays, uncertainty-band visibility, weak-signal hotspot detection,
heuristic tower-site recommendations, terrain context, and CSV/GeoJSON exports.
The scorecard and tower recommendations are decision-support heuristics; they
do not replace a validated coverage model or radio-network design.

```powershell
.\.venv\Scripts\python.exe -m connectroute.tower_ingest --country-code 404 405 `
  --output data\processed\opencellid_india_lte_2021.parquet `
  --summary results\india_tower_ingest_summary.json
```

The route predictor is only an illustrative transfer: it applies a Salzburg-
fitted log-distance baseline to 2021 India LTE tower positions. Its empirical
Salzburg residual bands are not Karnataka-calibrated uncertainty intervals.
The app's expanded Salzburg model benchmark reads saved spatial out-of-fold
predictions and reports propagation-model errors separately from historical
ML development scores. The ML table is explicitly marked stale because it used
the previous 3GPP RMa LOS residual baseline and only two tuning trials; rerun
`.\experiments\run_pipeline.ps1` before treating it as a comparison with the
current fitted log-distance baseline. COST-231 Hata and rural Hata are also
shown with applicability rates; their current Salzburg best-server samples
have 0% in-domain coverage, so their RMSE values are extrapolations. These
benchmarks are not the Karnataka route predictions.
The app also profiles the route-to-nearest-tower DSM path at 30 m spacing and
displays elevation, Fresnel clearance, Deygout diffraction loss, terrain
roughness, and a 500 m terrain-position index (TPI). A TPI at or below -10 m is
labelled possible local low ground, not a confirmed valley. Copernicus GLO-30
tiles are read from their public cloud-optimized GeoTIFFs on first use; the
projected route-area DEM is cached under `data/processed/route_dems/`.
The terrain quantities are exploratory diagnostics and are not applied to
route RSRP predictions: adding them did not improve buffered spatial validation
on the Salzburg measurements. The app must continue to label the signal output
as unvalidated until suitable local drive-test measurements and a validated
local model are available.

To reproduce the initial terrain-feature fit comparison:

```powershell
.\.venv\Scripts\python.exe experiments\compare_terrain_features.py
```

This compares least-squares distance-only, distance-plus-profile-terrain, and
distance-plus-profile-terrain-plus-TPI candidates using five contiguous
Salzburg folds with 500 m training buffers. On the current data, RMSEs were
15.11 dB, 15.57 dB, and 17.52 dB respectively. This exploratory comparison is
not the final S6-S8 model search; it indicates that terrain and valley features
should not be forced into the current route signal estimate.

The S6/S7 model stage uses LightGBM for pure-ML and hybrid residual prediction,
with an XGBoost hybrid comparison. The hybrid residual is trained against the
fold-fitted nearest-tower log-distance baseline. Optuna tuning runs only within each
outer training partition; the outer spatial test fold is never used for tuning.
Use `connectroute-models` (or `python -m connectroute.models`) without
`--trials` to use the configured 50 trials per model. The two-trial command
above is a quick development run, not the final hyperparameter search.

For S7, each outer test fold has a different contiguous fold held out for
conformal calibration. That calibration fold is excluded from fitting and
tuning, with the configured 500 m buffer also applied around both held-out
areas. LightGBM quantile models estimate the 5th, 10th, 50th, 90th, and 95th
percentiles; conformalized quantile regression expands the central 80% and 90%
intervals. Outage probabilities are interpolated from those quantiles at the
preselected -110 dBm primary and -100 dBm secondary thresholds. They are
probability estimates, not a guarantee of calibration; Brier score and PR-AUC
are reported on the held-out spatial folds.

The pure-ML feature list excludes RSRP and other signal-derived measurements,
raw coordinates/chainage, trip and timestamp identifiers, cell/tower IDs, and
tower record age (the historical snapshot contains records newer than some
measurements). The hybrid adds path-loss and applicability features plus its
training-fold-calibrated RMa LOS baseline. Exact tower counts within 2 km/5 km
are not present in the three-nearest-tower table and are omitted rather than
approximated as density.

The raw Salzburg CSV is stored locally at `data/raw/SRFG-v1.csv` and is
intentionally excluded from version control. The cleaner writes a Parquet
product under `data/processed/` and an audit summary under `results/`.
The Austria LTE tower extractor reads the compressed global snapshot in chunks,
keeps the source reliability fields, and writes a compact local Parquet subset.
The geometry stage projects measurements to EPSG:32633, derives a median route
centerline, and assigns 2 km blocks and five contiguous spatial folds. The tower
stage matches the three nearest eligible LTE records and saves them alongside
the measurement, chainage, and fold columns in a single spatial feature Parquet
file. Tower distances, bearings, identifiers, reliability fields, and signed
record dates are preserved for later feature-selection and audit.
The terrain stage uses the Copernicus GLO-30 Public surface model to derive
30 m tower-to-receiver profiles, Fresnel clearance, Deygout diffraction loss,
and profile roughness for each candidate tower.
The propagation stage adds free-space, COST-231 Hata, rural Hata, and 3GPP RMa
and UMa LOS/NLOS path-loss features. It fits global power offsets and the
log-distance baseline independently within each buffered spatial training fold,
then writes held-out predictions and baseline metrics.
The modeling stage writes held-out point predictions, calibrated quantile
intervals, outage probabilities, feature lists, tuning parameters, and per-fold
metrics to `data/processed/spatial_model_predictions.parquet` and
`results/model_summary.json`.
The S8 evaluator creates point and 50 m chainage-bin metrics, per-fold tables,
outage reliability bins/plots, and paired 2 km block-bootstrap intervals under
`results/evaluation/`.

## Data provenance

- Salzburg drive-test measurements: `data/raw/SRFG-v1.csv`, downloaded from
  [mherlich/wireless-data-set](https://github.com/mherlich/wireless-data-set).
  Cite Farthofer et al., *IEEE Open Journal of the Communications Society*,
  2022, [doi:10.1109/OJCOMS.2022.3210289](https://doi.org/10.1109/OJCOMS.2022.3210289).
  The source repository's Apache-2.0 license covers its software; verify the
  dataset's own attribution/license terms from the dataset publication before
  redistributing the CSV.
- OpenCelliD tower data: obtain an Austria LTE export from
  [OpenCelliD downloads](https://opencellid.org/downloads.php) after reviewing
  its current terms and attribution requirements. Current country exports
  require an access token and cover a recent observation window, so they may not
  represent the 2018-2019 network. The public ClickHouse snapshot is about
  719 MB compressed and has been downloaded for historical matching.
- Terrain: Copernicus DEM GLO-30 Public, tile
  `N47_00_E013_00`, downloaded from the public AWS bucket. The GLO-30 product is
  a DSM (surface elevations include vegetation and buildings), not bare-earth
  terrain. Its source license, data description, and citation are documented in
  `data/raw/README.md`.

OpenCelliD data is licensed CC BY-SA 4.0 and requires visible attribution in
maps, reports, and other outputs that use it. See the linked attribution
guidance. Never put an OpenCelliD token, password, or other secret in source
files or chat. If API access is needed, configure the token in a local
environment variable named `OPENCELLID_API_TOKEN`; do not commit it.

## Current data check

The downloaded measurement file has 267,198 rows, all labelled LTE, and 22
columns. It contains GPS (`lat`, `long`), measured `rsrp`, timestamps, and a
`cell_id` field. There are 3,371 missing RSRP values, 2,944 missing cell IDs,
and no exact duplicate rows. RSRP median is -81 dBm. The ingestion stage drops
records with missing coordinates or RSRP, preserves cell IDs when present, and
records each filter count. It does not deduplicate observations.

The derived median centerline is 19.67 km long, represented by 751 vertices
from 25 m easting bins. It yields ten spatial blocks and five contiguous folds.
This observed extent is shorter than the plan's approximate 25 km description;
the pipeline records its measured extent rather than stretching or fabricating
the route.

The public tower snapshot contains 43,276,150 rows. The chunked extractor
selected 79,384 Austria LTE records with valid coordinates and retained tower
sample counts, ranges, and timestamps. A diagnostic comparison found 72 of the
88 measurement `cell_id` values among the tower `cell` values, covering 261,548
rows when the possible zero/sentinel ID is included. Excluding zero, 71 of 87
IDs overlap, covering 261,240 of 263,519 measurements with a nonzero ID. This is
not treated as a verified cell-to-tower join: the measurement file lacks the
full operator/area identifiers, and tower `cell` values are not unique on their
own. Tower assignment remains a spatial matching task, with cell-ID agreement
available only as a diagnostic.

Tower matching filters to at least five OpenCelliD samples and uses EPSG:32633.
The nearest eligible tower has a median separation of about 83 m (95th
percentile about 272 m); these are nearest-record distances, not validated
propagation distances. For about 110,000 measurements, the nearest tower
record's `created` date is later than the drive timestamp. The signed
`tower_1_created_age_at_measurement_days` field deliberately preserves that
fact; do not use it as an ordinary historical feature without handling the
2021 snapshot's temporal mismatch.

The same snapshot was filtered for MCC 404 and 405 for the route demo. That
India subset contains 359,817 valid LTE records; 71,823 have at least five
samples and are eligible for the default route estimator. It is a historical
location snapshot, not a current or complete coverage inventory. The route
estimator does not use a Karnataka DEM or any locally measured signal data.
It refuses routes outside the coordinate extent of eligible local tower
records rather than treating distant Indian towers as serving a route elsewhere.
Being inside that extent does not guarantee local coverage or make the transferred
Salzburg model valid.

The terrain stage projects the source GLO-30 DSM to a local 30 m EPSG:32633
GeoTIFF and samples all three tower-to-receiver paths. It uses a 30 m nominal
profile spacing, 1800 MHz Fresnel radius, assumed 30 m tower and 1.5 m receiver
heights, and a 4/3 effective-Earth-radius factor. The Copernicus DSM includes
vegetation/buildings; its obstruction estimates are therefore DSM-based and
should not be interpreted as bare-earth terrain-only losses.
The configured 30 m tower height is applied above the sampled DSM surface; this
is an explicit modeling assumption, not an observed antenna height.

The generated terrain feature table covers all 263,827 measurement rows. At
the default 1800 MHz and antenna heights, the nearest-tower Fresnel clearance
fraction has a median of 6.58; median Deygout diffraction loss is 0 dB, with a
95th percentile of 15.42 dB. These are engineering features from the stated
assumptions and DSM, not measured propagation losses or validation results.

The propagation baselines use the equations documented in COST 231 and 3GPP
TR 38.901 clause 7.4.1. Free-space attenuation follows ITU-R P.525. Hata and
COST-231 formula outputs are retained outside their published applicability
domains so comparisons can be computed, but every formula has explicit
per-tower and best-server validity flags. These extrapolated results must not be
presented as in-domain Hata accuracy. At the observed tower distances (all
below 1 km) COST-231 Hata is out of its 1-20 km distance range; the rural Hata
model is also out of range at the default 1800 MHz frequency. 3GPP scenario
validity is tracked separately.

Physics baselines are compared on the planned 5-fold contiguous spatial split.
Each fold's common RSRP offset is fitted only on its training rows after the
500 m buffer; the fitted log-distance regression is likewise trained only on
those rows. The output metrics are an initial baseline gate, not the complete
S8 evaluation or a result from random point splitting. Sources:
[COST 231 Final Report](https://www.lx.it.pt/cost231/final_report.htm),
[3GPP TR 38.901](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/17.00.00_60/tr_138901v170000p.pdf),
[ITU-R P.525](https://www.itu.int/rec/R-REC-P.525/en).

Initial 5-fold spatial baseline RMSEs are 15.72 dB (free space), 18.43 dB
(COST-231 Hata), 18.43 dB (rural Hata), 15.20 dB (3GPP RMa LOS), 17.00 dB
(3GPP RMa NLOS), 15.27 dB (3GPP UMa LOS), 17.03 dB (3GPP UMa NLOS), and
15.11 dB (fitted log-distance using the nearest tower). These are preliminary
baseline results, not final paper claims; Hata/COST scores are out-of-domain
extrapolations for this corridor. At fixed frequency, the rural-Hata correction
is a constant added to the urban Hata curve, so a separately fitted fold
intercept absorbs that difference and makes their calibrated predictions
identical. The held-out predictions are in
`data/processed/spatial_baseline_predictions.parquet`; model equations and
domain flags are in `data/processed/salzburg_propagation_features.parquet`,
with fold calibration details and per-fold metrics in
`results/propagation_summary.json`.

## S6/S7 development-run results (stale; prior baseline)

The results below were generated before correcting the hybrid residual
baseline. They used 3GPP RMa LOS, whereas the current implementation uses the
fold-fitted nearest-tower log-distance baseline. They are retained as
historical development artifacts only and must not be cited as results for the
current implementation. Rerun the pipeline before reporting updated S6-S8
metrics. The original run used two Optuna trials per model (the reproducible
default remains 50), 300 boosting iterations, seed 42, five outer spatial
folds, and three buffered inner folds. S7 assigns the next contiguous spatial
fold to calibration and never uses that fold for fitting or tuning.

| Model | Spatial-CV RMSE |
|---|---:|
| Calibrated 3GPP RMa LOS baseline | 15.20 dB |
| Pure LightGBM | 16.41 dB |
| LightGBM hybrid residual | 16.70 dB |
| XGBoost hybrid residual | 16.87 dB |
| LightGBM quantile median | 16.63 dB |

These historical two-trial results do not establish whether the current
implementation beats its fitted log-distance baseline. Run the configured
50-trial search and regenerate the evaluation before making a final model
comparison.
The RMa LOS formula is in-domain for 97.77% of best-server rows; the remaining
2.23% are retained with the propagation-stage validity flag and should be
treated as extrapolations.

The conformalized 80% interval covered 84.5% of observations (mean width
46.55 dB); the 90% interval covered 88.7% (mean width 58.77 dB). The 90%
coverage varied from 52.6% to 99.9% across outer spatial folds. This large
fold-to-fold variation and wide intervals show that aggregate coverage does
not establish reliable route-level uncertainty under spatial distribution
shift.

At the primary -110 dBm outage threshold, observed prevalence was 3.21%,
average precision was 0.0304, Brier score was 0.0387, and F1 at probability
0.5 was 0. The secondary -100 dBm threshold had 15.13% prevalence, average
precision 0.196, Brier score 0.145, and F1 0.0095 at 0.5. Treat these
interpolated probabilities as weak development estimates, not calibrated
outage-risk claims. Per-fold metrics and calibration adjustments are recorded
in `results/model_summary.json`; row-level predictions are in
`data/processed/spatial_model_predictions.parquet`.

The S8 evaluator reports point and 50 m chainage-bin metrics, per-fold tables,
and paired 2 km spatial block-bootstrap confidence intervals. It produced
394 occupied 50 m bins; bin-level RMSE was 14.05 dB for the RMa LOS baseline,
16.40 dB for pure LightGBM, 16.72 dB for the LightGBM hybrid, 16.83 dB for
XGBoost, and 16.52 dB for the quantile median. Candidate-minus-baseline
point-RMSE bootstrap intervals all included zero (pure LightGBM: +1.22 dB,
95% CI [-1.65, 3.70]; LightGBM hybrid: +1.51 dB, CI [-1.64, 4.50];
XGBoost: +1.68 dB, CI [-1.17, 4.40]; quantile median: +1.43 dB,
CI [-1.84, 4.97]). Tables, figures, and the generated report are under
`results/evaluation/`.

## Remaining research gates

The reproducible pipeline, initial S8 report, and S9 route-comparison
prototype are present, but this is not yet a completed publication/demo
release:

- The reported S6/S7 models use two Optuna trials as a development run, not the
  planned 50-trial final search. Use `.\experiments\run_pipeline.ps1` for the
  full pipeline when the machine can support the longer nested spatial search.
- Required ablations (terrain, physics, K=1..5, tower reliability, sample
  filter, antenna height, and frequency), the 20-seed tower-noise/removal study,
  and SHAP explanations have not been run. The current spatial feature table
  contains only three matched candidate towers, so K=4/5 also requires
  regenerating tower, terrain, and propagation features with K>=5.
- The app now automatically estimates a route profile from the 2021 India LTE
  subset using the Salzburg-fitted distance baseline. This is a demonstration
  only, not a validated Karnataka model; tower completeness and local
  propagation effects are unknown. Do not paste an OpenCelliD token into
  source or chat; configure it locally as `OPENCELLID_API_TOKEN` if a current
  export is later needed.
- The app reports marginal expected outage fraction and a thresholded
  high-risk stretch when predictions are supplied. It does not report the
  probability of any outage exceeding a distance, because that requires a
  defensible joint spatial predictive distribution, not independent point
  probabilities.
- A final paper/capstone release still needs the full experiment suite,
  independent review of dataset redistribution/attribution terms, and the
  institution's final rubric/deadline. No commit or publication package has
  been created.

The source timestamps have no timezone offset. They remain timezone-naive until
the dataset documentation establishes their time basis; ingestion does not
silently assume UTC.

## Repository layout

```text
config/       experiment defaults
data/raw/     original downloads (not versioned)
data/processed/ generated data (not versioned)
src/          project package
experiments/  scripts that generate research outputs
tests/        automated tests
app/          route demo
paper/        manuscript and figures
results/      generated outputs (not versioned)
```
