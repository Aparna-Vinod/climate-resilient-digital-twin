# Climate-Resilient Digital Twin (NYC Street Flooding)

M.Sc. capstone project: **A Cloud-Based Digital Twin for Climate-Resilient Urban Infrastructure Using Big Data Analytics.**

A prototype that estimates, for each New York City borough and each day, how likely a street-flooding
"flood day" is, given the rainfall. A digital twin keeps one state per borough and replays historical
rainfall day by day. A what-if tool asks the model how its risk would change if the rain were heavier,
and a dashboard shows everything.

> **Read this first (honest scope)**
> - The flood label is built from **NYC 311 complaints**. It is a **proxy** for flooding, not measured flooding.
> - The twin is a **replay model driven by historical rainfall**. It has **no live sensors** and is **not real-time**.
> - Rainfall comes from **one point (Central Park)** and is used for all five boroughs.
> - The data is **moderate in size** (about 29.7 thousand complaint records, 5,250 borough-day rows), not "massive".
> - **Cloud storage is not implemented.** It is listed as future work.
> - The risk value is an **uncalibrated score**, not a true probability (see Results).
> - What-if results are **model outputs, not predictions of reality**.

## Status

Done and tested (unless marked otherwise):

- [x] Repository and project structure
- [x] Data downloaded and documented (NYC 311 and Open-Meteo)
- [x] PySpark pipeline (local mode) that cleans and joins the data into `data/processed/borough_day.csv`
- [x] Exploratory analysis (`results/fig1` to `fig4`, `results/eda_summary.txt`)
- [x] Machine learning: rainfall baseline, logistic regression, random forest
- [x] Digital twin (replay of historical rainfall, one state per borough)
- [x] What-if rainfall scenarios (x1.0, x1.2, x1.5, x2.0)
- [x] Streamlit dashboard with four tabs
- [x] Automated tests (10 tests, all passing)
- [ ] Cloud storage: **not implemented, future work**
- [ ] Live sensors / real-time updating: **not implemented, future work**
- [ ] Final report: in progress

## Pipeline

```
NYC 311 complaints + Open-Meteo rainfall
        -> PySpark cleaning and joining
        -> data/processed/borough_day.csv   (5,250 rows)
        -> machine learning (baseline, logistic regression, random forest)
        -> digital twin (replay) -> results/risk_table.csv
        -> what-if scenarios      -> results/whatif_results.csv
        -> Streamlit dashboard (reads the result files)
```

## Data

| Source | What it gives | Notes |
|---|---|---|
| NYC 311 Service Requests (NYC Open Data, id `erm2-nwe9`) | Flood-related complaints: complaint type "Sewer" with four descriptors (street flooding, highway flooding, catch basin, manhole) | 29,704 records from 2024-01-01; licence not stated on the dataset page, so cite the source |
| Open-Meteo Historical Weather API | Daily and hourly precipitation for Central Park (40.78, -73.97) | 1,050 days, 2024-01-01 to 2026-11-15; CC BY 4.0, non-commercial use; reanalysis data, not a rain gauge |

**Processed table** (`data/processed/borough_day.csv`): 5,250 rows = 1,050 days x 5 boroughs, no missing values.
During cleaning 13 of 29,704 complaints were dropped for having no valid borough; no duplicates were found.
Columns: `borough, date, precip_mm, rain_mm, max_hourly_mm, wet_hours, precip_prev1, precip_3day,
precip_7day, month, day_of_week, flood_count, catch_basin_count, manhole_count`.
More detail is in `docs/dataset_documentation/dataset.md`.

## Modelling choices (decided before training)

- **Label:** a flood day is a borough-day with `flood_count >= 4` (street + highway flooding complaints).
  4 is the 90th percentile of the training data.
- **Split:** time-based, no shuffling. Train = dates before 2026-01-01 (3,655 rows, 10.8% flood days).
  Test = 2026 (1,595 rows, 9.5% flood days).
- **Features:** `precip_mm`, `precip_prev1`, `precip_3day`, `max_hourly_mm`, `month`, and `borough` (one-hot).
  `flood_count`, `catch_basin_count` and `manhole_count` are **not** used because they leak the label.
- **Tuning:** only on training data, with a 5-fold `TimeSeriesSplit` (validation always after the fitting data).
  The best model was chosen from these validation results before 2026 was looked at.
- **Models:** (1) baseline: flood if `precip_mm >= t`, with t = 10.0 mm chosen on training data; (2) logistic
  regression (scaled inputs, balanced class weights); (3) random forest (balanced class weights, `random_state=42`).

## Results (2026 test period)

| Model | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| Baseline (rain >= 10.0 mm) | 0.245 | 0.325 | 0.279 | 0.610 |
| Logistic regression (**best**) | 0.251 | 0.576 | 0.350 | 0.816 |
| Random forest | 0.240 | 0.616 | 0.346 | 0.821 |

How to read this:

- Both learned models rank flood days better than the rainfall rule and catch more of them, but precision stays
  near 0.25. Performance is **modest**, as expected: the correlation between complaint counts and same-day rain
  is only about 0.28.
- Logistic regression was chosen as the best model from the training validation (average precision 0.483 vs 0.471
  for the random forest). The two are close on 2026, so one is not claimed to be better than the other.
- The random forest numbers can differ slightly between scikit-learn versions even with a fixed seed; the table
  above is from the final run.
- **The risk is not a calibrated probability.** Balanced class weights inflate it: the 2026 average risk is 0.349
  while the actual flood-day share is 0.095. It still ranks days sensibly (actual flood share rises from 0.8% in
  the lowest risk band to 28.6% in the highest, see `results/calibration_2026.csv`). Read it as a relative risk score.
- Per-borough results (`results/metrics_by_borough.csv`) rest on few flood days for some boroughs in 2026
  (Bronx 7, Manhattan 10, Staten Island 9), so they are unreliable.

### Digital twin

`src/digital_twin/twin.py` keeps one state per borough (recent rainfall, last features, current risk) and replays
the rainfall series from 2024-01-01 to 2026-11-15, scoring each borough each day with the saved best model.
Output: `results/risk_table.csv` (columns `date, borough, risk`; 5,250 rows). **Rows before 2026 are in-sample**
(the model was trained on them); only the 2026 rows are out-of-sample.

### What-if scenarios

`src/scenario_analysis/whatif.py` multiplies the four rainfall inputs by 1.0, 1.2, 1.5 and 2.0 for every 2026 day,
re-scores with the saved model and averages per borough. Output: `results/whatif_results.csv`
(columns `factor, borough, risk_baseline, risk_scenario`). Average risk across all boroughs rises from 0.349
(x1.0) to 0.437 (x2.0). Larger factors push inputs beyond anything observed in 2024-2026 (maximum 65.6 mm per day,
16.1 mm per hour), so the model is **extrapolating**; the script prints a warning for each case.

## Project structure

```
data/processed/borough_day.csv     5,250-row table made by the PySpark job (committed)
data/raw/                          raw downloads (not committed; see .gitignore)
src/data_collection/               download_data.py
src/bigdata_processing/            process_data.py (PySpark), explore_data.py
src/machine_learning/              features.py, train_models.py
src/digital_twin/                  twin.py
src/scenario_analysis/             whatif.py
models/best_model.joblib           fitted scikit-learn Pipeline (preprocessing + logistic regression)
results/                           metrics, risk table, what-if results, figures
dashboard/                         app.py, dashboard_data.py (Streamlit; reads result files only)
tests/                             automated tests
docs/                              dataset documentation, report notes, evidence screenshots
```

## How to run

Install the packages (Python 3.10 or newer). On Windows, if `python` points to another program, use `py` instead:

```
python -m pip install -r requirements.txt
python -m pip install matplotlib joblib pytest
```

`borough_day.csv` is already in the repository, so **the modelling steps need no Java and no PySpark**.
Run them in this order from the project root:

```
python src/machine_learning/train_models.py     # models, metrics, models/best_model.joblib
python src/digital_twin/twin.py                 # results/risk_table.csv
python src/scenario_analysis/whatif.py          # results/whatif_results.csv
python -m pytest tests -v                       # 10 tests should pass
python -m streamlit run dashboard/app.py        # dashboard at http://localhost:8501
```

To rebuild the data from scratch (needs Java and PySpark; the raw files are not in the repository):

```
python src/data_collection/download_data.py
python src/bigdata_processing/process_data.py
python src/bigdata_processing/explore_data.py
```

## Limitations

- 311 complaints depend on reporting behaviour, not only on flooding.
- One rainfall point for the whole city; borough differences come only from the borough indicator.
- No drainage network, elevation, sewer capacity or land-use information is modelled.
- The twin is a replay of past data; there are no live sensors and no forecasts.
- Only one train/test split was used (2024-2025 / 2026); no confidence intervals were computed.
- Rainfall inputs are correlated, so individual model coefficients should not be read as separate effects.
- Parquet output was not produced on Windows (Hadoop winutils missing); CSV is used instead.

## Future work

- Cloud storage for the raw and processed data (not implemented).
- Live rainfall and sensor feeds, and a forecast-driven twin.
- Borough-specific rainfall, probability calibration, and richer infrastructure data.

## Team

- **Aparna V (Member 1):** data collection, PySpark processing, exploratory analysis, dataset documentation, dashboard, tests.
- **Archana T (Member 2):** machine learning, digital twin, what-if analysis, saved model, report notes for these parts, final report.
