# REPORT NOTES - Phase 3: Digital twin (replay model)
Chapter: Methodology > Digital Twin; Results > Twin Risk Table

Report-ready text (implemented and tested only)

The digital twin is a replay-based model driven by historical rainfall. It has no live sensors and is not a real-time system. For each of the five boroughs the twin keeps a state: the rainfall of the last three days, the most recent rainfall features and the current risk. The twin replays the daily rainfall series from 1 January 2024 to 15 November 2026 (1,050 days). Each day it updates every borough's state with that day's rainfall and peak hourly rainfall, recomputes the previous-day and 3-day rainfall features from the state, and scores the borough with the saved best model (logistic regression). The result is stored in results/risk_table.csv with one row per borough per day (5,250 rows; columns date, borough, risk).

The twin was checked in two ways. The previous-day and 3-day rainfall features it rebuilt from its own state matched the processed dataset in all 5,250 rows, and its risk values equalled scoring the dataset directly with the saved model (maximum difference about 3e-16).

Rows dated before 1 January 2026 (3,655 rows) are in-sample, because the model was trained on that period; only the 2026 rows (1,595) are out-of-sample.

The risk is the model's predicted probability of a flood day, but it is not a calibrated probability. Because the model was trained with balanced class weights, the 2026 average risk is 0.349 while the actual share of flood days in 2026 is 0.095. The risk does rank days usefully: the actual flood-day share rose from 0.8% in the 0.0-0.2 risk band to 28.6% in the 0.8-1.0 band (results/calibration_2026.csv):

Risk band	Rows (2026)	Mean risk	Actual flood-day share
0.0-0.2	730	0.096	0.008
0.2-0.4	255	0.325	0.067
0.4-0.6	287	0.520	0.157
0.6-0.8	176	0.686	0.233
0.8-1.0	147	0.910	0.286

The report therefore treats the risk as a relative risk score, not as the chance that a flood will happen.

Evidence
results/risk_table.csv (5,250 rows), results/calibration_2026.csv
Terminal output of python src/digital_twin/twin.py (state per borough on the last day, feature check = 0 mismatches, calibration table)
Dashboard screenshot: docs/report_notes/evidence/07_dashboard_twin.png
Figures/tables

Risk by borough for a chosen date (dashboard); risk over time per borough; calibration table.

Limitations to state
Replay of past rainfall only: no sensors, no forecasts, no live data, not real-time.
The state is simple (rainfall history and current risk); there is no drainage network, elevation or sewer model.
Pre-2026 rows are in-sample and look better than the model would on unseen days.
Risk is an uncalibrated score (see table); it overstates the chance of a flood day.
One rainfall point is used for the whole city, so on a given day all boroughs see the same rain and differ only through the borough effect.

Commit message: Add digital twin replay