# REPORT NOTES - Phase 5: Testing and dashboard check
Chapter: Implementation > Testing and Dashboard

## Report-ready text (implemented and tested only)

Automated tests (`python -m pytest tests -v`) check the processed table, the label rule, the
time-based split, and the format of the result files: the metrics file (values between 0 and 1),
the risk table (risk between 0 and 1, all five boroughs), the what-if file (factor 1.0 equals the
baseline; average risk does not fall as rainfall increases) and that the saved model loads. After
the model, twin and what-if scripts were run, all 10 tests passed and none were skipped. The three
result files were also checked for exact column names and row counts (3, 5,250 and 20 rows) and for
missing values (none).

The Streamlit dashboard (`python -m streamlit run dashboard/app.py`) reads the result files and has
four tabs: data overview, model results, twin risk by borough, and what-if rainfall. All four tabs
displayed data without errors. The dashboard shows a notice that the twin replays historical rainfall
with no live sensors and that labels are 311 complaints, a proxy for flooding.

## Evidence
- Output of `python -m pytest tests -v` (10 passed)
- docs/report_notes/evidence/05_dashboard_data_overview.png
- docs/report_notes/evidence/06_dashboard_model_results.png
- docs/report_notes/evidence/07_dashboard_twin.png
- docs/report_notes/evidence/08_dashboard_whatif.png

## Limitations to state
- The tests check file formats and simple consistency, not model quality.
- The dashboard only reads saved results; it does not train or score models, and there is no live
  data feed or cloud deployment (cloud storage is future work).

## Open items for Aparna (dashboard file, not edited by Archana)
- The what-if and model-results bar charts are stacked, so baseline + scenario bars add up to more
  than 1; side-by-side bars would be clearer.
- The twin tab caption calls the risk a "predicted probability"; it is an uncalibrated risk score.

Commit message: Check tests and dashboard with model outputs