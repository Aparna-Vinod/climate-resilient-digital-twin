# # REPORT NOTES - Phase 4: What-if rainfall scenarios
Chapter: Methodology > Scenario Analysis; Results > What-if Analysis

## Report-ready text (implemented and tested only)

The what-if analysis asks what the model would output if the 2026 rainfall had been heavier.
For rainfall multipliers of 1.0, 1.2, 1.5 and 2.0, the four rainfall inputs (same-day rainfall,
previous-day rainfall, 3-day total and peak hourly rainfall) of every 2026 borough-day were
multiplied by the factor, month and borough were kept, the saved best model scored each row,
and the risk was averaged over the 2026 dates for each borough. The result is stored in
`results/whatif_results.csv` (20 rows). At factor 1.0 the scenario equals the baseline.

Average modelled risk (a model score, see Phase 3 calibration note):

| Borough | x1.0 | x1.2 | x1.5 | x2.0 |
|---|---|---|---|---|
| Bronx | 0.136 | 0.158 | 0.189 | 0.235 |
| Brooklyn | 0.517 | 0.537 | 0.562 | 0.597 |
| Manhattan | 0.160 | 0.183 | 0.214 | 0.261 |
| Queens | 0.675 | 0.690 | 0.708 | 0.732 |
| Staten Island | 0.256 | 0.281 | 0.313 | 0.360 |
| All boroughs (mean) | 0.349 | 0.370 | 0.397 | 0.437 |

Risk rises with the rainfall multiplier in every borough. The increase is largest in boroughs
that start with low risk (Bronx +0.099, Manhattan +0.101 at x2.0) and smaller where the score is
already high (Queens +0.057), as expected from a logistic model whose score flattens at high
values.

Larger factors push inputs beyond anything observed in 2024-2026 (maxima: 65.6 mm per day,
16.1 mm per hour, 96.1 mm over 3 days). At x1.2, 5 borough-days exceed the daily maximum; at
x1.5, 15 borough-days exceed it and 20 exceed the hourly maximum; at x2.0, 30 exceed the daily
maximum, 50 the hourly maximum and 70 the 3-day maximum. The script prints these warnings. In
those cases the model is extrapolating beyond its training data.

These are model outputs, not predictions of reality. They show how the fitted model responds to
heavier rainfall; they do not say how many floods or complaints would happen.

## Evidence
- results/whatif_results.csv
- Terminal output of `python src/scenario_analysis/whatif.py` (extrapolation warnings, table)
- Dashboard screenshot: docs/report_notes/evidence/08_dashboard_whatif.png

## Figures/tables
Scenario table (above); dashboard bar chart of baseline vs scenario.

## Limitations to state
- Scaling all rainfall inputs by the same factor is a simple assumption; real heavier storms
  change duration, intensity and timing differently.
- Extrapolation beyond observed rainfall for factors above 1.0 (counts above).
- Risk is an uncalibrated score; compare scenarios with each other, not with the 9.5% actual
  flood-day share.
- The logistic regression has a slightly negative coefficient on peak hourly rainfall, so the
  response to the four scaled inputs is the net effect of all of them, not a clean
  "rain causes risk" relationship.
- No drainage capacity, infrastructure or land-use changes are modelled.

Commit message: Add what-if scenarios