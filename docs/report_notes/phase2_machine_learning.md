# report notes - Phase 2: Machine learning models
Chapter: Methodology > Machine Learning Models; Results > Model Performance

Report-ready text (implemented and tested only)
A borough-day was labelled a flood day when it had at least four street or highway flooding complaints in the NYC 311 data (the 90th percentile of the training period). The label is a proxy for flooding built from citizen complaints; it is not measured flooding.

The data were split by time: dates before 1 January 2026 (3,655 borough-days, 10.8% flood days) were used for training and 2026 (1,595 borough-days, 9.5% flood days) for testing, without shuffling. Only rainfall and calendar information was used as input: same-day rainfall, previous-day rainfall, 3-day rainfall total, peak hourly rainfall, month, and borough (one-hot encoded). The complaint counts (flood, catch basin, manhole) were excluded because they define the label and would leak it.

Three models were compared. (1) A rule-based baseline that predicts a flood day when same-day rainfall is at least t mm, with t = 10.0 mm chosen as the value with the best F1 on the training data. (2) Logistic regression with standardised inputs and balanced class weights. (3) A random forest (300 trees, balanced class weights, random_state = 42). Hyper-parameters (logistic regression C; forest depth and minimum leaf size) were tuned with a 5-fold TimeSeriesSplit on the training dates only, using average precision, so validation data always came after the data used for fitting. The cut-off that turns a model score into a yes/no prediction was chosen as the value with the best F1 on the out-of-fold training scores. The 2026 data were used once, at the end, for reporting.

The best model was chosen before 2026 was evaluated, by the highest mean validation average precision: logistic regression (0.483, C = 0.1) against random forest (0.471, depth 5, minimum leaf 5). The two are close, so this choice is not a strong one.

Results on 2026 (computed by src/machine_learning/train_models.py):

Model	Precision	Recall	F1	ROC AUC
Baseline (precip_mm >= 10.0)	0.245	0.325	0.279	0.610
Logistic regression	0.251	0.576	0.350	0.816
Random forest	0.240	0.616	0.346	0.821

Both learned models rank flood days better than the rainfall rule (ROC AUC about 0.82 against 0.61) and catch more of them (recall 0.58 to 0.62 against 0.33), but precision stays near 0.25: about three of every four flagged borough-days did not reach four complaints. Performance is modest, as expected from the weak link between rainfall and complaint counts (correlation 0.28 with same-day rain). Logistic regression and random forest differ by less than 0.01 in F1, which is within what 151 positive test cases can resolve, so the report does not claim one is better than the other on the test data.

Confusion matrices on 2026 (rows actual, columns predicted; no flood / flood): baseline 1293, 151 / 102, 49; logistic regression 1185, 259 / 64, 87; random forest 1150, 294 / 58, 93.

Evidence
results/model_metrics.csv, results/metrics_by_borough.csv, results/feature_importances.csv
results/model_selection.json (parameters, validation scores, thresholds, selection rule)
results/fig5_confusion_matrices.png
models/best_model.joblib (scikit-learn Pipeline: preprocessing + logistic regression)
Terminal output of python src/machine_learning/train_models.py
Figures/tables

Table of 2026 metrics (above); fig5 confusion matrices; per-borough table; feature importance table (random forest impurity importances; logistic regression standardised coefficients).

Limitations to state
The label is 311 complaints, a proxy; complaint volume also depends on reporting behaviour.
One rainfall point (Central Park) is used for all boroughs, so the models cannot see differences in rainfall between boroughs; borough differences come only from the borough indicator.
Per-borough results rest on few flood days in 2026 (Bronx 7, Manhattan 10, Staten Island 9, Brooklyn 57, Queens 68), so Bronx, Manhattan and Staten Island figures are unreliable.
The logistic regression coefficient for peak hourly rainfall is slightly negative (-0.17) while same-day and 3-day rainfall are positive. The rainfall inputs are correlated, so individual coefficients should not be read as separate effects.
Class weights and the F1-based cut-off push the models towards recall, which lowers precision.
Only one train/test split (2024-2025 / 2026) was used; no confidence intervals were computed.

Commit message: Train baseline and ML models

Note: random forest results can differ slightly between scikit-learn versions even with a fixed random_state. The numbers above are from the final run on the submitting machine (Archana, scikit-learn 1.8.0); logistic regression results were identical on both machines.