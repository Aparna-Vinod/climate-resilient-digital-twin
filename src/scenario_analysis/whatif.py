"""
Task 3: what-if rainfall scenarios -> results/whatif_results.csv

Run from the project root (after train_models.py):
    py src/scenario_analysis/whatif.py

Method
    For each factor in 1.0, 1.2, 1.5, 2.0: multiply precip_mm, precip_prev1, precip_3day and
    max_hourly_mm by the factor, keep month and borough unchanged, score every 2026 borough-day
    with the saved best model, and average the risk over the 2026 dates for each borough.
    risk_baseline is the same average at factor 1.0, so factor 1.0 gives
    risk_scenario == risk_baseline.

Important
    These are MODEL OUTPUTS for "what would the model say if the same days had more rain".
    They are not predictions of reality. Scaled rainfall that is larger than anything observed
    in 2024-2026 is extrapolation (a warning is printed). The risk is an uncalibrated score
    (see results/calibration_2026.csv), so compare scenarios with each other, not with the
    actual 9.5% flood-day rate.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.machine_learning.features import (FEATURES, RAIN_FEATURES, ROOT,  # noqa: E402
                                           load_data, split_train_test)

MODEL_FILE = ROOT / "models" / "best_model.joblib"
OUT_FILE = ROOT / "results" / "whatif_results.csv"
FACTORS = [1.0, 1.2, 1.5, 2.0]


def average_risk(model, test, factor):
    """Mean risk per borough over the 2026 dates after scaling the rainfall features."""
    scaled = test[FEATURES].copy()
    scaled[RAIN_FEATURES] = scaled[RAIN_FEATURES] * factor
    risk = pd.Series(model.predict_proba(scaled)[:, 1], index=test.index)
    return risk.groupby(test["borough"]).mean()


def extrapolation_warnings(all_data, test, factor):
    """Which scaled inputs go beyond the largest value observed in 2024-2026?"""
    msgs = []
    for col in RAIN_FEATURES:
        observed_max = all_data[col].max()
        scaled_max = test[col].max() * factor
        if scaled_max > observed_max + 1e-9:
            n = int((test[col] * factor > observed_max + 1e-9).sum())
            msgs.append(f"{col}: scaled maximum {scaled_max:.1f} exceeds observed maximum "
                        f"{observed_max:.1f} ({n} borough-days)")
    return msgs


def main():
    if not MODEL_FILE.exists():
        sys.exit("models/best_model.joblib not found. Run src/machine_learning/train_models.py first.")
    model = joblib.load(MODEL_FILE)
    df = load_data()
    _, test = split_train_test(df)
    print(f"Scenarios on {test['date'].nunique()} test dates in 2026 "
          f"({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"Observed maxima 2024-2026: {df['precip_mm'].max()} mm/day, "
          f"{df['max_hourly_mm'].max()} mm/hour\n")

    baseline = average_risk(model, test, 1.0)
    rows = []
    for factor in FACTORS:
        scenario = baseline if factor == 1.0 else average_risk(model, test, factor)
        for b in baseline.index:
            rows.append({"factor": factor, "borough": b,
                         "risk_baseline": float(baseline[b]), "risk_scenario": float(scenario[b])})
        for msg in extrapolation_warnings(df, test, factor):
            print(f"WARNING (factor {factor}): extrapolation, {msg}")
    out = pd.DataFrame(rows, columns=["factor", "borough", "risk_baseline", "risk_scenario"])
    out.to_csv(OUT_FILE, index=False)

    check = out[out["factor"] == 1.0]
    assert (check["risk_baseline"] - check["risk_scenario"]).abs().max() < 1e-9
    print("\nAverage 2026 risk (model output) by factor and borough:")
    print(out.pivot(index="borough", columns="factor", values="risk_scenario").round(3).to_string())
    print("\nAll-borough mean by factor:")
    print(out.groupby("factor")["risk_scenario"].mean().round(4).to_string())
    print("\nNOTE: model outputs, not predictions of reality; risk is an uncalibrated score.")
    print(f"Saved {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()