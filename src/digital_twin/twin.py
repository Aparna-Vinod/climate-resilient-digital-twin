"""
Task 2: the digital twin (replay model) -> results/risk_table.csv

Run from the project root (after train_models.py):
    py src/digital_twin/twin.py

What this is
    A replay-based twin. It keeps ONE STATE PER BOROUGH (current risk, last rainfall features
    and the recent rainfall history) and updates it day by day by replaying the historical
    rainfall series from 2024-01-01 to 2026-11-15. Each day the borough's rainfall features
    are recomputed from the state, scored by the saved best model, and the new risk is stored.
    It has NO live sensors and is NOT real-time.

Important
    * Rows dated before 2026-01-01 are IN-SAMPLE: the model was trained on that period, so
      the risk there looks better than it would on unseen days. Only 2026 rows are out-of-sample.
    * risk = the model's predicted probability of a flood day (>= 4 complaints, a proxy label).
      It is a model output, not a measured flood.
"""
import sys
from collections import deque
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.machine_learning.features import (FEATURES, ROOT, TRAIN_END,  # noqa: E402
                                           load_data)

MODEL_FILE = ROOT / "models" / "best_model.joblib"
OUT_FILE = ROOT / "results" / "risk_table.csv"
START, END = "2024-01-01", "2026-11-15"


class BoroughTwin:
    """State of one borough: rainfall history, last features and current risk."""

    def __init__(self, borough):
        self.borough = borough
        self.history = deque(maxlen=3)   # last 3 daily rainfall totals, newest last
        self.last_features = None
        self.risk = None
        self.date = None

    def update(self, date, precip_mm, max_hourly_mm, model):
        """Take one day of rainfall, recompute features from the state, re-score."""
        prev1 = self.history[-1] if self.history else 0.0        # yesterday's rainfall
        self.history.append(float(precip_mm))
        self.last_features = {
            "precip_mm": float(precip_mm),
            "precip_prev1": prev1,
            "precip_3day": float(sum(self.history)),             # today + previous 2 days
            "max_hourly_mm": float(max_hourly_mm),
            "month": int(date.month),
            "borough": self.borough,
        }
        row = pd.DataFrame([self.last_features])[FEATURES]
        self.risk = float(model.predict_proba(row)[0, 1])
        self.date = date
        return self.risk


def run_replay(df, model):
    """Replay the rainfall series day by day. Returns the risk table and the final twin states."""
    df = df[(df["date"] >= START) & (df["date"] <= END)].sort_values(["date", "borough"])
    twins = {b: BoroughTwin(b) for b in sorted(df["borough"].unique())}
    records, mismatches = [], 0
    for date, day in df.groupby("date"):
        for _, r in day.iterrows():
            twin = twins[r["borough"]]
            risk = twin.update(date, r["precip_mm"], r["max_hourly_mm"], model)
            # consistency check: features rebuilt by the twin must equal the pipeline's columns
            f = twin.last_features
            if (abs(f["precip_prev1"] - r["precip_prev1"]) > 1e-6
                    or abs(f["precip_3day"] - r["precip_3day"]) > 1e-6):
                mismatches += 1
            records.append({"date": date.strftime("%Y-%m-%d"), "borough": r["borough"],
                            "risk": risk})
    return pd.DataFrame(records, columns=["date", "borough", "risk"]), twins, mismatches


def main():
    if not MODEL_FILE.exists():
        sys.exit("models/best_model.joblib not found. Run src/machine_learning/train_models.py first.")
    model = joblib.load(MODEL_FILE)
    df = load_data()

    risk, twins, mismatches = run_replay(df, model)
    OUT_FILE.parent.mkdir(exist_ok=True)
    risk.to_csv(OUT_FILE, index=False)

    n_days = risk["date"].nunique()
    in_sample = (pd.to_datetime(risk["date"]) < pd.Timestamp(TRAIN_END)).sum()
    print(f"Replayed {n_days} days x {len(twins)} boroughs = {len(risk)} rows "
          f"({risk['date'].min()} to {risk['date'].max()})")
    print(f"Feature check: rainfall features rebuilt by the twin differ from borough_day.csv "
          f"in {mismatches} rows (should be 0)")
    print(f"In-sample rows (before {TRAIN_END}, used for training): {in_sample}; "
          f"out-of-sample (2026): {len(risk) - in_sample}")
    print("Risk range:", round(risk["risk"].min(), 4), "to", round(risk["risk"].max(), 4))
    print("\nFinal twin state (last replayed day):")
    for b, t in twins.items():
        print(f"  {b:14s} {t.date.date()}  risk={t.risk:.3f}  rain={t.last_features['precip_mm']} mm")

    # Calibration check on 2026 (out-of-sample): is the risk a true probability?
    te = risk.assign(date=pd.to_datetime(risk["date"])).merge(
        df[["date", "borough", "is_flood_day"]], on=["date", "borough"])
    te = te[te["date"] >= pd.Timestamp(TRAIN_END)]
    te["risk_band"] = pd.cut(te["risk"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True,
                             labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    calib = (te.groupby("risk_band", observed=True)
             .agg(rows=("risk", "size"), mean_risk=("risk", "mean"),
                  actual_flood_share=("is_flood_day", "mean")).reset_index())
    calib["risk_band"] = calib["risk_band"].astype(str)
    calib.to_csv(OUT_FILE.parent / "calibration_2026.csv", index=False)
    print(f"\nCalibration check on 2026: mean risk {te['risk'].mean():.3f} vs actual flood-day "
          f"share {te['is_flood_day'].mean():.3f}")
    print(calib.round(3).to_string(index=False))
    print("-> The risk ranks days sensibly but is NOT a calibrated probability "
          "(class_weight='balanced' inflates it). Read it as a risk score.")
    print("\nNOTE: replay of historical rainfall, no live sensors; pre-2026 rows are in-sample.")
    print(f"Saved {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()