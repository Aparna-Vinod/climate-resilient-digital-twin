"""
Data helpers for the dashboard. No Streamlit code here, so they can be tested.

The dashboard only READS these files (it never trains a model):
  data/processed/borough_day.csv   made by src/bigdata_processing/process_data.py
  results/model_metrics.csv        columns: model, precision, recall, f1, roc_auc
  results/risk_table.csv           columns: date, borough, risk
  results/whatif_results.csv       columns: factor, borough, risk_baseline, risk_scenario
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 4            # flood day = flood_count >= 4 (decided from training data)
TRAIN_END = "2026-01-01"  # dates before this are the training period

FILES = {
    "borough_day": ("data/processed/borough_day.csv",
                    ["borough", "date", "precip_mm", "max_hourly_mm", "flood_count"]),
    "metrics": ("results/model_metrics.csv",
                ["model", "precision", "recall", "f1", "roc_auc"]),
    "risk": ("results/risk_table.csv", ["date", "borough", "risk"]),
    "whatif": ("results/whatif_results.csv",
               ["factor", "borough", "risk_baseline", "risk_scenario"]),
}


def load(name, root=ROOT):
    """Return the DataFrame, or None if the file does not exist yet.
    Raises ValueError with a clear message if columns are missing."""
    rel, required = FILES[name]
    path = Path(root) / rel
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{rel} is missing columns {missing}; found {list(df.columns)}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def with_label(df, threshold=THRESHOLD):
    out = df.copy()
    out["is_flood_day"] = (out["flood_count"] >= threshold).astype(int)
    return out


def flood_share_by_borough(df, threshold=THRESHOLD):
    d = with_label(df, threshold)
    return d.groupby("borough")["is_flood_day"].mean().sort_values(ascending=False)


def split_summary(df, threshold=THRESHOLD, train_end=TRAIN_END):
    d = with_label(df, threshold)
    train = d[d["date"] < pd.Timestamp(train_end)]
    test = d[d["date"] >= pd.Timestamp(train_end)]
    return {
        "rows": len(d),
        "days": d["date"].nunique(),
        "date_min": d["date"].min().date(),
        "date_max": d["date"].max().date(),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_flood_share": float(train["is_flood_day"].mean()),
        "test_flood_share": float(test["is_flood_day"].mean()),
    }


def risk_on_date(risk_df, date):
    d = risk_df[risk_df["date"] == pd.Timestamp(date)]
    return d.set_index("borough")["risk"].sort_index()


def whatif_for_factor(whatif_df, factor):
    d = whatif_df[whatif_df["factor"] == factor].set_index("borough").sort_index()
    d = d[["risk_baseline", "risk_scenario"]].copy()
    d["change"] = d["risk_scenario"] - d["risk_baseline"]
    return d