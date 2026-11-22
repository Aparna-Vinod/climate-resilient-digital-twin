"""
Shared settings and helpers for the machine-learning, twin and what-if scripts.
Keeping them in one place guarantees that every script uses the same label rule,
the same split and the same features.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "borough_day.csv"

THRESHOLD = 4              # flood day = flood_count >= 4 (decided earlier, do not change)
TRAIN_END = "2026-01-01"   # train: dates before this; test: 2026

# Allowed features only. flood_count, catch_basin_count and manhole_count are NOT used
# because they are (part of) the label and would leak it.
RAIN_FEATURES = ["precip_mm", "precip_prev1", "precip_3day", "max_hourly_mm"]
NUMERIC_FEATURES = RAIN_FEATURES + ["month"]
CATEGORICAL_FEATURES = ["borough"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    """Read borough_day.csv, add the label column and sort by date."""
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["is_flood_day"] = (df["flood_count"] >= THRESHOLD).astype(int)
    return df.sort_values(["date", "borough"]).reset_index(drop=True)


def split_train_test(df):
    """Time-based split, no shuffling."""
    train = df[df["date"] < pd.Timestamp(TRAIN_END)].copy()
    test = df[df["date"] >= pd.Timestamp(TRAIN_END)].copy()
    return train, test