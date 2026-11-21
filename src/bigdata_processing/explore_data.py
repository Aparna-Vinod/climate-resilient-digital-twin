"""
Exploratory analysis of data/processed/borough_day.csv for the report.
Saves figures to results/ and a text summary to results/eda_summary.txt.

Run from the project root:
    pip install matplotlib
    python src/bigdata_processing/explore_data.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN = Path("data") / "processed" / "borough_day.csv"
OUT = Path("results")
TRAIN_END = "2026-01-01"
THRESHOLD = 4  # flood day = flood_count >= 4 (90th percentile of training data)

OUT.mkdir(exist_ok=True)
df = pd.read_csv(IN, parse_dates=["date"])
df["is_flood_day"] = (df["flood_count"] >= THRESHOLD).astype(int)
train = df[df["date"] < TRAIN_END]
test = df[df["date"] >= TRAIN_END]

lines = []
lines.append(f"Rows: {len(df)}  | days: {df['date'].nunique()}  | boroughs: {df['borough'].nunique()}")
lines.append(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
lines.append(f"Train rows: {len(train)}  Test rows: {len(test)}")
lines.append(f"Flood-day share (flood_count >= {THRESHOLD}): train {train['is_flood_day'].mean():.3f}, test {test['is_flood_day'].mean():.3f}")
lines.append("")
lines.append("Flood days by borough (count and share of its days):")
g = df.groupby("borough")["is_flood_day"].agg(["sum", "mean"]).round(3)
lines.append(g.to_string())
lines.append("")
lines.append("Missing values per column:")
lines.append(df.isna().sum().to_string())
lines.append("")
lines.append("Rainfall (mm) summary:")
lines.append(df.drop_duplicates("date")[["precip_mm", "max_hourly_mm"]].describe().round(2).to_string())
(OUT / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# Figure 1: complaints per borough
fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("borough")["flood_count"].sum().sort_values().plot.barh(ax=ax, color="#3b6ea5")
ax.set_xlabel("Street/highway flooding complaints (2024 to Nov 2026)")
ax.set_ylabel("")
ax.set_title("311 flood complaints by borough")
fig.tight_layout()
fig.savefig(OUT / "fig1_complaints_by_borough.png", dpi=200)
plt.close(fig)

# Figure 2: citywide daily complaints vs rainfall
daily = df.groupby("date").agg(complaints=("flood_count", "sum"), rain=("precip_mm", "first"))
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(daily.index, daily["rain"], color="#9ecae1", width=1.0, label="Rainfall (mm)")
ax1.set_ylabel("Daily rainfall (mm)")
ax2 = ax1.twinx()
ax2.plot(daily.index, daily["complaints"], color="#c0392b", lw=0.8, label="Complaints")
ax2.set_ylabel("Flood complaints per day (all boroughs)")
ax1.set_title("Daily rainfall and 311 flood complaints, New York City")
fig.tight_layout()
fig.savefig(OUT / "fig2_rainfall_vs_complaints.png", dpi=200)
plt.close(fig)

# Figure 3: scatter, same-day rain vs complaints (citywide)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(daily["rain"], daily["complaints"], s=8, alpha=0.5, color="#2c3e50")
ax.set_xlabel("Daily rainfall (mm)")
ax.set_ylabel("Flood complaints per day (all boroughs)")
ax.set_title(f"Rainfall vs complaints (r = {daily['rain'].corr(daily['complaints']):.2f})")
fig.tight_layout()
fig.savefig(OUT / "fig3_rain_scatter.png", dpi=200)
plt.close(fig)

# Figure 4: distribution of flood_count in training period
fig, ax = plt.subplots(figsize=(6, 4))
vc = train["flood_count"].clip(upper=15).value_counts().sort_index()
ax.bar(vc.index, vc.values, color="#3b6ea5")
ax.axvline(THRESHOLD - 0.5, color="#c0392b", ls="--", label=f"flood day threshold (>= {THRESHOLD})")
ax.set_xlabel("Complaints per borough-day (15 = 15 or more)")
ax.set_ylabel("Borough-days")
ax.set_title("Training period 2024-2025")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig4_flood_count_distribution.png", dpi=200)
plt.close(fig)
print("Saved 4 figures and eda_summary.txt in results/")