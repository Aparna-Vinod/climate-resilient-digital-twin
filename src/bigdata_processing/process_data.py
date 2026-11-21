"""
PySpark pipeline: clean NYC 311 flood complaints, join with rainfall,
and build one row per borough per day.

Input  (from download_data.py):
    data/raw/311_flood_complaints.csv
    data/raw/rainfall_daily.csv
    data/raw/rainfall_hourly.csv
Output:
    data/processed/borough_day.csv        (handover table, small, committed to Git)
    data/processed/borough_day_parquet/   (Spark Parquet, only if the write works)
    data/processed/cleaning_summary.json  (row counts for the report)

Run from the project root:
    python src/bigdata_processing/process_data.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from pyspark.sql import SparkSession, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

RAW = Path("data") / "raw"
OUT = Path("data") / "processed"

BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
MAIN_FLOOD = ["Street Flooding (SJ)", "Highway Flooding (SH)"]
CATCH_BASIN = "Catch Basin Clogged/Flooding (Use Comments) (SC)"
MANHOLE = "Manhole Overflow (Use Comments) (SA1)"
TRAIN_END = "2026-01-01"  # rows before this date are the training period


def uri(path):
    return Path(path).resolve().as_uri()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("nyc-flood-twin")
        .config("spark.sql.session.timeZone", "America/New_York")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    summary = {}

    # ---------- 1. Load and clean the 311 complaints ----------
    raw = spark.read.option("header", True).csv(uri(RAW / "311_flood_complaints.csv"))
    summary["311_rows_loaded"] = raw.count()

    c = raw.withColumn(
        "created_ts", F.to_timestamp("created_date", "yyyy-MM-dd'T'HH:mm:ss.SSS")
    )
    summary["311_bad_timestamp_dropped"] = c.filter(F.col("created_ts").isNull()).count()
    c = c.filter(F.col("created_ts").isNotNull())

    n_before = c.count()
    c = c.dropDuplicates(["unique_key"])
    summary["311_duplicates_dropped"] = n_before - c.count()

    c = c.withColumn("borough", F.upper(F.trim("borough")))
    bad_b = F.col("borough").isNull() | (~F.col("borough").isin(BOROUGHS))
    summary["311_bad_borough_dropped"] = c.filter(bad_b).count()
    c = c.filter(F.col("borough").isin(BOROUGHS))
    c = c.withColumn("date", F.to_date("created_ts"))

    # ---------- 2. Load rainfall and build features ----------
    rain = (
        spark.read.option("header", True)
        .csv(uri(RAW / "rainfall_daily.csv"))
        .select(
            F.to_date("time").alias("date"),
            F.col("precipitation_sum").cast("double").alias("precip_mm"),
            F.col("rain_sum").cast("double").alias("rain_mm"),
        )
    )
    hourly = (
        spark.read.option("header", True)
        .csv(uri(RAW / "rainfall_hourly.csv"))
        .select(
            F.to_date(F.substring("time", 1, 10)).alias("date"),
            F.col("precipitation").cast("double").alias("p"),
        )
        .groupBy("date")
        .agg(
            F.max("p").alias("max_hourly_mm"),
            F.sum(F.when(F.col("p") > 0, 1).otherwise(0)).alias("wet_hours"),
        )
    )
    rain = rain.join(hourly, "date", "left")
    summary["rain_days"] = rain.count()
    summary["rain_null_values_filled_with_0"] = rain.filter(
        F.col("precip_mm").isNull() | F.col("max_hourly_mm").isNull()
    ).count()
    rain = rain.fillna(0, subset=["precip_mm", "rain_mm", "max_hourly_mm", "wet_hours"])

    w = Window.orderBy("date")
    rain = (
        rain.withColumn("precip_prev1", F.lag("precip_mm", 1).over(w))
        .withColumn("precip_3day", F.sum("precip_mm").over(w.rowsBetween(-2, 0)))
        .withColumn("precip_7day", F.sum("precip_mm").over(w.rowsBetween(-6, 0)))
        .fillna(0, subset=["precip_prev1"])  # first day has no previous day
        .withColumn("month", F.month("date"))
        .withColumn("day_of_week", F.dayofweek("date"))
    )

    # ---------- 3. Count complaints per borough per day ----------
    max_date = rain.agg(F.max("date")).collect()[0][0]
    summary["311_after_last_rain_day_ignored"] = c.filter(
        F.col("date") > F.lit(max_date)
    ).count()

    counts = c.groupBy("borough", "date").agg(
        F.sum(F.when(F.col("descriptor").isin(MAIN_FLOOD), 1).otherwise(0)).alias("flood_count"),
        F.sum(F.when(F.col("descriptor") == CATCH_BASIN, 1).otherwise(0)).alias("catch_basin_count"),
        F.sum(F.when(F.col("descriptor") == MANHOLE, 1).otherwise(0)).alias("manhole_count"),
    )

    # Every borough on every rain day, so days with no complaints become 0
    boroughs = spark.createDataFrame([(b,) for b in BOROUGHS], ["borough"])
    grid = (
        rain.crossJoin(boroughs)
        .join(counts, ["borough", "date"], "left")
        .fillna(0, subset=["flood_count", "catch_basin_count", "manhole_count"])
        .orderBy("date", "borough")
        .cache()
    )
    summary["output_rows"] = grid.count()
    summary["output_rows_expected"] = summary["rain_days"] * len(BOROUGHS)
    summary["date_min"] = str(grid.agg(F.min("date")).collect()[0][0])
    summary["date_max"] = str(max_date)

    # ---------- 4. Quick look at the training period ----------
    train = grid.filter(F.col("date") < F.to_date(F.lit(TRAIN_END)))
    print("\n--- flood_count per borough-day, training period (before 2026) ---")
    for row in train.groupBy("flood_count").count().orderBy("flood_count").limit(15).collect():
        print(f"  flood_count={row['flood_count']}: {row['count']} borough-days")
    q = train.approxQuantile("flood_count", [0.5, 0.75, 0.9, 0.95, 0.99], 0.0)
    print("flood_count quantiles 50/75/90/95/99%:", q)
    print("total flood complaints by borough (all days):")
    for row in grid.groupBy("borough").agg(F.sum("flood_count").alias("n")).orderBy("borough").collect():
        print(f"  {row['borough']}: {row['n']}")
    print("correlation of flood_count with rainfall (training period):")
    for col in ["precip_mm", "precip_prev1", "precip_3day", "max_hourly_mm"]:
        print(f"  {col}: {train.stat.corr('flood_count', col):.3f}")

    # ---------- 5. Save ----------
    pdf = grid.toPandas()
    pdf["date"] = pdf["date"].astype(str)
    pdf.to_csv(OUT / "borough_day.csv", index=False)
    try:
        grid.write.mode("overwrite").parquet(uri(OUT / "borough_day_parquet"))
        summary["parquet_written"] = True
    except Exception as e:  # common on Windows without Hadoop tools
        summary["parquet_written"] = False
        print("Parquet write skipped (CSV was saved):", str(e)[:150])

    with open(OUT / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n--- cleaning summary ---")
    print(json.dumps(summary, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()