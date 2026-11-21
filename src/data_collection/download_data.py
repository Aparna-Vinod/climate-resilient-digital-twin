"""
Download raw data for the NYC flood Digital Twin capstone.

What it downloads (2024-01-01 to now):
  1. NYC 311 flood-related complaints  -> data/raw/311_flood_complaints.csv
  2. Open-Meteo rainfall (Central Park) -> data/raw/rainfall_daily.csv
                                           data/raw/rainfall_hourly.csv
  3. A source log with URLs, time and row counts -> data/raw/source_log.json

Run from the project root:
    python src/data_collection/download_data.py

Needs only: requests  (pip install requests)
Raw files go in data/raw/, which must be listed in .gitignore.
"""

import csv
import json
import os
from datetime import date, datetime, timedelta

import requests

START_DATE = "2024-01-01"
RAW_DIR = os.path.join("data", "raw")

# --- 311 ---------------------------------------------------------------
# Dataset: "311 Service Requests from 2020 to Present" (id erm2-nwe9)
URL_311 = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Descriptors seen in the live data (complaint_type = 'Sewer').
# We download these four and decide the final label rule later in PySpark.
DESCRIPTORS = [
    "Street Flooding (SJ)",
    "Highway Flooding (SH)",
    "Catch Basin Clogged/Flooding (Use Comments) (SC)",
    "Manhole Overflow (Use Comments) (SA1)",
]

FIELDS_311 = [
    "unique_key",
    "created_date",
    "closed_date",
    "complaint_type",
    "descriptor",
    "borough",
    "incident_zip",
    "latitude",
    "longitude",
    "status",
]

# --- Open-Meteo --------------------------------------------------------
URL_RAIN = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 40.78, -73.97  # Central Park, one point for the whole city
# ERA5 data runs about 5 days behind, so stop a few days before today.
END_DATE = (date.today() - timedelta(days=6)).isoformat()


def download_311():
    where_desc = ",".join("'" + d.replace("'", "''") + "'" for d in DESCRIPTORS)
    where = (
        f"complaint_type='Sewer' AND descriptor in({where_desc}) "
        f"AND created_date>='{START_DATE}T00:00:00'"
    )
    rows, offset, page = [], 0, 50000
    while True:
        params = {
            "$select": ",".join(FIELDS_311),
            "$where": where,
            "$order": "created_date,unique_key",
            "$limit": page,
            "$offset": offset,
        }
        r = requests.get(URL_311, params=params, timeout=120)
        r.raise_for_status()
        batch = r.json()
        rows.extend(batch)
        print(f"311: fetched {len(batch)} rows (total {len(rows)})")
        if len(batch) < page:
            break
        offset += page

    path = os.path.join(RAW_DIR, "311_flood_complaints.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_311, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS_311})
    return path, len(rows)


def write_series(block, path):
    """block is a dict like {'time': [...], 'precipitation': [...]}."""
    keys = list(block.keys())
    n = len(block["time"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([block[k][i] for k in keys])
    return n


def download_rain():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum,rain_sum",
        "hourly": "precipitation",
        "timezone": "America/New_York",
    }
    r = requests.get(URL_RAIN, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    daily_path = os.path.join(RAW_DIR, "rainfall_daily.csv")
    hourly_path = os.path.join(RAW_DIR, "rainfall_hourly.csv")
    n_daily = write_series(data["daily"], daily_path)
    n_hourly = write_series(data["hourly"], hourly_path)
    print(f"Rain: {n_daily} daily rows, {n_hourly} hourly rows")
    return (daily_path, n_daily), (hourly_path, n_hourly), r.url


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    p311, n311 = download_311()
    (pd_, nd), (ph, nh), rain_url = download_rain()

    log = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": START_DATE,
        "end_date_rainfall": END_DATE,
        "files": {
            p311: {"rows": n311, "source": URL_311, "descriptors": DESCRIPTORS},
            pd_: {"rows": nd, "source": rain_url},
            ph: {"rows": nh, "source": rain_url},
        },
    }
    log_path = os.path.join(RAW_DIR, "source_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print("Done. Log written to", log_path)


if __name__ == "__main__":
    main()
