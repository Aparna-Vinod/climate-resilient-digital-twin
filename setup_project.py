# setup_project.py  (run inside the cloned repo folder)
#   python setup_project.py              -> create folders and files
#   python setup_project.py --download   -> also download the data
import os
import subprocess
import sys

FOLDERS = [
    "data/sample", "data/processed", "notebooks",
    "src/data_collection", "src/bigdata_processing", "src/digital_twin",
    "src/machine_learning", "src/scenario_analysis",
    "dashboard", "models", "tests", "results",
    "docs/architecture", "docs/dataset_documentation", "docs/report_notes",
]

GITIGNORE_LINES = ["data/raw/", "data/processed/*.parquet", ".env"]

REQUIREMENTS = "requests\npyspark\npandas\nscikit-learn\nstreamlit\n"

README = '''# Climate-Resilient Digital Twin (NYC Street Flooding)

M.Sc. capstone: A Cloud-Based Digital Twin for Climate-Resilient Urban Infrastructure Using Big Data Analytics.

## Status
In progress. Everything is PROPOSED unless marked done.
- [x] Repository created
- [ ] Data downloaded and documented
- [ ] PySpark pipeline
- [ ] Digital Twin, ML, what-if analysis, dashboard

## Proposed scope
Street and drain flooding from heavy rainfall in New York City, per borough per day.
Signal: NYC 311 flood complaints (a proxy for flooding, not a measured flood).
Rainfall: Open-Meteo. Data window: 2024-01-01 to now. Train 2024-2025, test 2026.

## Data sources
- NYC 311 Service Requests from 2020 to Present (NYC Open Data, id erm2-nwe9)
- Open-Meteo Historical Weather API (CC BY 4.0, non-commercial use)

## Setup
    pip install -r requirements.txt
    python src/data_collection/download_data.py

## Team
- Aparna V (Member 1): data collection, cloud storage, PySpark
- Archana T (Member 2): Digital Twin, ML, scenarios, dashboard
'''

DOWNLOAD_SCRIPT = '''"""
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
'''


def write_if_missing(path, text):
    if os.path.exists(path):
        print("kept existing:", path)
        return
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("created:", path)


def main():
    if not os.path.isdir(".git"):
        print("WARNING: no .git folder here. Run this inside your cloned repo folder.")
    for d in FOLDERS:
        os.makedirs(d, exist_ok=True)
        keep = os.path.join(d, ".gitkeep")
        if not os.path.exists(keep):
            open(keep, "a").close()
    print("folders ready")

    existing = ""
    if os.path.exists(".gitignore"):
        with open(".gitignore", encoding="utf-8") as f:
            existing = f.read()
    missing = [x for x in GITIGNORE_LINES if x not in existing.splitlines()]
    if missing:
        with open(".gitignore", "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n# project data and secrets\n" + "\n".join(missing) + "\n")
        print("added to .gitignore:", missing)

    write_if_missing("requirements.txt", REQUIREMENTS)
    write_if_missing("src/data_collection/download_data.py", DOWNLOAD_SCRIPT)
    if os.path.exists("README.md") and os.path.getsize("README.md") < 200:
        os.remove("README.md")
    write_if_missing("README.md", README)

    if "--download" in sys.argv:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        script = os.path.join("src", "data_collection", "download_data.py")
        subprocess.check_call([sys.executable, script])
    print("Setup finished.")


if __name__ == "__main__":
    main()