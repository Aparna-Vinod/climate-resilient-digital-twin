# Climate-Resilient Digital Twin (NYC Street Flooding)

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
