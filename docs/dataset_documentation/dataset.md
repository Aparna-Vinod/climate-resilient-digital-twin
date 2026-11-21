# Datasets

## 1. NYC 311 Service Requests from 2020 to Present
- Publisher: NYC Open Data (City of New York), dataset id erm2-nwe9
- Access: Socrata API https://data.cityofnewyork.us/resource/erm2-nwe9.json
- Filter used: complaint_type = 'Sewer', four descriptors (Street Flooding (SJ), Highway Flooding (SH),
  Catch Basin Clogged/Flooding (SC), Manhole Overflow (SA1)), created on or after 2024-01-01
- Downloaded: 29,704 records, 10 fields (see data/raw/source_log.json for time of download)
- Use: complaint counts per borough per day; label = flood_count >= 4 (street + highway flooding)
- Limitation: complaints are reported by people, so they measure reporting, not actual flooding
- Licence: not stated on the dataset page (checked); cite the source

## 2. Open-Meteo Historical Weather API
- Source: https://archive-api.open-meteo.com/v1/archive
- Location: Central Park (40.78, -73.97), one point for the whole city
- Downloaded: 1,050 daily rows and 25,200 hourly rows, 2024-01-01 to 2026-11-15
- Variables: precipitation_sum, rain_sum (daily); precipitation (hourly)
- Terms: data under CC BY 4.0, non-commercial use (cite Open-Meteo)
- Limitation: reanalysis data (about 25 km grid), not a rain gauge reading

## Processed table
data/processed/borough_day.csv: 5,250 rows (1,050 days x 5 boroughs). Built by
src/bigdata_processing/process_data.py. Row counts at each cleaning step are in
data/processed/cleaning_summary.json.