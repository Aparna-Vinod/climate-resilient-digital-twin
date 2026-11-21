# REPORT NOTES - Phase 1: Data collection and PySpark processing
Chapter: Methodology > Data Collection and Big Data Processing

Report-ready text (implemented and run):
Flood-related NYC 311 complaints (complaint type "Sewer", four descriptors) created on or after
1 January 2024 were downloaded from NYC Open Data through the Socrata API (29,704 records).
Daily and hourly precipitation for Central Park (40.78, -73.97) was downloaded from the
Open-Meteo Historical Weather API (1,050 days, 25,200 hourly values, 2024-01-01 to 2026-11-15).
A PySpark job (local mode) removed duplicate records, parsed timestamps, standardised borough
names and dropped 13 records without a valid borough. Rainfall features (same-day, previous-day,
3-day and 7-day totals, peak hourly rainfall, wet hours) were joined with daily complaint counts for
each of the five boroughs, giving 5,250 borough-day rows.
A borough-day is labelled a flood day if it has at least 4 street/highway flooding complaints,
the 90th percentile of the training period (2024-2025). Testing uses 2026 data.

Limitations to state: complaints are a proxy for flooding, not measured flooding; one rainfall
point is used for the whole city; the data is moderate in size (about 29.7k records), so Spark
is used for the pipeline design, not because the data needs it; Parquet output was not
produced on Windows (Hadoop winutils missing), CSV is used instead.

Evidence: docs/ screenshot of terminal output; data/processed/cleaning_summary.json
Figures/tables: cleaning summary table; flood_count distribution; correlation table
Commit message: Implement PySpark data cleaning and rainfall join
