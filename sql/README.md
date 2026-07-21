# SQL Pipeline Structure

- 01_schema_and_raw_tables.sql: Creates raw PostgreSQL tables.
- 02_load_raw_csv_notes.sql: Loads CSV data into raw tables using psql \copy.
- 03_features.sql: Builds transformed feature tables for downstream ML/analytics.
- 04_data_quality_checks.sql: Validates nulls and referential integrity.
- 05_analytics_views.sql: Creates reporting-friendly SQL views for dashboards.