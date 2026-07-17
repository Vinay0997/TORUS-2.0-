# TORUS Remote Ultrasound AI Platform

This project simulates a healthcare data platform for a tele-operated robotic ultrasound system,
inspired by my work at Plebs Innovations on the TORUS Remote Diagnostic Data Platform.[file:1]

## Step 1 – Healthcare data model & synthetic data

- Synthetic patients, ultrasound exams, and device telemetry.
- FHIR/HL7-inspired JSON schemas for Patient and ImagingStudy.
- Warehouse DDL for patients, exams, and telemetry tables.

Next steps: build ETL pipelines with Airflow + Spark and ML models tracked with MLflow.