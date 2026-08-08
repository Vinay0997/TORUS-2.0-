-- =====================================================================
-- redshift_ddl.sql
-- TORUS-2.0: Data Warehouse Schema Definition
-- Target: Amazon Redshift (also compatible with PostgreSQL with minor edits)
--
-- Defines the target tables that raw CSVs (patients, ultrasound_exams,
-- device_telemetry) will be loaded into via the ETL/Airflow pipeline.
-- =====================================================================

-- Drop tables if they already exist (useful for dev/rebuild cycles)
DROP TABLE IF EXISTS stg_device_telemetry;
DROP TABLE IF EXISTS stg_ultrasound_exams;
DROP TABLE IF EXISTS stg_patients;

DROP TABLE IF EXISTS fact_device_telemetry;
DROP TABLE IF EXISTS fact_ultrasound_exams;
DROP TABLE IF EXISTS dim_patients;
DROP TABLE IF EXISTS dim_devices;

-- =====================================================================
-- DIMENSION TABLES
-- =====================================================================

CREATE TABLE dim_patients (
    patient_id      INTEGER      NOT NULL PRIMARY KEY,
    age             SMALLINT     NOT NULL,
    sex             VARCHAR(1)   NOT NULL,
    risk_category   VARCHAR(10)  NOT NULL,
    created_at      TIMESTAMP    DEFAULT GETDATE()
)
DISTSTYLE ALL
SORTKEY (patient_id);

CREATE TABLE dim_devices (
    device_id       VARCHAR(20)  NOT NULL PRIMARY KEY,
    device_name     VARCHAR(50),
    location        VARCHAR(100),
    installed_date  DATE,
    created_at      TIMESTAMP    DEFAULT GETDATE()
)
DISTSTYLE ALL
SORTKEY (device_id);

-- =====================================================================
-- FACT TABLES
-- =====================================================================

CREATE TABLE fact_ultrasound_exams (
    exam_id             INTEGER       NOT NULL PRIMARY KEY,
    patient_id          INTEGER       NOT NULL REFERENCES dim_patients(patient_id),
    device_id           VARCHAR(20)   NOT NULL REFERENCES dim_devices(device_id),
    exam_type           VARCHAR(20)   NOT NULL,
    exam_timestamp      TIMESTAMP     NOT NULL,
    image_quality_score DECIMAL(4,2),
    outcome_label       VARCHAR(30)   NOT NULL,
    loaded_at           TIMESTAMP     DEFAULT GETDATE()
)
DISTSTYLE KEY
DISTKEY (patient_id)
SORTKEY (exam_timestamp);

CREATE TABLE fact_device_telemetry (
    telemetry_id        BIGINT IDENTITY(1,1) PRIMARY KEY,
    device_id           VARCHAR(20)   NOT NULL REFERENCES dim_devices(device_id),
    "timestamp"         TIMESTAMP     NOT NULL,
    cpu_utilization     DECIMAL(5,2),
    memory_utilization  DECIMAL(5,2),
    error_count         SMALLINT,
    uptime_flag         SMALLINT,
    loaded_at           TIMESTAMP     DEFAULT GETDATE()
)
DISTSTYLE KEY
DISTKEY (device_id)
SORTKEY (device_id, "timestamp");

-- =====================================================================
-- STAGING TABLES (raw landing zone before transform/validate step)
-- Mirrors the shape of the source CSVs exactly, no constraints.
-- =====================================================================

CREATE TABLE stg_patients (
    patient_id      INTEGER,
    age             SMALLINT,
    sex             VARCHAR(1),
    risk_category   VARCHAR(10)
);

CREATE TABLE stg_ultrasound_exams (
    exam_id             INTEGER,
    patient_id          INTEGER,
    device_id           VARCHAR(20),
    exam_type           VARCHAR(20),
    exam_timestamp       TIMESTAMP,
    image_quality_score  DECIMAL(4,2),
    outcome_label        VARCHAR(30)
);

CREATE TABLE stg_device_telemetry (
    device_id           VARCHAR(20),
    "timestamp"          TIMESTAMP,
    cpu_utilization      DECIMAL(5,2),
    memory_utilization   DECIMAL(5,2),
    error_count          SMALLINT,
    uptime_flag          SMALLINT
);

-- =====================================================================
-- NOTES
-- 1. COPY commands (for real Redshift) would load CSVs from S3 into the
--    stg_* tables, e.g.:
--
--    COPY stg_patients
--    FROM 's3://torus-2-0-bucket/raw/patients.csv'
--    IAM_ROLE 'arn:aws:iam::<account_id>:role/RedshiftLoadRole'
--    CSV IGNOREHEADER 1;
--
-- 2. The Airflow DAG (see dags/etl_pipeline_dag.py) handles:
--    extract (read local CSVs) -> transform/validate (pandas checks)
--    -> load (upsert stg_* into dim_*/fact_* tables)
-- =====================================================================
