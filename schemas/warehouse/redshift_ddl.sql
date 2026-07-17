CREATE TABLE patients (
    patient_id      INTEGER PRIMARY KEY,
    age             INTEGER,
    sex             VARCHAR(10),
    risk_category   VARCHAR(20)
);

CREATE TABLE ultrasound_exams (
    exam_id              INTEGER PRIMARY KEY,
    patient_id           INTEGER REFERENCES patients(patient_id),
    device_id            INTEGER,
    exam_type            VARCHAR(50),
    exam_timestamp       TIMESTAMP,
    image_quality_score  DOUBLE PRECISION,
    outcome_label        VARCHAR(50)
);

CREATE TABLE device_telemetry (
    device_id        INTEGER,
    telemetry_ts     TIMESTAMP,
    cpu_usage        DOUBLE PRECISION,
    memory_usage     DOUBLE PRECISION,
    errors_count     INTEGER,
    uptime_seconds   BIGINT
);