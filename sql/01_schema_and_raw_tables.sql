DROP TABLE IF EXISTS ultrasound_exams;
DROP TABLE IF EXISTS device_telemetry;
DROP TABLE IF EXISTS patients;

CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER,
    sex VARCHAR(10),
    risk_category VARCHAR(20)
);

CREATE TABLE ultrasound_exams (
    exam_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    device_id INTEGER,
    exam_type VARCHAR(50),
    exam_timestamp TIMESTAMP,
    image_quality_score DOUBLE PRECISION,
    outcome_label VARCHAR(50),
    CONSTRAINT fk_patient
        FOREIGN KEY (patient_id)
        REFERENCES patients(patient_id)
);

CREATE TABLE device_telemetry (
    device_id INTEGER,
    telemetry_ts TIMESTAMP,
    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,
    errors_count INTEGER,
    uptime_seconds INTEGER
);