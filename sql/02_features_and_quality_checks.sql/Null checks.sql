-- Check patients key columns for NULLs
SELECT *
FROM patients
WHERE patient_id IS NULL
   OR age IS NULL
   OR sex IS NULL
   OR risk_category IS NULL;

-- Check ultrasound_exams key columns for NULLs
SELECT *
FROM ultrasound_exams
WHERE exam_id IS NULL
   OR patient_id IS NULL
   OR device_id IS NULL
   OR exam_type IS NULL
   OR exam_timestamp IS NULL
   OR image_quality_score IS NULL
   OR outcome_label IS NULL;

-- Check device_telemetry key columns for NULLs
SELECT *
FROM device_telemetry
WHERE device_id IS NULL
   OR telemetry_ts IS NULL
   OR cpu_usage IS NULL
   OR memory_usage IS NULL
   OR errors_count IS NULL
   OR uptime_seconds IS NULL;

''' If all three queries return “0 rows”, your data passes the basic null check.'''
