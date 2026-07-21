SELECT *
FROM patients
WHERE patient_id IS NULL
   OR age IS NULL
   OR sex IS NULL
   OR risk_category IS NULL;

SELECT *
FROM ultrasound_exams
WHERE exam_id IS NULL
   OR patient_id IS NULL
   OR device_id IS NULL
   OR exam_type IS NULL
   OR exam_timestamp IS NULL
   OR image_quality_score IS NULL
   OR outcome_label IS NULL;

SELECT *
FROM device_telemetry
WHERE device_id IS NULL
   OR telemetry_ts IS NULL
   OR cpu_usage IS NULL
   OR memory_usage IS NULL
   OR errors_count IS NULL
   OR uptime_seconds IS NULL;

SELECT e.*
FROM ultrasound_exams e
LEFT JOIN patients p
    ON e.patient_id = p.patient_id
WHERE p.patient_id IS NULL;