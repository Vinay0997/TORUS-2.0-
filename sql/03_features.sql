DROP TABLE IF EXISTS patient_features;

CREATE TABLE patient_features AS
SELECT
    p.patient_id,
    p.age,
    p.sex,
    p.risk_category,
    COUNT(e.exam_id) AS total_exams,
    AVG(e.image_quality_score) AS avg_image_quality,
    SUM(
        CASE
            WHEN e.outcome_label = 'follow_up_required' THEN 1
            ELSE 0
        END
    ) AS follow_up_count
FROM patients p
LEFT JOIN ultrasound_exams e
    ON p.patient_id = e.patient_id
GROUP BY
    p.patient_id,
    p.age,
    p.sex,
    p.risk_category;

DROP TABLE IF EXISTS device_uptime_stats;

CREATE TABLE device_uptime_stats AS
SELECT
    device_id,
    COUNT(*) AS telemetry_points,
    AVG(cpu_usage) AS avg_cpu_usage,
    AVG(memory_usage) AS avg_memory_usage,
    SUM(errors_count) AS total_errors,
    MAX(uptime_seconds) AS max_uptime_seconds
FROM device_telemetry
GROUP BY device_id;