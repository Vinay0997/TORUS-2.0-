CREATE OR REPLACE VIEW vw_exam_volume_by_day AS
SELECT
    DATE(exam_timestamp) AS exam_date,
    COUNT(*) AS exam_count
FROM ultrasound_exams
GROUP BY DATE(exam_timestamp)
ORDER BY exam_date;

CREATE OR REPLACE VIEW vw_follow_up_rate_by_risk AS
SELECT
    p.risk_category,
    COUNT(*) AS total_exams,
    SUM(CASE WHEN e.outcome_label = 'follow_up_required' THEN 1 ELSE 0 END) AS follow_up_exams,
    ROUND(
        100.0 * SUM(CASE WHEN e.outcome_label = 'follow_up_required' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS follow_up_rate_pct
FROM ultrasound_exams e
JOIN patients p
    ON e.patient_id = p.patient_id
GROUP BY p.risk_category;

CREATE OR REPLACE VIEW vw_device_error_summary AS
SELECT
    device_id,
    SUM(errors_count) AS total_errors,
    AVG(cpu_usage) AS avg_cpu_usage,
    AVG(memory_usage) AS avg_memory_usage
FROM device_telemetry
GROUP BY device_id;