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



''' for preview 
SELECT * FROM device_uptime_stats LIMIT 5; '''

''' This gives you a device‑level health summary table for analytics and dashboards.

'''