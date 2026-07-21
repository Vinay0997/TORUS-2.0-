\copy patients(patient_id, age, sex, risk_category) 
FROM 'C:/Users/chaitu/torus-remote-ultrasound-ai-platform/data/raw/patients.csv' DELIMITER ',' CSV HEADER;
\copy ultrasound_exams(exam_id, patient_id, device_id, exam_type, exam_timestamp, image_quality_score, outcome_label) 
FROM 'C:/Users/chaitu/torus-remote-ultrasound-ai-platform/data/raw/ultrasound_exams.csv' DELIMITER ',' CSV HEADER;
\copy device_telemetry(device_id, telemetry_ts, cpu_usage, memory_usage, errors_count, uptime_seconds) 

FROM 'C:/Users/chaitu/torus-remote-ultrasound-ai-platform/data/raw/device_telemetry.csv' DELIMITER ',' CSV HEADER;