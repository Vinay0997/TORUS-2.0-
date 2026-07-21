DROP INDEX IF EXISTS idx_ultrasound_exams_patient_id;
DROP INDEX IF EXISTS idx_ultrasound_exams_exam_timestamp;
DROP INDEX IF EXISTS idx_device_telemetry_device_id;
DROP INDEX IF EXISTS idx_device_telemetry_telemetry_ts;

CREATE INDEX IF NOT EXISTS idx_ultrasound_exams_patient_id
ON ultrasound_exams(patient_id);

CREATE INDEX IF NOT EXISTS idx_ultrasound_exams_exam_timestamp
ON ultrasound_exams(exam_timestamp);

CREATE INDEX IF NOT EXISTS idx_device_telemetry_device_id
ON device_telemetry(device_id);

CREATE INDEX IF NOT EXISTS idx_device_telemetry_telemetry_ts
ON device_telemetry(telemetry_ts);