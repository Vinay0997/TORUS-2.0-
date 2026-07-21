SELECT e.*
FROM ultrasound_exams e
LEFT JOIN patients p
    ON e.patient_id = p.patient_id
WHERE p.patient_id IS NULL;


'''If this returns no rows, it means every exam is linked to a valid patient.

If it returns rows, those are exams whose patient_id doesn’t exist in patients (you’d fix the data or CSVs) '''