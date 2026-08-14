# FHIR and HL7 Basics for TORUS

## What is FHIR?
FHIR (Fast Healthcare Interoperability Resources) is a modern healthcare
data exchange standard maintained by HL7. It represents clinical and
administrative data as discrete "Resources" — modular JSON or XML objects
that can be linked together to build a full clinical picture without
requiring a single monolithic record format.

## Key resources relevant to TORUS

### Patient
Represents demographic and administrative information about a patient:
identifiers, name, birth date, gender, and administrative risk flags. In
TORUS's synthetic model, this maps conceptually to the `patients` table
(`patient_id`, `age`, `sex`, `risk_category`).

### ImagingStudy
Represents a medical imaging exam, including references to the patient,
the performing device, modality (e.g., ultrasound), and the study's
timestamp. In TORUS, this maps to the `ultrasound_exams` table
(`exam_id`, `patient_id`, `device_id`, `exam_type`, `timestamp`).

### Observation
Represents a single measured or asserted clinical fact — a lab result, a
vital sign, or in TORUS's case, an exam's `quality_score` or
`outcome_label` (normal vs. follow-up). Observations often reference the
ImagingStudy or encounter they were derived from.

### Device
Represents a physical piece of medical equipment, including its
identifier, type, and status. TORUS's `device_telemetry` table
conceptually extends this idea by tracking a Device's operational health
over time (CPU, memory, uptime, errors), which is not a standard FHIR
field but represents realistic IoT/telemetry augmentation of a Device
resource in a fleet-monitoring context.

## Why this matters for TORUS
TORUS does not implement full FHIR compliance — it borrows the conceptual
structure (separating patient identity, exam/study data, and
observations/results into related-but-distinct entities) to keep the
synthetic data pipeline realistic and to demonstrate familiarity with how
real EHR/imaging systems structure their data for interoperability.

## HL7 v2 vs FHIR
HL7 v2 (an older, pipe-delimited messaging standard) is still common in
hospital systems for real-time event messages (e.g., ADT — Admit,
Discharge, Transfer). FHIR is the newer, RESTful, JSON-based standard
increasingly used for both messaging and bulk data exchange. TORUS's
architecture is conceptually closer to FHIR's resource-based model.
