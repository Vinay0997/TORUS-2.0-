# Synthetic Case Summaries

These are illustrative, fully synthetic examples of how TORUS data flows
through the system. No real patient data is used anywhere in this
project.

## Case 1: Routine exam, normal outcome
A synthetic patient (age 34, risk_category "low") undergoes a routine
abdominal ultrasound on device DEV-004. The exam records a quality_score
of 0.91 and outcome_label "normal". The follow-up prediction model
correctly assigns a low predicted probability of follow-up (target=0),
consistent with the high quality score and low-risk category.

## Case 2: Flagged for follow-up
A synthetic patient (age 67, risk_category "high") undergoes an exam on
device DEV-002 with a lower quality_score of 0.58. The outcome_label is
"follow-up". The model's feature importance analysis (Random Forest)
typically ranks quality_score and risk_category among the top predictive
features, consistent with this case being flagged correctly.

## Case 3: Device anomaly correlating with exam quality
Device DEV-002 shows a rising anomaly_rate_pct in the telemetry anomaly
summary over the same period. Rolling error_count sums increase in the
72 hours before several exams on that device receive lower quality
scores. While TORUS does not currently build an explicit causal or joined
model linking device health directly to exam quality, this pattern
illustrates why device fleet monitoring matters for exam data quality —
a technician reviewing DEV-002's anomaly chart would have grounds to
inspect the unit before more low-quality exams occur.

## Case 4: False-positive anomaly
Device DEV-007 shows a single anomalous CPU spike reading with no
corresponding rise in error_count and no sustained elevation in
subsequent readings. Per the troubleshooting SOP, this is treated as a
likely transient event (e.g., background process) rather than a genuine
fault, and no escalation is triggered.

## Notes on interpreting these cases
These examples are meant to demonstrate reasoning patterns for the
assistant to draw on when answering questions about how TORUS's
components relate to each other. They are illustrative constructs from
synthetic data, not statements about any real device, patient, or
clinical event.
