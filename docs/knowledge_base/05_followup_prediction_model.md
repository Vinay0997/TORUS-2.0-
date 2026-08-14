# Follow-Up Prediction Model

## Purpose
Predicts whether an ultrasound exam is likely to require clinical
follow-up (`target = 1`) versus being classified as normal
(`target = 0`), based on exam metadata rather than image content.

## Data source
Trained on `data/processed/modeling_dataset.csv`, which joins patient
demographics, exam metadata, and quality scores into a single modeling
table. The label column is `target`; `outcome_label` is the human-readable
version of the same signal and is dropped from the feature set to avoid
leakage.

## Model candidates
Two pipelines are trained and compared:

1. **Logistic Regression** — `LogisticRegression(max_iter=1000,
   class_weight="balanced")`, chosen as an interpretable baseline.
   `class_weight="balanced"` compensates for the fact that most exams are
   normal and follow-up cases are the minority class.

2. **Random Forest** — `RandomForestClassifier(n_estimators=200,
   max_depth=8, random_state=42, class_weight="balanced")`, chosen to
   capture non-linear interactions between features (e.g., age and
   quality_score jointly influencing follow-up likelihood) that logistic
   regression cannot.

Both models share the same preprocessing: a `ColumnTransformer` applying
`OneHotEncoder(handle_unknown="ignore")` to categorical columns and
passing numeric columns through unchanged.

## Model selection
The two models are compared on **F1 score** rather than raw accuracy,
because the follow-up class is a minority class — a model that always
predicts "normal" could still show high accuracy while being clinically
useless. F1 balances precision (avoiding unnecessary follow-ups) and
recall (not missing exams that genuinely need follow-up).

## Experiment tracking
Both models are logged as separate MLflow runs under the
`torus-followup-prediction` experiment, with accuracy, precision, recall,
F1, and ROC-AUC recorded for each, plus the fitted pipeline as a model
artifact. This allows direct comparison in the MLflow UI rather than only
seeing the final winner.

## Output
The winning model is saved to
`models/followup_prediction_{best_model_name}.pkl` via `joblib.dump`.

## Clinical framing and limitations
This model uses metadata only (no image analysis) and is a
proof-of-concept for triage prioritization, not a diagnostic tool. It
should never be used to make or imply a clinical diagnosis — its purpose
is purely to flag exams for human review based on statistical patterns in
synthetic data.
