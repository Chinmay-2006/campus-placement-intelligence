DATA_PATH = "data/student_placement_prediction_dataset_2026.csv"

CLASSIFIER_PATH = "models/placement_classifier.pkl"
REGRESSOR_PATH = "models/ctc_regressor.pkl"

EVALUATION_PATH = "reports/model_evaluation.json"

CLASSIFICATION_TARGET = "placement_status"
REGRESSION_TARGET = "salary_package_lpa"
ID_COLUMN = "student_id"

FEATURE_COLUMNS = [
    "age",
    "gender",
    "cgpa",
    "branch",
    "college_tier",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "hackathons_participated",
    "github_repos",
    "linkedin_connections",
    "mock_interview_score",
    "attendance_percentage",
    "backlogs",
    "extracurricular_score",
    "leadership_score",
    "volunteer_experience",
    "sleep_hours",
    "study_hours_per_day"
]

CATEGORICAL_FEATURES = [
    "gender",
    "branch",
    "college_tier",
    "volunteer_experience"
]

NUMERICAL_FEATURES = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURES
]

PLACEMENT_CLASSES = [
    "Not Placed",
    "Placed"
]