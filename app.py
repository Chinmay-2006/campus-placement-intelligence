import streamlit as st
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)

from src.data_loader import (
    load_dataset,
    get_dataset_summary,
)

from src.prediction import (
    load_models,
    predict_student,
)

from src.evaluation import (
    load_evaluation_results,
    get_classification_results,
    get_classification_report,
    get_confusion_matrix,
    get_regression_results,
    get_regression_error_statistics,
    get_selected_models,
    get_cross_validation_results,
)

from src.explainability import (
    get_classification_feature_effects,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Campus Placement & CTC Intelligence Hub",
    layout="wide",
)


# ============================================================
# LOAD PROJECT RESOURCES
# ============================================================

@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_dataset():
    return load_dataset()


@st.cache_data
def get_evaluation():
    return load_evaluation_results()


classifier, regressor = get_models()
df = get_dataset()
evaluation = get_evaluation()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_student_profile(
    age,
    gender,
    cgpa,
    branch,
    college_tier,
    internships,
    projects,
    certifications,
    coding,
    aptitude,
    communication,
    logical_reasoning,
    hackathons,
    github,
    linkedin,
    mock_interview,
    attendance,
    backlogs,
    extracurricular,
    leadership,
    volunteer,
    sleep_hours,
    study_hours,
):
    return {
        "age": age,
        "gender": gender,
        "cgpa": cgpa,
        "branch": branch,
        "college_tier": college_tier,
        "internships_count": internships,
        "projects_count": projects,
        "certifications_count": certifications,
        "coding_skill_score": coding,
        "aptitude_score": aptitude,
        "communication_skill_score": communication,
        "logical_reasoning_score": logical_reasoning,
        "hackathons_participated": hackathons,
        "github_repos": github,
        "linkedin_connections": linkedin,
        "mock_interview_score": mock_interview,
        "attendance_percentage": attendance,
        "backlogs": backlogs,
        "extracurricular_score": extracurricular,
        "leadership_score": leadership,
        "volunteer_experience": volunteer,
        "sleep_hours": sleep_hours,
        "study_hours_per_day": study_hours,
    }


def profile_to_dataframe(profile):
    return pd.DataFrame(
        [profile],
        columns=FEATURE_COLUMNS,
    )


def save_history(result):
    st.session_state.prediction_history.append(
        {
            "Prediction": result["placement_prediction"],
            "Placement Probability": (
                f"{result['placement_probability'] * 100:.2f}%"
            ),
            "Expected CTC": (
                f"{result['predicted_ctc']:.2f} LPA"
                if result["predicted_ctc"] is not None
                else "Not Generated"
            ),
        }
    )


def show_prediction(result):

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Status",
            result["placement_prediction"].upper(),
        )

    with col2:
        st.metric(
            "Placement Probability",
            f"{result['placement_probability'] * 100:.2f}%",
        )

    with col3:
        if result["predicted_ctc"] is not None:
            st.metric(
                "Expected Starting CTC",
                f"{result['predicted_ctc']:.2f} LPA",
            )
        else:
            st.metric(
                "Expected Starting CTC",
                "Not Generated",
            )

    st.subheader("Prediction Summary")

    probability = result["placement_probability"] * 100

    if result["predicted_ctc"] is not None:
        ctc_text = (
            f"The regression model estimates an expected "
            f"starting CTC of "
            f"{result['predicted_ctc']:.2f} LPA."
        )
    else:
        ctc_text = (
            "The regression model does not generate a CTC "
            "prediction for a Not Placed prediction."
        )

    st.write(
        f"The classification model predicts "
        f"{result['placement_prediction']} with a "
        f"model-estimated placement probability of "
        f"{probability:.2f}%. {ctc_text}"
    )


def show_history(history_key):

    if not st.session_state.prediction_history:
        return

    st.subheader("Session Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Clear Prediction History",
        key=f"clear_prediction_history_{history_key}",
    ):
        st.session_state.prediction_history = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("Campus Placement & CTC Intelligence Hub")

st.caption(
    "Two-stage machine learning system for placement "
    "prediction and expected starting CTC estimation."
)


# ============================================================
# TABS
# ============================================================

prediction_tab, what_if_tab, model_tab, about_tab = st.tabs(
    [
        "Student Prediction",
        "What-If Analysis",
        "Model & Dataset",
        "About Project",
    ]
)


# ============================================================
# STUDENT PREDICTION
# ============================================================

with prediction_tab:

    st.header("Student Profile")

    st.write(
        "Enter the student's academic, technical, "
        "communication and activity profile."
    )

    st.subheader("Academic Profile")

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=30,
            value=21,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.50,
            step=0.01,
        )

    with c2:

        branch = st.selectbox(
            "Branch",
            ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"],
        )

        college_tier = st.selectbox(
            "College Tier",
            ["Tier 1", "Tier 2", "Tier 3"],
        )

        backlogs = st.number_input(
            "Backlogs",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
        )

    with c3:

        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=80.00,
            step=0.01,
        )

        study_hours = st.number_input(
            "Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=3.00,
            step=0.1,
        )

        sleep_hours = st.number_input(
            "Sleep Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=7.00,
            step=0.1,
        )

    st.subheader("Experience & Projects")

    c1, c2, c3 = st.columns(3)

    with c1:

        internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

        projects = st.number_input(
            "Projects",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
        )

    with c2:

        hackathons = st.number_input(
            "Hackathons Participated",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
        )

        github = st.number_input(
            "GitHub Repositories",
            min_value=0,
            max_value=1000,
            value=10,
            step=1,
        )

        linkedin = st.number_input(
            "LinkedIn Connections",
            min_value=0,
            max_value=5000,
            value=300,
            step=10,
        )

    with c3:

        volunteer = st.selectbox(
            "Volunteer Experience",
            ["No", "Yes"],
        )

        extracurricular = st.number_input(
            "Extracurricular Score",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
        )

        leadership = st.number_input(
            "Leadership Score",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
        )

    st.subheader("Skills & Assessment")

    c1, c2, c3 = st.columns(3)

    with c1:

        coding = st.number_input(
            "Coding Skill Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )

        aptitude = st.number_input(
            "Aptitude Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )

    with c2:

        logical_reasoning = st.number_input(
            "Logical Reasoning Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )

        communication = st.number_input(
            "Communication Skill Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )

    with c3:

        mock_interview = st.number_input(
            "Mock Interview Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )

    st.divider()

    if st.button(
        "Predict Placement & CTC",
        type="primary",
        use_container_width=True,
        key="predict_student",
    ):

        profile = create_student_profile(
            age,
            gender,
            cgpa,
            branch,
            college_tier,
            internships,
            projects,
            certifications,
            coding,
            aptitude,
            communication,
            logical_reasoning,
            hackathons,
            github,
            linkedin,
            mock_interview,
            attendance,
            backlogs,
            extracurricular,
            leadership,
            volunteer,
            sleep_hours,
            study_hours,
        )

        student_df = profile_to_dataframe(profile)

        try:

            result = predict_student(
                student_df,
                classifier,
                regressor,
            )

            st.session_state.last_profile = profile
            st.session_state.last_prediction = result

            save_history(result)

            show_prediction(result)

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )

    show_history("student_prediction")


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

with what_if_tab:

    st.header("What-If Analysis")

    st.write(
        "Adjust selected profile variables to examine how "
        "the trained models respond to a changed student profile."
    )

    st.warning(
        "This is a model sensitivity tool. Changes in the output "
        "do not establish causal relationships."
    )

    st.subheader("Adjust Profile Variables")

    c1, c2, c3 = st.columns(3)

    with c1:

        whatif_cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.50,
            step=0.01,
            key="whatif_cgpa",
        )

        whatif_coding = st.number_input(
            "Coding Skill",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            key="whatif_coding",
        )

    with c2:

        whatif_aptitude = st.number_input(
            "Aptitude Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            key="whatif_aptitude",
        )

        whatif_communication = st.number_input(
            "Communication Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            key="whatif_communication",
        )

    with c3:

        whatif_internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            key="whatif_internships",
        )

        whatif_projects = st.number_input(
            "Projects",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
            key="whatif_projects",
        )

    if st.button(
        "Run What-If Prediction",
        type="primary",
        use_container_width=True,
        key="run_whatif",
    ):

        base_profile = {
            "age": 21,
            "gender": "Female",
            "cgpa": whatif_cgpa,
            "branch": "CSE",
            "college_tier": "Tier 1",
            "internships_count": whatif_internships,
            "projects_count": whatif_projects,
            "certifications_count": 1,
            "coding_skill_score": whatif_coding,
            "aptitude_score": whatif_aptitude,
            "communication_skill_score": whatif_communication,
            "logical_reasoning_score": 70,
            "hackathons_participated": 1,
            "github_repos": 10,
            "linkedin_connections": 300,
            "mock_interview_score": 70,
            "attendance_percentage": 80,
            "backlogs": 0,
            "extracurricular_score": 60,
            "leadership_score": 60,
            "volunteer_experience": "No",
            "sleep_hours": 7,
            "study_hours_per_day": 3,
        }

        whatif_df = profile_to_dataframe(
            base_profile
        )

        try:

            result = predict_student(
                whatif_df,
                classifier,
                regressor,
            )

            save_history(result)

            show_prediction(result)

        except Exception as error:
            st.error(
                f"What-If prediction failed: {error}"
            )

    show_history("what_if")


# ============================================================
# MODEL & DATASET
# ============================================================

with model_tab:

    st.header("Model & Dataset")

    summary = get_dataset_summary(df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Records",
            f"{summary['total_records']:,}",
        )

    with c2:
        st.metric(
            "Input Features",
            summary["total_features"],
        )

    with c3:
        st.metric(
            "Placed",
            f"{summary['placed_records']:,}",
        )

    with c4:
        st.metric(
            "Not Placed",
            f"{summary['not_placed_records']:,}",
        )

    st.subheader("Dataset Quality")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Missing Values",
            summary["missing_values"],
        )

    with c2:
        st.metric(
            "Duplicate Rows",
            summary["duplicate_rows"],
        )

    st.subheader("Placement Distribution")

    placement_distribution = (
        df["placement_status"]
        .value_counts()
        .reindex(
            ["Placed", "Not Placed"],
            fill_value=0,
        )
    )

    st.bar_chart(
        placement_distribution
    )

    st.subheader(
        "Placement Distribution by Branch"
    )

    branch_distribution = pd.crosstab(
        df["branch"],
        df["placement_status"],
    )

    st.bar_chart(
        branch_distribution
    )

    st.subheader(
        "Placement Distribution by College Tier"
    )

    tier_distribution = pd.crosstab(
        df["college_tier"],
        df["placement_status"],
    )

    st.bar_chart(
        tier_distribution
    )

    st.subheader(
        "Numerical Feature Summary"
    )

    numerical_summary = (
        df[NUMERICAL_FEATURES]
        .describe()
        .T
    )

    st.dataframe(
        numerical_summary,
        use_container_width=True,
    )

    st.subheader("Architecture")

    st.markdown(
        """
        **Stage 1 — Classification**

        Student profile → Placement Classification →
        Placement Probability

        **Stage 2 — Regression**

        Placed prediction → CTC Regression →
        Expected Starting CTC
        """
    )

    st.subheader("Preprocessing")

    st.markdown(
        """
        - Numerical features are standardized using `StandardScaler`.
        - Categorical features use `OneHotEncoder`.
        - Unknown categorical values are handled using
          `handle_unknown="ignore"`.
        - Preprocessing is contained inside the ML pipelines.
        """
    )

    st.subheader("Model Artifacts")

    st.code(
        "models/placement_classifier.pkl\n"
        "models/ctc_regressor.pkl",
        language="text",
    )

    st.subheader("Model Evaluation")

    selected_models = get_selected_models(
        evaluation
    )

    st.write("Selected Models")

    st.json(selected_models)

    st.subheader(
        "Classification Model Comparison"
    )

    classification_results = (
        get_classification_results(
            evaluation
        )
    )

    if isinstance(
        classification_results,
        pd.DataFrame
    ):
        st.dataframe(
            classification_results,
            use_container_width=True,
        )
    else:
        st.dataframe(
            pd.DataFrame(
                classification_results
            ).T,
            use_container_width=True,
        )

    st.subheader(
        "Detailed Classification Report"
    )

    classification_report = (
        get_classification_report(
            evaluation
        )
    )

    if isinstance(
        classification_report,
        pd.DataFrame
    ):
        st.dataframe(
            classification_report,
            use_container_width=True,
        )
    else:
        st.json(
            classification_report
        )

    st.subheader("Confusion Matrix")

    confusion_matrix = get_confusion_matrix(
        evaluation
    )

    st.dataframe(
        pd.DataFrame(confusion_matrix),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Regression Model Comparison"
    )

    regression_results = (
        get_regression_results(
            evaluation
        )
    )

    if isinstance(
        regression_results,
        pd.DataFrame
    ):
        st.dataframe(
            regression_results,
            use_container_width=True,
        )
    else:
        st.dataframe(
            pd.DataFrame(
                regression_results
            ).T,
            use_container_width=True,
        )

    st.subheader(
        "Regression Error Statistics"
    )

    regression_errors = (
        get_regression_error_statistics(
            evaluation
        )
    )

    st.json(regression_errors)

    st.subheader(
        "Classification Cross-Validation"
    )

    cv_results = get_cross_validation_results(
        evaluation
    )

    classification_cv = cv_results.get(
        "classification",
        {},
    )

    if classification_cv:
        st.dataframe(
            pd.DataFrame(
                classification_cv
            ).T,
            use_container_width=True,
        )

    st.subheader(
        "Regression Cross-Validation"
    )

    regression_cv = cv_results.get(
        "regression",
        {},
    )

    if regression_cv:
        st.dataframe(
            pd.DataFrame(
                regression_cv
            ).T,
            use_container_width=True,
        )

    st.subheader(
        "Classification Feature Effects"
    )

    feature_effects = (
        get_classification_feature_effects(
            classifier
        )
    )

    if not feature_effects.empty:

        st.dataframe(
            feature_effects,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Logistic regression coefficients indicate the direction "
        "and magnitude of the model's linear association. "
        "They do not establish causal relationships."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

with about_tab:

    st.header("About Project")

    st.write(
        "Campus Placement & CTC Intelligence Hub is a "
        "two-stage machine learning application designed "
        "to analyze student placement-related attributes."
    )

    st.subheader("How the System Works")

    st.markdown(
        """
        1. Student profile is entered.
        2. Classification predicts placement status.
        3. The model provides an estimated probability of the
           Placed class.
        4. If the prediction is Placed, regression estimates
           expected starting CTC.
        5. If the prediction is Not Placed, CTC is not generated.
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - Predictions are estimates, not guarantees.
        - Predicted CTC is not a guaranteed salary offer.
        - Model relationships should not automatically be interpreted
          as causal relationships.
        - Real-world placement outcomes can depend on factors
          not represented in the dataset.
        """
    )

    st.subheader("Technology Stack")

    st.write(
        "Python • Pandas • NumPy • Scikit-learn • "
        "Streamlit • Joblib • Git/GitHub"
    )

    st.subheader("Team")

    team = pd.DataFrame(
        {
            "Team Member": [
                "Chinmay Patil",
                "Sanika Mhatre",
                "Siddharth Parchande",
                "Dhruva",
            ]
        }
    )

    st.dataframe(
        team,
        use_container_width=True,
        hide_index=True,
    )