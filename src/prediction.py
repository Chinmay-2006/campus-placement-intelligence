import joblib
import pandas as pd

from .config import (
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
    FEATURE_COLUMNS
)


def load_models():
    """
    Load the persisted classification and regression pipelines.
    """

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    return classifier, regressor


def validate_student_data(student_data):
    """
    Validate that the input contains all required features.
    """

    if not isinstance(student_data, pd.DataFrame):
        raise TypeError(
            "student_data must be a pandas DataFrame."
        )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in student_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return student_data[FEATURE_COLUMNS].copy()


def predict_placement_probability(
    student_data,
    classifier
):
    """
    Predict the probability that a student will be placed.
    """

    student_data = validate_student_data(
        student_data
    )

    probabilities = classifier.predict_proba(
        student_data
    )

    classes = list(
        classifier.named_steps["model"].classes_
    )

    placed_index = classes.index("Placed")

    probability = probabilities[
        0,
        placed_index
    ]

    return float(probability)


def predict_student(
    student_data: pd.DataFrame,
    classifier,
    regressor
):
    """
    Run the complete two-stage prediction pipeline.

    Stage 1:
        Predict placement status and placement probability.

    Stage 2:
        Predict CTC only when the placement prediction is Placed.
    """

    student_data = validate_student_data(
        student_data
    )

    placement_prediction = classifier.predict(
        student_data
    )[0]

    placement_probability = (
        predict_placement_probability(
            student_data,
            classifier
        )
    )

    predicted_ctc = None

    if placement_prediction == "Placed":

        predicted_ctc = regressor.predict(
            student_data
        )[0]

        predicted_ctc = max(
            0.0,
            float(predicted_ctc)
        )

    return {
        "placement_prediction":
            placement_prediction,

        "placement_probability":
            placement_probability,

        "predicted_ctc":
            predicted_ctc
    }