import joblib
import pandas as pd

from .config import (
    CLASSIFIER_PATH,
    REGRESSOR_PATH
)


def load_models():

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    regressor = joblib.load(
        REGRESSOR_PATH
    )

    return classifier, regressor


def predict_student(
    student_data: pd.DataFrame,
    classifier,
    regressor
):

    placement_prediction = classifier.predict(
        student_data
    )[0]

    classes = list(
        classifier.named_steps[
            "model"
        ].classes_
    )

    placed_index = classes.index(
        "Placed"
    )

    placement_probability = (
        classifier.predict_proba(
            student_data
        )[0, placed_index]
    )

    predicted_ctc = None

    if placement_prediction == "Placed":

        predicted_ctc = regressor.predict(
            student_data
        )[0]

        predicted_ctc = max(
            0,
            float(predicted_ctc)
        )

    return {
        "placement_prediction":
            placement_prediction,

        "placement_probability":
            float(
                placement_probability
            ),

        "predicted_ctc":
            predicted_ctc
    }


def predict_placement_probability(
    student_data,
    classifier
):

    classes = list(
        classifier.named_steps[
            "model"
        ].classes_
    )

    placed_index = classes.index(
        "Placed"
    )

    probability = (
        classifier.predict_proba(
            student_data
        )[0, placed_index]
    )

    return float(probability)