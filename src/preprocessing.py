from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def create_preprocessor():
    """
    Create the preprocessing transformer.

    Numerical features:
    - StandardScaler

    Categorical features:
    - OneHotEncoder
    - Unknown categories are ignored
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )


def create_pipeline(preprocessor, model):
    """
    Combine preprocessing and a machine-learning model
    into one Scikit-Learn pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )