import pandas as pd


def get_feature_names(classifier):
    """
    Return the transformed feature names produced by
    the classifier's preprocessing pipeline.
    """

    preprocessor = classifier.named_steps["preprocessor"]

    try:
        return list(
            preprocessor.get_feature_names_out()
        )

    except Exception:
        return []


def get_classification_feature_effects(
    classifier,
    top_n=15
):
    """
    Return the most influential transformed features
    for the classification model.

    For linear models:
        Uses model coefficients.

    For tree-based models:
        Uses feature importances.

    Returns:
        pandas.DataFrame
    """

    model = classifier.named_steps["model"]

    feature_names = get_feature_names(
        classifier
    )

    if not feature_names:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Linear Model
    # --------------------------------------------------------

    if hasattr(model, "coef_"):

        coefficients = model.coef_[0]

        if len(feature_names) != len(coefficients):
            return pd.DataFrame()

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients
            }
        )

        result["Absolute Effect"] = (
            result["Coefficient"].abs()
        )

        result = result.sort_values(
            "Absolute Effect",
            ascending=False
        )

        return result.head(top_n).reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # Tree-Based Model
    # --------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        if len(feature_names) != len(importances):
            return pd.DataFrame()

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )

        result = result.sort_values(
            "Importance",
            ascending=False
        )

        return result.head(top_n).reset_index(
            drop=True
        )

    return pd.DataFrame()