import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .config import (
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
    EVALUATION_PATH
)

from .data_loader import (
    load_dataset,
    get_classification_data,
    get_regression_data
)

from .preprocessing import (
    create_preprocessor,
    create_pipeline
)


def evaluate_classification():
    df = load_dataset()
    X, y = get_classification_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_models = {}

    for model_name, model in models.items():

        pipeline = create_pipeline(
            create_preprocessor(),
            model
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        classes = list(
            pipeline.named_steps["model"].classes_
        )

        placed_index = classes.index("Placed")

        y_probability = probabilities[:, placed_index]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            pos_label="Placed"
        )
        recall = recall_score(
            y_test,
            y_pred,
            pos_label="Placed"
        )
        f1 = f1_score(
            y_test,
            y_pred,
            pos_label="Placed"
        )
        roc_auc = roc_auc_score(
            (y_test == "Placed").astype(int),
            y_probability
        )

        results[model_name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
                labels=["Not Placed", "Placed"]
            ).tolist()
        }

        trained_models[model_name] = pipeline

    selected_model_name = max(
        results,
        key=lambda name: results[name]["roc_auc"]
    )

    return (
        results,
        trained_models,
        selected_model_name,
        X,
        y
    )


def evaluate_classification_cross_validation(X, y):

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    cv_results = {}

    for model_name, model in models.items():

        pipeline = create_pipeline(
            create_preprocessor(),
            model
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=5,
            scoring=[
                "accuracy",
                "f1_macro",
                "roc_auc"
            ],
            n_jobs=-1
        )

        cv_results[model_name] = {
            "accuracy_mean": float(
                scores["test_accuracy"].mean()
            ),
            "accuracy_std": float(
                scores["test_accuracy"].std()
            ),
            "f1_mean": float(
                scores["test_f1_macro"].mean()
            ),
            "f1_std": float(
                scores["test_f1_macro"].std()
            ),
            "roc_auc_mean": float(
                scores["test_roc_auc"].mean()
            ),
            "roc_auc_std": float(
                scores["test_roc_auc"].std()
            )
        }

    return cv_results


def evaluate_regression():

    df = load_dataset()

    X, y, placed_df = get_regression_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_models = {}

    for model_name, model in models.items():

        pipeline = create_pipeline(
            create_preprocessor(),
            model
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        errors = y_test.to_numpy() - y_pred

        results[model_name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "error_mean": float(errors.mean()),
            "error_std": float(errors.std()),
            "error_min": float(errors.min()),
            "error_max": float(errors.max()),
            "absolute_error_mean": float(
                np.abs(errors).mean()
            ),
            "absolute_error_median": float(
                np.median(np.abs(errors))
            )
        }

        trained_models[model_name] = pipeline

    selected_model_name = max(
        results,
        key=lambda name: results[name]["r2"]
    )

    return (
        results,
        trained_models,
        selected_model_name,
        X,
        y,
        placed_df
    )


def evaluate_regression_cross_validation(X, y):

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    cv_results = {}

    for model_name, model in models.items():

        pipeline = create_pipeline(
            create_preprocessor(),
            model
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=5,
            scoring=[
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "r2"
            ],
            n_jobs=-1
        )

        cv_results[model_name] = {
            "mae_mean": float(
                -scores[
                    "test_neg_mean_absolute_error"
                ].mean()
            ),
            "mae_std": float(
                scores[
                    "test_neg_mean_absolute_error"
                ].std()
            ),
            "rmse_mean": float(
                -scores[
                    "test_neg_root_mean_squared_error"
                ].mean()
            ),
            "rmse_std": float(
                scores[
                    "test_neg_root_mean_squared_error"
                ].std()
            ),
            "r2_mean": float(
                scores["test_r2"].mean()
            ),
            "r2_std": float(
                scores["test_r2"].std()
            )
        }

    return cv_results


def generate_evaluation_report():

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    (
        classification_results,
        classification_models,
        selected_classification,
        classification_X,
        classification_y
    ) = evaluate_classification()

    classification_cv = (
        evaluate_classification_cross_validation(
            classification_X,
            classification_y
        )
    )

    (
        regression_results,
        regression_models,
        selected_regression,
        regression_X,
        regression_y,
        placed_df
    ) = evaluate_regression()

    regression_cv = (
        evaluate_regression_cross_validation(
            regression_X,
            regression_y
        )
    )

    evaluation_results = {
        "classification": classification_results,
        "classification_cross_validation": classification_cv,
        "regression": regression_results,
        "regression_cross_validation": regression_cv,
        "regression_error_statistics": {
            model_name: {
                "error_mean": values["error_mean"],
                "error_std": values["error_std"],
                "error_min": values["error_min"],
                "error_max": values["error_max"],
                "absolute_error_mean": values[
                    "absolute_error_mean"
                ],
                "absolute_error_median": values[
                    "absolute_error_median"
                ]
            }
            for model_name, values
            in regression_results.items()
        },
        "selected_classification_model": selected_classification,
        "selected_regression_model": selected_regression,
        "dataset_information": {
            "total_records": int(len(load_dataset())),
            "placed_records_for_regression": int(
                len(placed_df)
            )
        }
    }

    os.makedirs(
        os.path.dirname(EVALUATION_PATH),
        exist_ok=True
    )

    with open(
        EVALUATION_PATH,
        "w"
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    print("\nClassification Results")

    for model_name, values in classification_results.items():
        print(
            f"{model_name}: "
            f"Accuracy={values['accuracy']:.4f}, "
            f"F1={values['f1_score']:.4f}, "
            f"ROC-AUC={values['roc_auc']:.4f}"
        )

    print("\nClassification Cross-Validation")

    for model_name, values in classification_cv.items():
        print(
            f"{model_name}: "
            f"Accuracy={values['accuracy_mean']:.4f}, "
            f"F1={values['f1_mean']:.4f}, "
            f"ROC-AUC={values['roc_auc_mean']:.4f}"
        )

    print("\nRegression Results")

    for model_name, values in regression_results.items():
        print(
            f"{model_name}: "
            f"MAE={values['mae']:.4f}, "
            f"RMSE={values['rmse']:.4f}, "
            f"R²={values['r2']:.4f}"
        )

    print("\nRegression Cross-Validation")

    for model_name, values in regression_cv.items():
        print(
            f"{model_name}: "
            f"MAE={values['mae_mean']:.4f}, "
            f"RMSE={values['rmse_mean']:.4f}, "
            f"R²={values['r2_mean']:.4f}"
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Selected classification model: "
        f"{selected_classification}"
    )

    print(
        f"Selected regression model: "
        f"{selected_regression}"
    )

    print(
        f"Saved evaluation report to: "
        f"{EVALUATION_PATH}"
    )

    return evaluation_results


# ------------------------------------------------------------------
# Existing helper functions used by the application
# ------------------------------------------------------------------

def load_evaluation_results():
    if not os.path.exists(EVALUATION_PATH):
        return None

    with open(
        EVALUATION_PATH,
        "r"
    ) as file:
        return json.load(file)


def get_classification_results(evaluation_results):
    if evaluation_results is None:
        return pd.DataFrame()

    return pd.DataFrame(
        evaluation_results.get(
            "classification",
            []
        )
    ).transpose()


def get_classification_report(evaluation_results):
    if evaluation_results is None:
        return pd.DataFrame()

    reports = {}

    for model_name, values in evaluation_results.get(
        "classification",
        {}
    ).items():

        report = values.get(
            "classification_report",
            {}
        )

        reports[model_name] = report

    if not reports:
        return pd.DataFrame()

    selected_model = evaluation_results.get(
        "selected_classification_model"
    )

    if selected_model in reports:
        return pd.DataFrame(
            reports[selected_model]
        ).transpose()

    return pd.DataFrame()


def get_confusion_matrix(evaluation_results):
    if evaluation_results is None:
        return None

    selected_model = evaluation_results.get(
        "selected_classification_model"
    )

    return evaluation_results.get(
        "classification",
        {}
    ).get(
        selected_model,
        {}
    ).get(
        "confusion_matrix"
    )


def get_regression_results(evaluation_results):
    if evaluation_results is None:
        return pd.DataFrame()

    return pd.DataFrame(
        evaluation_results.get(
            "regression",
            {}
        )
    ).transpose()


def get_regression_error_statistics(evaluation_results):
    if evaluation_results is None:
        return {}

    return evaluation_results.get(
        "regression_error_statistics",
        {}
    )


def get_selected_models(evaluation_results):
    if evaluation_results is None:
        return {
            "classification": None,
            "regression": None
        }

    return {
        "classification": evaluation_results.get(
            "selected_classification_model"
        ),
        "regression": evaluation_results.get(
            "selected_regression_model"
        )
    }


def get_cross_validation_results(evaluation_results):
    if evaluation_results is None:
        return {
            "classification": {},
            "regression": {}
        }

    return {
        "classification": evaluation_results.get(
            "classification_cross_validation",
            {}
        ),
        "regression": evaluation_results.get(
            "regression_cross_validation",
            {}
        )
    }


if __name__ == "__main__":
    generate_evaluation_report()