import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from .config import (
    DATA_PATH,
    CLASSIFIER_PATH,
    CLASSIFICATION_TARGET
)

from .data_loader import load_dataset, get_classification_data
from .preprocessing import create_preprocessor, create_pipeline


def train_classification_models():
    # Load dataset
    df = load_dataset()

    # Separate features and target
    X, y = get_classification_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Models
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

        print(f"\nTraining: {model_name}")

        # Create a fresh preprocessing pipeline for each model
        preprocessor = create_preprocessor()
        pipeline = create_pipeline(preprocessor, model)

        # Train
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_probability = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
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

        # ROC-AUC
        classes = list(
            pipeline.named_steps["model"].classes_
        )
        placed_index = classes.index("Placed")

        y_probability_placed = pipeline.predict_proba(
            X_test
        )[:, placed_index]

        roc_auc = roc_auc_score(
            (y_test == "Placed").astype(int),
            y_probability_placed
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=["Not Placed", "Placed"]
        )

        results[model_name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }

        trained_models[model_name] = pipeline

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

    # Select model using ROC-AUC
    selected_model_name = max(
        results,
        key=lambda name: results[name]["roc_auc"]
    )

    selected_model = trained_models[selected_model_name]

    # Save selected classifier
    joblib.dump(
        selected_model,
        CLASSIFIER_PATH
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)

    print(f"Selected model: {selected_model_name}")
    print(
        f"ROC-AUC: "
        f"{results[selected_model_name]['roc_auc']:.4f}"
    )

    print(f"Saved to: {CLASSIFIER_PATH}")

    return results, selected_model_name


if __name__ == "__main__":
    train_classification_models()