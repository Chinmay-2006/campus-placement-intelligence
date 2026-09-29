import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .config import REGRESSOR_PATH
from .data_loader import load_dataset, get_regression_data
from .preprocessing import create_preprocessor, create_pipeline


def train_regression_models():
    # Load dataset
    df = load_dataset()

    # Use only placed students for CTC prediction
    X, y, placed_df = get_regression_data(df)

    print(f"Placed students used for regression: {len(placed_df)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Models
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

        print(f"\nTraining: {model_name}")

        # Fresh preprocessing pipeline
        preprocessor = create_preprocessor()
        pipeline = create_pipeline(preprocessor, model)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(
            y_test,
            y_pred
        ) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }

        trained_models[model_name] = pipeline

        print(f"MAE : {mae:.4f} LPA")
        print(f"RMSE: {rmse:.4f} LPA")
        print(f"R²  : {r2:.4f}")

    # Select model using R²
    selected_model_name = max(
        results,
        key=lambda name: results[name]["r2"]
    )

    selected_model = trained_models[selected_model_name]

    # Save selected regressor
    joblib.dump(
        selected_model,
        REGRESSOR_PATH
    )

    print("\n" + "=" * 60)
    print("CTC REGRESSION COMPLETE")
    print("=" * 60)

    print(f"Selected model: {selected_model_name}")
    print(
        f"R²: "
        f"{results[selected_model_name]['r2']:.4f}"
    )
    print(f"Saved to: {REGRESSOR_PATH}")

    return results, selected_model_name


if __name__ == "__main__":
    train_regression_models()