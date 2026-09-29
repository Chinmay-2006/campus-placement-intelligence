import pandas as pd

from .config import (
    DATA_PATH,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    FEATURE_COLUMNS
)


def load_dataset():
    return pd.read_csv(DATA_PATH)


def get_classification_data(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[CLASSIFICATION_TARGET].copy()

    return X, y


def get_regression_data(df):
    placed_df = df[
        df[CLASSIFICATION_TARGET] == "Placed"
    ].copy()

    X = placed_df[FEATURE_COLUMNS].copy()
    y = placed_df[REGRESSION_TARGET].copy()

    return X, y, placed_df


def get_dataset_summary(df):
    return {
        "total_records": len(df),
        "total_features": len(FEATURE_COLUMNS),
        "placed_records": int(
            (df[CLASSIFICATION_TARGET] == "Placed").sum()
        ),
        "not_placed_records": int(
            (df[CLASSIFICATION_TARGET] == "Not Placed").sum()
        ),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }