import pandas as pd

from .config import (
    DATA_PATH,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET
)


df = pd.read_csv(DATA_PATH)

print(
    "Dataset shape:",
    df.shape
)

print(
    "\nMissing values:",
    df.isna().sum().sum()
)

print(
    "\nDuplicate rows:",
    df.duplicated().sum()
)

print(
    "\nPlacement distribution:"
)

print(
    df[
        CLASSIFICATION_TARGET
    ].value_counts()
)

print(
    "\nSalary summary:"
)

print(
    df[
        REGRESSION_TARGET
    ].describe()
)