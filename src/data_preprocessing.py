"""Leakage-safe preprocessing for telecom-to-bank churn transfer learning."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

TELECOM_FEATURES = [
    "Account length",
    "Total day charge",
    "Customer service calls",
    "International plan",
    "Voice mail plan",
    "Total day minutes",
    "Total intl calls",
]

BANK_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]

TELECOM_NUMERIC = [
    "Account length",
    "Total day charge",
    "Customer service calls",
    "Total day minutes",
    "Total intl calls",
]

TELECOM_CATEGORICAL = [
    "International plan",
    "Voice mail plan",
]

BANK_NUMERIC = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Satisfaction Score",
    "Point Earned",
]

BANK_CATEGORICAL = [
    "Geography",
    "Gender",
    "Card Type",
]


def _split_data(X, y):
    """Create stratified train, validation and test partitions."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    validation_fraction = VALIDATION_SIZE / (
        TEST_SIZE + VALIDATION_SIZE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_fraction,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_preprocessor(numeric_features, categorical_features):
    """Build a preprocessing pipeline for one domain."""

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _prepare_domain(
    X,
    y,
    numeric_features,
    categorical_features,
):
    """Split data and fit preprocessing only on training data."""

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = _split_data(X, y)

    preprocessor = _build_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "X_train": np.asarray(X_train_processed, dtype=np.float32),
        "X_val": np.asarray(X_val_processed, dtype=np.float32),
        "X_test": np.asarray(X_test_processed, dtype=np.float32),
        "y_train": y_train.to_numpy(dtype=np.float32),
        "y_val": y_val.to_numpy(dtype=np.float32),
        "y_test": y_test.to_numpy(dtype=np.float32),
        "preprocessor": preprocessor,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
    }


def load_and_preprocess_data(
    telecom_path="telecom dataset.csv",
    bank_path="bank dataset.csv",
):
    """Load both domains and return independently processed datasets."""

    telecom_path = Path(telecom_path)
    bank_path = Path(bank_path)

    if not telecom_path.is_file():
        raise FileNotFoundError(f"Telecom dataset not found: {telecom_path}")

    if not bank_path.is_file():
        raise FileNotFoundError(f"Bank dataset not found: {bank_path}")

    telecom = pd.read_csv(telecom_path)
    bank = pd.read_csv(bank_path)

    required_telecom = TELECOM_FEATURES + ["Churn"]
    required_bank = BANK_FEATURES + ["Exited"]

    missing_telecom = set(required_telecom) - set(telecom.columns)
    missing_bank = set(required_bank) - set(bank.columns)

    if missing_telecom:
        raise ValueError(
            f"Missing telecom columns: {sorted(missing_telecom)}"
        )

    if missing_bank:
        raise ValueError(
            f"Missing bank columns: {sorted(missing_bank)}"
        )

    telecom_X = telecom[TELECOM_FEATURES].copy()
    telecom_y = telecom["Churn"].astype(int)

    bank_X = bank[BANK_FEATURES].copy()
    bank_y = bank["Exited"].astype(int)

    if telecom_X.isnull().any().any() or telecom_y.isnull().any():
        raise ValueError("Telecom dataset contains missing values.")

    if bank_X.isnull().any().any() or bank_y.isnull().any():
        raise ValueError("Bank dataset contains missing values.")

    if set(telecom_y.unique()) != {0, 1}:
        raise ValueError("Telecom target must contain both classes 0 and 1.")

    if set(bank_y.unique()) != {0, 1}:
        raise ValueError("Bank target must contain both classes 0 and 1.")

    telecom_data = _prepare_domain(
        telecom_X,
        telecom_y,
        TELECOM_NUMERIC,
        TELECOM_CATEGORICAL,
    )

    bank_data = _prepare_domain(
        bank_X,
        bank_y,
        BANK_NUMERIC,
        BANK_CATEGORICAL,
    )

    return {
        "telecom": telecom_data,
        "bank": bank_data,
    }


if __name__ == "__main__":
    datasets = load_and_preprocess_data()

    for domain, data in datasets.items():
        print(f"\n{domain.upper()} DATA")
        print("Training:", data["X_train"].shape)
        print("Validation:", data["X_val"].shape)
        print("Testing:", data["X_test"].shape)
        print("Features after encoding:", len(data["feature_names"]))
        print("Training churn distribution:", np.unique(
            data["y_train"], return_counts=True
        ))
