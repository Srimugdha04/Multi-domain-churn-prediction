
"""Reusable bank customer churn inference module."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

NUMERIC_FEATURES = [
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


@lru_cache(maxsize=1)
def load_inference_components():
    """Load model, fitted preprocessor and configuration once."""

    config_path = ARTIFACT_DIR / "inference_config.json"
    preprocessor_path = ARTIFACT_DIR / "bank_preprocessor.pkl"

    with open(config_path, "r") as file:
        config = json.load(file)

    model_path = ARTIFACT_DIR / config["model_file"]

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor, config


def predict_churn(customer_data):
    """
    Predict churn for one bank customer.

    customer_data must be a dictionary containing all
    13 raw bank input features.
    """

    if not isinstance(customer_data, dict):
        raise TypeError("Customer data must be a dictionary.")

    model, preprocessor, config = load_inference_components()

    required_features = config["input_features"]

    missing = [
        feature
        for feature in required_features
        if feature not in customer_data
    ]

    if missing:
        raise ValueError(f"Missing customer fields: {missing}")

    # Keep only the expected features and preserve training order.
    customer_df = pd.DataFrame(
        [[customer_data[name] for name in required_features]],
        columns=required_features,
    )

    # Convert numeric inputs and reject invalid values.
    for feature in NUMERIC_FEATURES:
        customer_df[feature] = pd.to_numeric(
            customer_df[feature],
            errors="raise",
        )

    if customer_df.isnull().any().any():
        raise ValueError("Customer inputs cannot contain empty values.")

    numeric_values = customer_df[NUMERIC_FEATURES].to_numpy(
        dtype=float
    )

    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("Numeric inputs must be finite numbers.")

    # Apply the exact fitted preprocessing used during training.
    processed_data = preprocessor.transform(customer_df)

    expected_features = config["processed_feature_count"]

    if processed_data.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} processed features, "
            f"received {processed_data.shape[1]}."
        )

    # Predict churn probability.
    probability = float(
        model.predict(
            np.asarray(processed_data, dtype=np.float32),
            verbose=0,
        ).ravel()[0]
    )

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability.")

    threshold = float(config["threshold"])
    predicted_class = int(probability >= threshold)

    # Dashboard risk categories.
    if probability < 0.30:
        risk_level = "Low"
    elif probability < threshold:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "customer_id": customer_data.get("CustomerId"),
        "churn_probability": probability,
        "churn_probability_percent": round(probability * 100, 2),
        "predicted_class": predicted_class,
        "prediction": (
            "Churn" if predicted_class == 1 else "No Churn"
        ),
        "risk_level": risk_level,
        "threshold": threshold,
        "model_name": config["model_name"],
    }
