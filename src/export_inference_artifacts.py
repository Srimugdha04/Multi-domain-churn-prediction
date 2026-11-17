
"""Export fitted preprocessing and inference configuration."""

import json
import joblib
from pathlib import Path

from src.data_preprocessing import (
    load_and_preprocess_data,
    BANK_FEATURES,
)

ARTIFACT_DIR = Path("artifacts")

MODEL_NAME = "Random Transfer - Fine-tuned"
MODEL_FILE = "random_transfer_control_finetuned_best.keras"


def main():
    print("Loading fitted bank preprocessing pipeline...")

    datasets = load_and_preprocess_data()
    bank_data = datasets["bank"]

    preprocessor_path = ARTIFACT_DIR / "bank_preprocessor.pkl"

    joblib.dump(
        bank_data["preprocessor"],
        preprocessor_path,
    )

    threshold_path = ARTIFACT_DIR / "recommended_thresholds.json"

    with open(threshold_path, "r") as file:
        thresholds = json.load(file)

    threshold = thresholds[MODEL_NAME]["threshold"]

    config = {
        "model_name": MODEL_NAME,
        "model_file": MODEL_FILE,
        "threshold": float(threshold),
        "input_features": BANK_FEATURES,
        "processed_feature_count": len(bank_data["feature_names"]),
        "positive_class": 1,
        "positive_class_label": "Churn",
    }

    config_path = ARTIFACT_DIR / "inference_config.json"

    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

    print("\nInference artifacts exported successfully.")
    print(f"Preprocessor: {preprocessor_path}")
    print(f"Configuration: {config_path}")
    print(f"Selected model: {MODEL_NAME}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Processed features: {len(bank_data['feature_names'])}")


if __name__ == "__main__":
    main()
