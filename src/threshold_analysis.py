
"""Select classification thresholds using bank validation data only."""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data_preprocessing import load_and_preprocess_data


ARTIFACT_DIR = Path("artifacts")

# Thresholds to compare: 0.20, 0.21, ..., 0.80
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)

BANK_MODELS = [
    ("Bank Baseline", "bank_baseline_best.keras"),
    (
        "Pretrained Transfer - Stage 1",
        "pretrained_transfer_stage1_best.keras",
    ),
    (
        "Pretrained Transfer - Fine-tuned",
        "pretrained_transfer_finetuned_best.keras",
    ),
    (
        "Random Transfer - Stage 1",
        "random_transfer_control_stage1_best.keras",
    ),
    (
        "Random Transfer - Fine-tuned",
        "random_transfer_control_finetuned_best.keras",
    ),
]


def analyze_model(model_name, checkpoint, bank_data):
    """Compare thresholds using validation predictions."""

    model_path = ARTIFACT_DIR / checkpoint

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    print(f"\nAnalyzing: {model_name}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    X_val = bank_data["X_val"]
    y_true = np.asarray(bank_data["y_val"]).astype(int).ravel()

    probabilities = model.predict(
        X_val,
        verbose=0,
    ).ravel()

    if len(probabilities) != len(y_true):
        raise ValueError("Validation prediction count mismatch.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite validation predictions found.")

    model_results = []

    for threshold in THRESHOLDS:
        y_pred = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        result = {
            "model": model_name,
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        model_results.append(result)

    # Select the threshold with the highest validation F1-score.
    best_result = max(
        model_results,
        key=lambda item: item["f1_score"],
    )

    print(
        f"Best threshold: {best_result['threshold']:.2f} | "
        f"Validation F1: {best_result['f1_score']:.4f} | "
        f"Precision: {best_result['precision']:.4f} | "
        f"Recall: {best_result['recall']:.4f}"
    )

    return model_results, best_result


def main():
    print("Loading bank validation data...")

    datasets = load_and_preprocess_data()
    bank_data = datasets["bank"]

    all_results = []
    best_results = {}

    for model_name, checkpoint in BANK_MODELS:
        model_results, best_result = analyze_model(
            model_name,
            checkpoint,
            bank_data,
        )

        all_results.extend(model_results)
        best_results[model_name] = best_result

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save every model's results at every threshold.
    csv_path = ARTIFACT_DIR / "threshold_analysis.csv"

    fields = [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    # Save each model's recommended validation threshold.
    json_path = ARTIFACT_DIR / "recommended_thresholds.json"

    with open(json_path, "w") as file:
        json.dump(best_results, file, indent=4)

    print("\nThreshold analysis completed.")
    print(f"Detailed results: {csv_path}")
    print(f"Recommended thresholds: {json_path}")


if __name__ == "__main__":
    main()
