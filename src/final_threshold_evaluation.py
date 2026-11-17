
"""Final bank-model evaluation using validation-selected thresholds."""

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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.data_preprocessing import load_and_preprocess_data


ARTIFACT_DIR = Path("artifacts")

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


def evaluate_model(model_name, checkpoint, threshold, bank_data):
    """Evaluate one model on the untouched bank test dataset."""

    model_path = ARTIFACT_DIR / checkpoint

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"\nEvaluating: {model_name}")
    print(f"Selected threshold: {threshold:.2f}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    X_test = bank_data["X_test"]
    y_true = np.asarray(bank_data["y_test"]).astype(int).ravel()

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    if len(probabilities) != len(y_true):
        raise ValueError("Test prediction count mismatch.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite test predictions found.")

    y_pred = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    results = {
        "model": model_name,
        "domain": "Bank",
        "threshold": float(threshold),
        "test_samples": int(len(y_true)),
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
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
            output_dict=True,
        ),
    }

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1_score']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"PR-AUC:    {results['pr_auc']:.4f}")
    print(f"Confusion matrix:\n{matrix}")

    return results


def main():
    print("Loading final test dataset...")

    datasets = load_and_preprocess_data()
    bank_data = datasets["bank"]

    threshold_path = ARTIFACT_DIR / "recommended_thresholds.json"

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold recommendations not found: {threshold_path}"
        )

    with open(threshold_path, "r") as file:
        recommendations = json.load(file)

    results = []

    for model_name, checkpoint in BANK_MODELS:
        if model_name not in recommendations:
            raise KeyError(
                f"Missing recommended threshold for {model_name}"
            )

        threshold = recommendations[model_name]["threshold"]

        result = evaluate_model(
            model_name,
            checkpoint,
            threshold,
            bank_data,
        )

        results.append(result)

    # Save complete final test results.
    json_path = ARTIFACT_DIR / "final_threshold_evaluation.json"

    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)

    # Save compact final metrics.
    csv_path = ARTIFACT_DIR / "final_threshold_metrics.csv"

    fields = [
        "model",
        "threshold",
        "test_samples",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result[field]
                for field in fields
            })

    print("\nFinal threshold evaluation completed.")
    print(f"Full report: {json_path}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
