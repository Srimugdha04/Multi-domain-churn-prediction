
"""Evaluate trained telecom and bank churn prediction models."""

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
THRESHOLD = 0.5


def evaluate_model(model_name, checkpoint, data, domain):
    """Evaluate one saved model on its untouched test dataset."""

    checkpoint = ARTIFACT_DIR / checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    print(f"\nEvaluating: {model_name}")

    model = tf.keras.models.load_model(
        checkpoint,
        compile=False,
    )

    X_test = data["X_test"]
    y_true = np.asarray(data["y_test"]).astype(int).ravel()

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    if len(probabilities) != len(y_true):
        raise ValueError(
            f"Prediction count mismatch for {model_name}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            f"Non-finite predictions found in {model_name}"
        )

    y_pred = (probabilities >= THRESHOLD).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    results = {
        "model": model_name,
        "domain": domain,
        "threshold": THRESHOLD,
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
    print("Loading test datasets...")

    datasets = load_and_preprocess_data()

    telecom = datasets["telecom"]
    bank = datasets["bank"]

    experiments = [
        (
            "Telecom Source",
            "telecom_source_best.keras",
            telecom,
            "Telecom",
        ),
        (
            "Bank Baseline",
            "bank_baseline_best.keras",
            bank,
            "Bank",
        ),
        (
            "Pretrained Transfer - Stage 1",
            "pretrained_transfer_stage1_best.keras",
            bank,
            "Bank",
        ),
        (
            "Pretrained Transfer - Fine-tuned",
            "pretrained_transfer_finetuned_best.keras",
            bank,
            "Bank",
        ),
        (
            "Random Transfer - Stage 1",
            "random_transfer_control_stage1_best.keras",
            bank,
            "Bank",
        ),
        (
            "Random Transfer - Fine-tuned",
            "random_transfer_control_finetuned_best.keras",
            bank,
            "Bank",
        ),
    ]

    results = []

    for name, checkpoint, data, domain in experiments:
        result = evaluate_model(
            name,
            checkpoint,
            data,
            domain,
        )
        results.append(result)

    # Save the complete evaluation report.
    json_path = ARTIFACT_DIR / "evaluation_report.json"

    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)

    # Save a compact metrics comparison table.
    csv_path = ARTIFACT_DIR / "metrics_summary.csv"

    summary_fields = [
        "model",
        "domain",
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
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result[field]
                for field in summary_fields
            })

    print("\nEvaluation completed.")
    print(f"Full report: {json_path}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
