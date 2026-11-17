
"""Generate model comparison charts from final test metrics."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("artifacts/final_threshold_metrics.csv")
PLOT_DIR = Path("artifacts/plots")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

metrics = {
    "accuracy": "Accuracy Comparison",
    "precision": "Precision Comparison",
    "recall": "Recall Comparison",
    "f1_score": "F1-score Comparison",
}

for column, title in metrics.items():
    plt.figure(figsize=(11, 6))

    plt.bar(
        df["model"],
        df[column] * 100,
    )

    plt.title(title)
    plt.ylabel("Score (%)")
    plt.xlabel("Model")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    filename = f"{column}_comparison.png"
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()

    print(f"Saved: {PLOT_DIR / filename}")

print("\nAll comparison charts generated successfully.")
