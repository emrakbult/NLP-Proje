from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_matcher import LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bi-encoder training and evaluation artifacts.")
    parser.add_argument("--reports-dir", default="reports", help="Directory containing report CSV/JSON files.")
    return parser.parse_args()


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, Any]] = []
        for row in reader:
            row["semantic_score"] = float(row["semantic_score"])
            rows.append(row)
        return rows


def save_line_plot(
    path: Path,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    y_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, values in series:
        plt.plot(x_values, values, marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_score_distribution(path: Path, predictions: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for label in LABELS:
        scores = [
            row["semantic_score"]
            for row in predictions
            if row["label"] == label
        ]
        plt.hist(scores, bins=20, alpha=0.55, label=label)
    plt.title("Semantic Score Distribution By Label")
    plt.xlabel("Semantic score")
    plt.ylabel("Record count")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    labels = metrics["confusion_matrix_labels"]
    matrix = metrics["confusion_matrix"]

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()

    max_value = max(max(row) for row in matrix) if matrix else 0
    threshold = max_value / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            plt.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = read_history(reports_dir / "biencoder_training_history.csv")
    predictions = read_predictions(reports_dir / "biencoder_test_predictions.csv")
    metrics = json.loads((reports_dir / "biencoder_test_metrics.json").read_text(encoding="utf-8"))

    epochs = [row["epoch"] for row in history]
    save_line_plot(
        figures_dir / "loss_total.png",
        "Train vs Validation Total Loss",
        epochs,
        [
            ("Train total loss", [row["train_total_loss"] for row in history]),
            ("Validation total loss", [row["validation_total_loss"] for row in history]),
        ],
        "Loss",
    )
    save_line_plot(
        figures_dir / "loss_cosine.png",
        "Cosine MSE Loss Curve",
        epochs,
        [
            ("Train cosine loss", [row["train_cosine_loss"] for row in history]),
            ("Validation cosine loss", [row["validation_cosine_loss"] for row in history]),
        ],
        "Loss",
    )
    save_line_plot(
        figures_dir / "loss_classification.png",
        "Classification Loss Curve",
        epochs,
        [
            ("Train classification loss", [row["train_classification_loss"] for row in history]),
            ("Validation classification loss", [row["validation_classification_loss"] for row in history]),
        ],
        "Loss",
    )
    save_line_plot(
        figures_dir / "validation_accuracy.png",
        "Validation Accuracy By Epoch",
        epochs,
        [("Validation accuracy", [row["validation_accuracy"] for row in history])],
        "Accuracy",
    )
    save_line_plot(
        figures_dir / "validation_correlations.png",
        "Validation Correlation By Epoch",
        epochs,
        [
            ("Pearson", [row["validation_pearson"] for row in history]),
            ("Spearman", [row["validation_spearman"] for row in history]),
        ],
        "Correlation",
    )
    save_score_distribution(figures_dir / "score_distribution_by_label.png", predictions)
    save_confusion_matrix(figures_dir / "confusion_matrix.png", metrics)

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
