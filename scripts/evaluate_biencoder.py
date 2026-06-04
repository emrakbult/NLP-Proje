from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_split_path, load_records
from src.evaluation import (
    classification_accuracy,
    labels_to_scores,
    pearson_correlation,
    predict_label_from_score,
    spearman_correlation,
)
from src.neural_matcher import (
    DEFAULT_BIENCODER_MODEL_PATH,
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    ResumeJobBiEncoder,
)
from src.similarity import (
    BASE_MODEL_PATH,
    encode_texts,
    load_model,
    pairwise_cosine_similarities,
    similarity_to_percentage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the custom bi-encoder.")
    parser.add_argument("--model-dir", default=str(DEFAULT_BIENCODER_MODEL_PATH), help="Optimal model path.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full split.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum token length.")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Evaluation device.")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested_device


def safe_correlation(values_a: list[float], values_b: list[float], kind: str) -> float:
    if not values_a or len(set(values_b)) <= 1:
        return 0.0
    return pearson_correlation(values_a, values_b) if kind == "pearson" else spearman_correlation(values_a, values_b)


def confusion_matrix(true_labels: list[str], predicted_labels: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[LABEL_TO_ID[true_label]][LABEL_TO_ID[predicted_label]] += 1
    return matrix


def evaluate_optimal_model(
    records: list[dict[str, str]],
    model_dir: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ResumeJobBiEncoder.from_pretrained(model_dir).to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    semantic_scores: list[float] = []
    predicted_labels: list[str] = []
    true_labels = [record["label"] for record in records]
    target_scores = labels_to_scores(true_labels)

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            resume_texts = [record["resume_text"] for record in batch_records]
            job_texts = [record["job_description_text"] for record in batch_records]
            resume = tokenizer(
                resume_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            job = tokenizer(
                job_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            resume = {key: value.to(device) for key, value in resume.items()}
            job = {key: value.to(device) for key, value in job.items()}
            output = model.model(
                resume_input_ids=resume["input_ids"],
                resume_attention_mask=resume["attention_mask"],
                job_input_ids=job["input_ids"],
                job_attention_mask=job["attention_mask"],
            )
            probabilities = torch.softmax(output["class_logits"], dim=1)
            class_ids = torch.argmax(probabilities, dim=1).detach().cpu().tolist()
            batch_scores = (output["similarity_score"].detach().cpu() * 100).tolist()

            for record, score, class_id, row_probabilities in zip(
                batch_records,
                batch_scores,
                class_ids,
                probabilities.detach().cpu().tolist(),
            ):
                predicted_label = ID_TO_LABEL[int(class_id)]
                semantic_scores.append(float(score))
                predicted_labels.append(predicted_label)
                predictions.append(
                    {
                        "label": record["label"],
                        "predicted_label": predicted_label,
                        "semantic_score": float(score),
                        "prob_no_fit": row_probabilities[0],
                        "prob_potential_fit": row_probabilities[1],
                        "prob_good_fit": row_probabilities[2],
                    }
                )

    metrics = {
        "model": str(model_dir.relative_to(PROJECT_ROOT)),
        "records": len(records),
        "label_distribution": dict(Counter(true_labels)),
        "semantic_pearson": safe_correlation(semantic_scores, target_scores, "pearson"),
        "semantic_spearman": safe_correlation(semantic_scores, target_scores, "spearman"),
        "classification_accuracy": classification_accuracy(true_labels, predicted_labels),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels),
        "average_semantic_score_by_label": {
            label: sum(score for score, true_label in zip(semantic_scores, true_labels) if true_label == label)
            / max(sum(1 for true_label in true_labels if true_label == label), 1)
            for label in LABELS
        },
    }
    return metrics, predictions


def evaluate_embedding_model(
    records: list[dict[str, str]],
    model_path: Path,
    model_label: str,
    batch_size: int,
) -> dict[str, Any]:
    model = load_model(str(model_path))
    resume_embeddings = encode_texts(
        [record["resume_text"] for record in records],
        model,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    job_embeddings = encode_texts(
        [record["job_description_text"] for record in records],
        model,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    semantic_scores = [
        similarity_to_percentage(score)
        for score in pairwise_cosine_similarities(resume_embeddings, job_embeddings)
    ]
    true_labels = [record["label"] for record in records]
    target_scores = labels_to_scores(true_labels)
    predicted_labels = [predict_label_from_score(score) for score in semantic_scores]
    return {
        "model_label": model_label,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "semantic_pearson": safe_correlation(semantic_scores, target_scores, "pearson"),
        "semantic_spearman": safe_correlation(semantic_scores, target_scores, "spearman"),
        "threshold_accuracy": classification_accuracy(true_labels, predicted_labels),
        "records": len(records),
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    reports_dir = PROJECT_ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(get_split_path(args.split), limit=None if args.limit == 0 else args.limit)
    device = resolve_device(args.device)

    metrics, predictions = evaluate_optimal_model(
        records,
        model_dir=model_dir,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    (reports_dir / "biencoder_test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_predictions(reports_dir / "biencoder_test_predictions.csv", predictions)

    comparison_rows = [
        evaluate_embedding_model(records, BASE_MODEL_PATH, "Baseline local MiniLM", args.batch_size),
        {
            "model_label": "New optimal custom bi-encoder",
            "model_path": str(model_dir.relative_to(PROJECT_ROOT)),
            "semantic_pearson": metrics["semantic_pearson"],
            "semantic_spearman": metrics["semantic_spearman"],
            "threshold_accuracy": metrics["classification_accuracy"],
            "records": len(records),
        },
    ]
    write_comparison(reports_dir / "model_comparison_metrics.csv", comparison_rows)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics and predictions to: {reports_dir}")


if __name__ == "__main__":
    main()
