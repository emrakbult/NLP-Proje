from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_split_path, load_records
from src.evaluation import (
    average_scores_by_label,
    classification_accuracy,
    labels_to_scores,
    pearson_correlation,
    score_records,
    spearman_correlation,
)
from src.similarity import DEFAULT_MODEL_NAME, load_model
from src.skill_extractor import load_skills


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate resume-job matching scores.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of records to evaluate. Use 0 to evaluate the whole split.",
    )
    parser.add_argument(
        "--sample-mode",
        default="balanced",
        choices=["balanced", "first", "random"],
        help="How records are selected when --limit is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoder batch size.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence Transformer model name.")
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional path to save per-example evaluation results.",
    )
    return parser.parse_args()


def select_records(
    records: list[dict[str, str]],
    limit: int | None,
    sample_mode: str,
    seed: int,
) -> list[dict[str, str]]:
    """Select records for evaluation."""

    if limit is None or limit >= len(records):
        return records

    if sample_mode == "first":
        return records[:limit]

    rng = random.Random(seed)

    if sample_mode == "random":
        sampled_records = records[:]
        rng.shuffle(sampled_records)
        return sampled_records[:limit]

    grouped_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped_records[record["label"]].append(record)

    for group in grouped_records.values():
        rng.shuffle(group)

    label_order = ["No Fit", "Potential Fit", "Good Fit"]
    selected_records: list[dict[str, str]] = []
    group_index = 0

    while len(selected_records) < limit:
        added_record = False
        for label in label_order:
            group = grouped_records[label]
            if group_index < len(group):
                selected_records.append(group[group_index])
                added_record = True
                if len(selected_records) == limit:
                    break
        if not added_record:
            break
        group_index += 1

    return selected_records


def format_average_table(labels: list[str], scores: list[float], title: str) -> None:
    averages = average_scores_by_label(labels, scores)
    print(title)
    for label in ["No Fit", "Potential Fit", "Good Fit"]:
        if label in averages:
            print(f"  {label}: {averages[label]:.2f}")
    print()


def save_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "predicted_label",
        "semantic_score",
        "unweighted_skill_match_score",
        "skill_match_score",
        "overall_score",
        "matched_skills",
        "missing_skills",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "predicted_label": row["predicted_label"],
                    "semantic_score": f"{row['semantic_score']:.2f}",
                    "unweighted_skill_match_score": f"{row['unweighted_skill_match_score']:.2f}",
                    "skill_match_score": f"{row['skill_match_score']:.2f}",
                    "overall_score": f"{row['overall_score']:.2f}",
                    "matched_skills": "; ".join(row["matched_skills"]),
                    "missing_skills": "; ".join(row["missing_skills"]),
                }
            )


def main() -> None:
    args = parse_args()
    record_limit = None if args.limit == 0 else args.limit
    all_records = load_records(get_split_path(args.split))
    records = select_records(
        all_records,
        limit=record_limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    if not records:
        raise ValueError("No records loaded for evaluation.")

    print(f"Evaluating split: {args.split}")
    print(f"Records: {len(records)}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Encoder model: {args.model}")
    print("Label distribution:")
    for label, count in Counter(record["label"] for record in records).items():
        print(f"  {label}: {count}")
    print()

    model = load_model(args.model)
    skills = load_skills()

    labels = [record["label"] for record in records]

    print("Scoring records with encoder embeddings and skill extraction...")
    evaluation_rows = score_records(
        records,
        model=model,
        skill_dictionary=skills,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    semantic_scores = [float(row["semantic_score"]) for row in evaluation_rows]
    unweighted_skill_match_scores = [
        float(row["unweighted_skill_match_score"])
        for row in evaluation_rows
    ]
    skill_match_scores = [float(row["skill_match_score"]) for row in evaluation_rows]
    overall_scores = [float(row["overall_score"]) for row in evaluation_rows]

    label_scores = labels_to_scores(labels)
    predicted_labels = [row["predicted_label"] for row in evaluation_rows]

    format_average_table(labels, semantic_scores, "Average semantic score by label")
    format_average_table(labels, unweighted_skill_match_scores, "Average unweighted skill match score by label")
    format_average_table(labels, skill_match_scores, "Average skill match score by label")
    format_average_table(labels, overall_scores, "Average overall score by label")

    print("Correlation with label scores")
    print(f"  Pearson semantic: {pearson_correlation(semantic_scores, label_scores):.4f}")
    print(f"  Spearman semantic: {spearman_correlation(semantic_scores, label_scores):.4f}")
    print(f"  Pearson overall: {pearson_correlation(overall_scores, label_scores):.4f}")
    print(f"  Spearman overall: {spearman_correlation(overall_scores, label_scores):.4f}")
    print()

    print("Threshold-based label prediction")
    print(f"  Accuracy: {classification_accuracy(labels, predicted_labels):.4f}")
    print()

    if args.save_csv:
        output_path = Path(args.save_csv)
        save_results(output_path, evaluation_rows)
        print(f"Saved per-example results to: {output_path}")


if __name__ == "__main__":
    main()
