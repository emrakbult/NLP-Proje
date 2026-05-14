from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from src.preprocessing import clean_text


LABEL_TO_SIMILARITY = {
    "No Fit": 0.0,
    "Potential Fit": 0.5,
    "Good Fit": 1.0,
}

LABEL_ORDER = ("No Fit", "Potential Fit", "Good Fit")


def label_to_similarity(label: str) -> float:
    """Map a dataset fit label to a cosine similarity training target."""

    if label not in LABEL_TO_SIMILARITY:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_TO_SIMILARITY[label]


def select_records(
    records: Sequence[dict[str, str]],
    limit: int | None,
    sample_mode: str = "balanced",
    seed: int = 42,
) -> list[dict[str, str]]:
    """Select a deterministic subset of records for fine-tuning or evaluation."""

    records = list(records)
    if limit is None or limit <= 0 or limit >= len(records):
        return records

    if sample_mode == "first":
        return records[:limit]

    rng = random.Random(seed)

    if sample_mode == "random":
        sampled_records = records[:]
        rng.shuffle(sampled_records)
        return sampled_records[:limit]

    if sample_mode != "balanced":
        raise ValueError(f"Unknown sample mode: {sample_mode}")

    grouped_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped_records[record["label"]].append(record)

    for group in grouped_records.values():
        rng.shuffle(group)

    labels = [label for label in LABEL_ORDER if label in grouped_records]
    labels.extend(sorted(label for label in grouped_records if label not in LABEL_ORDER))

    if not labels:
        return []

    per_label = limit // len(labels)
    remainder = limit % len(labels)

    selected_records: list[dict[str, str]] = []
    remaining_records: list[dict[str, str]] = []

    for index, label in enumerate(labels):
        target_count = per_label + (1 if index < remainder else 0)
        group = grouped_records[label]
        selected_records.extend(group[:target_count])
        remaining_records.extend(group[target_count:])

    if len(selected_records) < limit:
        rng.shuffle(remaining_records)
        selected_records.extend(remaining_records[: limit - len(selected_records)])

    rng.shuffle(selected_records)
    return selected_records[:limit]


def records_to_input_examples(records: Sequence[dict[str, str]]) -> list[Any]:
    """Convert dataset records into Sentence Transformers InputExample objects."""

    try:
        from sentence_transformers import InputExample
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for fine-tuning. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    examples: list[Any] = []
    for record in records:
        examples.append(
            InputExample(
                texts=[
                    clean_text(record["resume_text"]),
                    clean_text(record["job_description_text"]),
                ],
                label=label_to_similarity(record["label"]),
            )
        )
    return examples


def records_to_similarity_columns(
    records: Sequence[dict[str, str]],
) -> tuple[list[str], list[str], list[float]]:
    """Return evaluator inputs from resume-job records."""

    resumes: list[str] = []
    jobs: list[str] = []
    scores: list[float] = []

    for record in records:
        resumes.append(clean_text(record["resume_text"]))
        jobs.append(clean_text(record["job_description_text"]))
        scores.append(label_to_similarity(record["label"]))

    return resumes, jobs, scores
