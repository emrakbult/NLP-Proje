from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from src.matcher import calculate_final_score, calculate_skill_match_score
from src.similarity import encode_texts, pairwise_cosine_similarities, similarity_to_percentage
from src.skill_extractor import SkillDictionary, extract_and_compare


LABEL_TO_SCORE = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}

SCORE_TO_LABEL = {
    0: "No Fit",
    1: "Potential Fit",
    2: "Good Fit",
}


def label_to_score(label: str) -> int:
    """Convert a dataset label into an ordinal numeric score."""

    if label not in LABEL_TO_SCORE:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_TO_SCORE[label]


def labels_to_scores(labels: Iterable[str]) -> list[int]:
    """Convert labels into ordinal numeric scores."""

    return [label_to_score(label) for label in labels]


def pearson_correlation(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Calculate Pearson correlation without external dependencies."""

    if len(values_a) != len(values_b):
        raise ValueError("Input sequences must have the same length.")
    if not values_a:
        raise ValueError("Input sequences must not be empty.")

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)

    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]

    numerator = sum(left * right for left, right in zip(centered_a, centered_b))
    denominator_a = math.sqrt(sum(value * value for value in centered_a))
    denominator_b = math.sqrt(sum(value * value for value in centered_b))

    if denominator_a == 0 or denominator_b == 0:
        return 0.0

    return numerator / (denominator_a * denominator_b)


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks for values, using 1-based ranks and tie averaging."""

    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0

    while index < len(indexed_values):
        tie_end = index
        while tie_end + 1 < len(indexed_values) and indexed_values[tie_end + 1][1] == indexed_values[index][1]:
            tie_end += 1

        average_rank = ((index + 1) + (tie_end + 1)) / 2
        for tied_index in range(index, tie_end + 1):
            original_index = indexed_values[tied_index][0]
            ranks[original_index] = average_rank

        index = tie_end + 1

    return ranks


def spearman_correlation(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Calculate Spearman rank correlation."""

    if len(values_a) != len(values_b):
        raise ValueError("Input sequences must have the same length.")
    if not values_a:
        raise ValueError("Input sequences must not be empty.")

    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    return pearson_correlation(ranks_a, ranks_b)


def average_scores_by_label(
    labels: Sequence[str],
    scores: Sequence[float],
) -> dict[str, float]:
    """Calculate average score for each dataset label."""

    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have the same length.")

    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for label, score in zip(labels, scores):
        grouped_scores[label].append(score)

    return {
        label: sum(group_scores) / len(group_scores)
        for label, group_scores in grouped_scores.items()
    }


def predict_label_from_score(score: float) -> str:
    """Map a 0-100 suitability score to a coarse fit label."""

    if score >= 70:
        return "Good Fit"
    if score >= 40:
        return "Potential Fit"
    return "No Fit"


def classification_accuracy(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    """Calculate simple label accuracy."""

    if len(true_labels) != len(predicted_labels):
        raise ValueError("Label sequences must have the same length.")
    if not true_labels:
        raise ValueError("Label sequences must not be empty.")

    correct_count = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(true_labels, predicted_labels)
    )
    return correct_count / len(true_labels)


def score_records(
    records: Sequence[dict[str, str]],
    model,
    skill_dictionary: SkillDictionary,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> list[dict[str, object]]:
    """Score resume-job records with encoder similarity and skill extraction."""

    if not records:
        return []

    resume_texts = [record["resume_text"] for record in records]
    job_texts = [record["job_description_text"] for record in records]

    resume_embeddings = encode_texts(
        resume_texts,
        model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    job_embeddings = encode_texts(
        job_texts,
        model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    similarities = pairwise_cosine_similarities(resume_embeddings, job_embeddings)

    rows: list[dict[str, object]] = []
    for record, similarity in zip(records, similarities):
        semantic_score = similarity_to_percentage(similarity)
        skill_result = extract_and_compare(
            record["resume_text"],
            record["job_description_text"],
            skill_dictionary,
        )
        unweighted_skill_match_score = calculate_skill_match_score(
            skill_result["matched_skills"],
            skill_result["job_skills"],
            weighted=False,
        )
        skill_match_score = calculate_skill_match_score(
            skill_result["matched_skills"],
            skill_result["job_skills"],
        )
        overall_score = calculate_final_score(semantic_score, skill_match_score)

        rows.append(
            {
                "label": record["label"],
                "predicted_label": predict_label_from_score(overall_score),
                "semantic_similarity": similarity,
                "semantic_score": semantic_score,
                "unweighted_skill_match_score": unweighted_skill_match_score,
                "skill_match_score": skill_match_score,
                "overall_score": overall_score,
                "resume_skills": skill_result["resume_skills"],
                "job_skills": skill_result["job_skills"],
                "matched_skills": skill_result["matched_skills"],
                "missing_skills": skill_result["missing_skills"],
                "resume_text": record["resume_text"],
                "job_description_text": record["job_description_text"],
            }
        )

    return rows
