import pytest

from src.evaluation import (
    average_ranks,
    average_scores_by_label,
    classification_accuracy,
    label_to_score,
    labels_to_scores,
    pearson_correlation,
    predict_label_from_score,
    score_records,
    spearman_correlation,
)
from src.skill_extractor import load_skills


def test_label_to_score() -> None:
    assert label_to_score("No Fit") == 0
    assert label_to_score("Potential Fit") == 1
    assert label_to_score("Good Fit") == 2


def test_labels_to_scores() -> None:
    assert labels_to_scores(["No Fit", "Good Fit", "Potential Fit"]) == [0, 2, 1]


def test_pearson_correlation_positive_relationship() -> None:
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_spearman_correlation_positive_ranking() -> None:
    assert spearman_correlation([10, 20, 30], [1, 2, 3]) == pytest.approx(1.0)


def test_average_ranks_handles_ties() -> None:
    assert average_ranks([10, 20, 20, 30]) == pytest.approx([1.0, 2.5, 2.5, 4.0])


def test_average_scores_by_label() -> None:
    averages = average_scores_by_label(
        labels=["No Fit", "No Fit", "Good Fit"],
        scores=[20, 40, 90],
    )

    assert averages["No Fit"] == pytest.approx(30.0)
    assert averages["Good Fit"] == pytest.approx(90.0)


def test_predict_label_from_score() -> None:
    assert predict_label_from_score(75) == "Good Fit"
    assert predict_label_from_score(55) == "Potential Fit"
    assert predict_label_from_score(20) == "No Fit"


def test_classification_accuracy() -> None:
    assert classification_accuracy(
        ["No Fit", "Good Fit", "Potential Fit"],
        ["No Fit", "No Fit", "Potential Fit"],
    ) == pytest.approx(2 / 3)


def test_score_records_uses_real_encoder_and_returns_expected_fields(encoder_model) -> None:
    records = [
        {
            "resume_text": "Python developer with SQL and Git experience.",
            "job_description_text": "Looking for Python, SQL, Git, Docker, and AWS experience.",
            "label": "Potential Fit",
        }
    ]

    rows = score_records(
        records,
        model=encoder_model,
        skill_dictionary=load_skills(),
    )

    assert len(rows) == 1
    assert rows[0]["label"] == "Potential Fit"
    assert rows[0]["matched_skills"] == ["Git", "Python", "SQL"]
    assert rows[0]["missing_skills"] == ["AWS", "Docker"]
    assert 0 <= rows[0]["unweighted_skill_match_score"] <= 100
    assert 0 <= rows[0]["skill_match_score"] <= 100
    assert 0 <= rows[0]["semantic_score"] <= 100
    assert 0 <= rows[0]["overall_score"] <= 100
