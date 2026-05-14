import pytest

from src.similarity import (
    calculate_cosine_similarity,
    calculate_semantic_similarity_score,
    cosine_similarity,
    similarity_to_percentage,
)


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_similarity_to_percentage_clamps_values() -> None:
    assert similarity_to_percentage(0.75) == pytest.approx(75.0)
    assert similarity_to_percentage(-0.2) == pytest.approx(0.0)
    assert similarity_to_percentage(1.2) == pytest.approx(100.0)


def test_calculate_cosine_similarity_with_real_encoder(encoder_model) -> None:
    text = "Python developer with SQL and Git experience."

    similarity = calculate_cosine_similarity(text, text, encoder_model)

    assert similarity > 0.99


def test_real_encoder_scores_related_text_higher_than_unrelated_text(encoder_model) -> None:
    resume = "Software developer with Python, SQL, Git, REST API, and backend development experience."
    related_job = "We need a backend software engineer with Python, SQL, Git, and REST API experience."
    unrelated_job = "We are hiring a restaurant manager with hospitality and customer service experience."

    related_score = calculate_semantic_similarity_score(resume, related_job, encoder_model)
    unrelated_score = calculate_semantic_similarity_score(resume, unrelated_job, encoder_model)

    assert related_score > unrelated_score
    assert 0 <= unrelated_score <= 100
    assert 0 <= related_score <= 100
