import pytest

from src.matcher import calculate_final_score, calculate_skill_match_score, match_resume_to_job
from src.skill_extractor import load_skills


def test_calculate_skill_match_score() -> None:
    score = calculate_skill_match_score(
        matched_skills=["Python", "SQL"],
        job_skills=["Python", "SQL", "Docker", "AWS"],
    )

    assert score == pytest.approx(50.0)


def test_calculate_unweighted_skill_match_score() -> None:
    score = calculate_skill_match_score(
        matched_skills=["Communication", "Customer Service"],
        job_skills=["Communication", "Customer Service", "Python", "AWS"],
        weighted=False,
    )

    assert score == pytest.approx(50.0)


def test_weighted_skill_match_reduces_soft_skill_overweighting() -> None:
    weighted_score = calculate_skill_match_score(
        matched_skills=["Communication", "Customer Service"],
        job_skills=["Communication", "Customer Service", "Python", "AWS"],
    )

    unweighted_score = calculate_skill_match_score(
        matched_skills=["Communication", "Customer Service"],
        job_skills=["Communication", "Customer Service", "Python", "AWS"],
        weighted=False,
    )

    assert weighted_score < unweighted_score


def test_weighted_skill_match_caps_general_only_job_skills() -> None:
    weighted_score = calculate_skill_match_score(
        matched_skills=["Communication", "Customer Service", "Leadership"],
        job_skills=["Communication", "Customer Service", "Leadership"],
    )

    assert weighted_score == pytest.approx(35.0)


def test_calculate_skill_match_score_without_job_skills() -> None:
    assert calculate_skill_match_score([], []) == pytest.approx(0.0)


def test_calculate_final_score() -> None:
    assert calculate_final_score(semantic_score=80, skill_match_score=50) == pytest.approx(74.0)


def test_match_resume_to_job_uses_real_encoder_and_combines_scores(encoder_model) -> None:
    result = match_resume_to_job(
        resume_text="Software developer with Python, SQL, Git, and REST API experience.",
        job_description_text="Looking for Python, SQL, Git, Docker, AWS, and RESTful services experience.",
        model=encoder_model,
        skill_dictionary=load_skills(),
    )

    assert 0 <= result["semantic_similarity"] <= 1
    assert 0 <= result["semantic_score"] <= 100
    assert result["unweighted_skill_match_score"] == pytest.approx(66.67)
    assert result["skill_match_score"] == pytest.approx(66.67)
    assert 0 <= result["overall_score"] <= 100
    assert result["matched_skills"] == ["Git", "Python", "REST API", "SQL"]
    assert result["missing_skills"] == ["AWS", "Docker"]


def test_match_resume_to_job_related_pair_scores_above_unrelated_pair(encoder_model) -> None:
    resume = "Software developer with Python, SQL, Git, and REST API experience."
    result = match_resume_to_job(
        resume_text=resume,
        job_description_text="Looking for Python, SQL, Git, Docker, AWS, and RESTful services experience.",
        model=encoder_model,
        skill_dictionary=load_skills(),
    )
    unrelated_result = match_resume_to_job(
        resume_text=resume,
        job_description_text="Restaurant manager needed for hospitality, scheduling, and customer service.",
        model=encoder_model,
        skill_dictionary=load_skills(),
    )

    assert result["overall_score"] > unrelated_result["overall_score"]
