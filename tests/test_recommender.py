from src.recommender import (
    CATEGORY_NOT_RECOMMENDED,
    CATEGORY_PARTIAL,
    CATEGORY_STRONG,
    CATEGORY_STRONG_SEMANTIC_GAPS,
    CATEGORY_WEAK,
    classify_match,
    generate_recommendation,
)


def test_classify_match() -> None:
    assert classify_match(85) == CATEGORY_STRONG
    assert classify_match(65) == CATEGORY_PARTIAL
    assert classify_match(45) == CATEGORY_WEAK
    assert classify_match(20) == CATEGORY_NOT_RECOMMENDED


def test_classify_match_reports_skill_gaps_for_high_semantic_match() -> None:
    assert (
        classify_match(
            overall_score=82,
            semantic_score=88,
            skill_match_score=60,
            missing_skills=["AWS", "Docker"],
        )
        == CATEGORY_STRONG_SEMANTIC_GAPS
    )


def test_generate_recommendation() -> None:
    recommendation = generate_recommendation(
        overall_score=35,
        semantic_score=30,
        skill_match_score=20,
        matched_skills=["Python"],
        missing_skills=["Docker"],
    )

    assert recommendation["match_category"] == CATEGORY_NOT_RECOMMENDED
    assert set(recommendation) == {"match_category"}
