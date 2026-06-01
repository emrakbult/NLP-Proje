from src.recommender import (
    classify_match,
    generate_candidate_suggestions,
    generate_hr_evaluation,
    generate_interview_focus,
    generate_recommendation,
)


def test_classify_match() -> None:
    assert classify_match(85) == "Güçlü Eşleşme"
    assert classify_match(65) == "Kısmi Eşleşme"
    assert classify_match(45) == "Zayıf Eşleşme"
    assert classify_match(20) == "Önerilmez"


def test_classify_match_reports_skill_gaps_for_high_semantic_match() -> None:
    assert (
        classify_match(
            overall_score=82,
            semantic_score=88,
            skill_match_score=60,
            missing_skills=["AWS", "Docker"],
        )
        == "Semantik Eşleşme Güçlü, Beceri Açıkları Var"
    )


def test_generate_hr_evaluation_mentions_scores_and_skills() -> None:
    text = generate_hr_evaluation(
        overall_score=65,
        semantic_score=70,
        skill_match_score=45,
        matched_skills=["Python", "SQL"],
        missing_skills=["Docker", "AWS"],
    )

    assert "kısmi" in text.casefold()
    assert "70.00" in text
    assert "45.00" in text
    assert "Python, SQL" in text
    assert "Docker, AWS" in text


def test_generate_hr_evaluation_explains_high_semantic_skill_gap() -> None:
    text = generate_hr_evaluation(
        overall_score=82,
        semantic_score=88,
        skill_match_score=60,
        matched_skills=["Python", "SQL", "Git"],
        missing_skills=["AWS", "Docker"],
    )

    assert "semantik uyumu güçlü" in text.casefold()
    assert "eksik veya belirsiz" in text.casefold()


def test_generate_interview_focus_from_missing_skills() -> None:
    focus = generate_interview_focus(["Docker", "AWS"])

    assert focus == [
        "Docker konusunda pratik deneyimi doğrulayın.",
        "AWS konusunda pratik deneyimi doğrulayın.",
    ]


def test_generate_candidate_suggestions_without_missing_skills() -> None:
    suggestions = generate_candidate_suggestions([])

    assert suggestions == ["Projeleri, araçları ve ölçülebilir sonuçları anlatarak özgeçmişi somut tutun."]


def test_generate_recommendation() -> None:
    recommendation = generate_recommendation(
        overall_score=35,
        semantic_score=30,
        skill_match_score=20,
        matched_skills=["Python"],
        missing_skills=["Docker"],
    )

    assert recommendation["match_category"] == "Önerilmez"
    assert "hr_evaluation" in recommendation
    assert recommendation["interview_focus"] == ["Docker konusunda pratik deneyimi doğrulayın."]
    assert recommendation["candidate_suggestions"]
