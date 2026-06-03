from __future__ import annotations

from src.skill_weights import has_role_specific_skill


CATEGORY_STRONG_SEMANTIC_GAPS = "Beceri Eksikleri Olan Güçlü Anlamsal Eşleşme"
CATEGORY_GOOD_GAPS = "Beceri Eksikleri Olan İyi Eşleşme"
CATEGORY_STRONG = "Güçlü Eşleşme"
CATEGORY_PARTIAL = "Kısmi Eşleşme"
CATEGORY_WEAK = "Zayıf Eşleşme"
CATEGORY_NOT_RECOMMENDED = "Önerilmiyor"


def classify_match(
    overall_score: float,
    semantic_score: float | None = None,
    skill_match_score: float | None = None,
    missing_skills: list[str] | None = None,
) -> str:
    """Convert an overall score into an HR-friendly match category."""

    semantic_score = overall_score if semantic_score is None else semantic_score
    skill_match_score = overall_score if skill_match_score is None else skill_match_score
    missing_skills = missing_skills or []
    has_key_skill_gaps = bool(missing_skills) and has_role_specific_skill(missing_skills)

    if overall_score >= 80 and semantic_score >= 80 and skill_match_score < 75 and has_key_skill_gaps:
        return CATEGORY_STRONG_SEMANTIC_GAPS
    if overall_score >= 70 and semantic_score >= 75 and skill_match_score < 70 and has_key_skill_gaps:
        return CATEGORY_GOOD_GAPS
    if overall_score >= 80:
        return CATEGORY_STRONG
    if overall_score >= 60:
        return CATEGORY_PARTIAL
    if overall_score >= 40:
        return CATEGORY_WEAK
    return CATEGORY_NOT_RECOMMENDED


def generate_recommendation(
    overall_score: float,
    semantic_score: float,
    skill_match_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
) -> dict[str, object]:
    """Generate deterministic match metadata from model and skill scores."""

    del matched_skills

    return {
        "match_category": classify_match(
            overall_score=overall_score,
            semantic_score=semantic_score,
            skill_match_score=skill_match_score,
            missing_skills=missing_skills,
        ),
    }
