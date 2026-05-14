from __future__ import annotations

from src.skill_weights import has_role_specific_skill


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
        return "Strong Semantic Match with Skill Gaps"
    if overall_score >= 70 and semantic_score >= 75 and skill_match_score < 70 and has_key_skill_gaps:
        return "Good Match with Skill Gaps"
    if overall_score >= 80:
        return "Strong Match"
    if overall_score >= 60:
        return "Partial Match"
    if overall_score >= 40:
        return "Weak Match"
    return "Not Recommended"


def _format_skills(skills: list[str], limit: int = 8) -> str:
    if not skills:
        return "none"

    visible_skills = skills[:limit]
    formatted = ", ".join(visible_skills)
    remaining_count = len(skills) - len(visible_skills)
    if remaining_count > 0:
        formatted += f", and {remaining_count} more"
    return formatted


def generate_hr_evaluation(
    overall_score: float,
    semantic_score: float,
    skill_match_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """Generate a deterministic HR-oriented explanation."""

    category = classify_match(
        overall_score=overall_score,
        semantic_score=semantic_score,
        skill_match_score=skill_match_score,
        missing_skills=missing_skills,
    )
    matched_text = _format_skills(matched_skills)
    missing_text = _format_skills(missing_skills)

    if category == "Strong Semantic Match with Skill Gaps":
        summary = (
            "The candidate has strong semantic alignment with this role, "
            "but important required skills are missing or unclear."
        )
    elif category == "Good Match with Skill Gaps":
        summary = (
            "The candidate appears to be a good match overall, "
            "but the resume still has notable skill gaps for this role."
        )
    elif category == "Strong Match":
        summary = "The candidate appears to be a strong match for this role."
    elif category == "Partial Match":
        summary = "The candidate appears to be a partial match for this role."
    elif category == "Weak Match":
        summary = "The candidate shows some limited alignment with this role."
    else:
        summary = "The candidate is not recommended for this role based on the current resume-job comparison."

    explanation = (
        f"{summary} The semantic similarity score is {semantic_score:.2f}, "
        f"and the weighted skill match score is {skill_match_score:.2f}. "
        f"Matched skills include: {matched_text}."
    )

    if missing_skills:
        explanation += f" Missing or unclear job requirements include: {missing_text}."
    else:
        explanation += " No explicit missing skills were found in the skill dictionary comparison."

    return explanation


def generate_interview_focus(missing_skills: list[str], limit: int = 6) -> list[str]:
    """Generate interview topics from missing skills."""

    return [
        f"Verify practical experience with {skill}."
        for skill in missing_skills[:limit]
    ]


def generate_candidate_suggestions(missing_skills: list[str], limit: int = 6) -> list[str]:
    """Generate candidate-facing improvement suggestions."""

    if not missing_skills:
        return ["Keep the resume specific by describing projects, tools, and measurable outcomes."]

    return [
        f"Add clear evidence of {skill} if the candidate has this experience, or develop this skill for similar roles."
        for skill in missing_skills[:limit]
    ]


def generate_recommendation(
    overall_score: float,
    semantic_score: float,
    skill_match_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
) -> dict[str, object]:
    """Generate the complete explainability block for a match result."""

    return {
        "match_category": classify_match(
            overall_score=overall_score,
            semantic_score=semantic_score,
            skill_match_score=skill_match_score,
            missing_skills=missing_skills,
        ),
        "hr_evaluation": generate_hr_evaluation(
            overall_score=overall_score,
            semantic_score=semantic_score,
            skill_match_score=skill_match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        ),
        "interview_focus": generate_interview_focus(missing_skills),
        "candidate_suggestions": generate_candidate_suggestions(missing_skills),
    }
