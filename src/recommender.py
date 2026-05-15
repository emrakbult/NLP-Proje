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


def _score_band(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 60:
        return "moderate"
    if score >= 40:
        return "limited"
    return "low"


def _gap_severity(skill_match_score: float, missing_skills: list[str]) -> str:
    if not missing_skills:
        return "no explicit skill gap"
    if skill_match_score >= 75:
        return "minor skill gap"
    if skill_match_score >= 50:
        return "moderate skill gap"
    return "significant skill gap"


def _recommended_hr_action(category: str, missing_skills: list[str]) -> str:
    if category == "Strong Match":
        return (
            "Recommended HR action: move the candidate forward, while still validating "
            "the depth of the listed experience during the interview."
        )
    if category in {"Strong Semantic Match with Skill Gaps", "Good Match with Skill Gaps"}:
        return (
            "Recommended HR action: keep the candidate in consideration, but use the interview "
            "to verify whether the missing skills are true gaps or simply not documented in the CV."
        )
    if category == "Partial Match":
        return (
            "Recommended HR action: consider the candidate for screening only if the role can tolerate "
            "some ramp-up time or if the missing requirements are not mandatory."
        )
    if category == "Weak Match":
        return (
            "Recommended HR action: do not prioritize the candidate unless the applicant pool is limited "
            "or the role requirements can be adjusted."
        )

    if missing_skills:
        return (
            "Recommended HR action: reject or hold for another role unless there is external evidence "
            "that addresses the missing requirements."
        )
    return "Recommended HR action: review manually before making a screening decision."


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
    semantic_band = _score_band(semantic_score)
    skill_band = _score_band(skill_match_score)
    gap_severity = _gap_severity(skill_match_score, missing_skills)

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
        f"{summary} Overall suitability is {overall_score:.2f}, which places the candidate in the "
        f"'{category}' category. The semantic similarity score is {semantic_score:.2f}, indicating a "
        f"{semantic_band} level of textual and role-context alignment between the CV and the job description. "
        f"The weighted skill match score is {skill_match_score:.2f}, indicating {skill_band} explicit skill "
        f"coverage and a {gap_severity} from the extracted job requirements. Matched skills include: "
        f"{matched_text}."
    )

    if missing_skills:
        explanation += (
            f" Missing or unclear job requirements include: {missing_text}. "
            "These gaps should be treated as interview validation points rather than automatic rejection "
            "criteria, because the CV may omit some practical experience."
        )
    else:
        explanation += (
            " No explicit missing skills were found in the skill dictionary comparison, so the interview "
            "should focus on depth of experience, project ownership, and evidence of real-world usage."
        )

    return f"{explanation} {_recommended_hr_action(category, missing_skills)}"


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
