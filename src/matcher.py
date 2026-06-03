from __future__ import annotations

from typing import Any

from src.recommender import generate_recommendation
from src.similarity import calculate_cosine_similarity, load_model, similarity_to_percentage
from src.skill_evidence import EvidenceClassifier, classify_resume_skill_evidence
from src.skill_extractor import SkillDictionary, compare_skills, extract_skills, load_skills
from src.skill_weights import GENERAL_ONLY_SKILL_SCORE_CAP, has_role_specific_skill, total_skill_weight


def calculate_skill_match_score(
    matched_skills: list[str],
    job_skills: list[str],
    weighted: bool = True,
) -> float:
    """Calculate the percentage of job skills found in the resume."""

    if not job_skills:
        return 0.0

    if weighted:
        total_weight = total_skill_weight(job_skills)
        if total_weight == 0:
            return 0.0
        weighted_score = (total_skill_weight(matched_skills) / total_weight) * 100
        if not has_role_specific_skill(job_skills):
            return min(weighted_score, GENERAL_ONLY_SKILL_SCORE_CAP)
        return weighted_score

    return (len(matched_skills) / len(job_skills)) * 100


def calculate_final_score(
    semantic_score: float,
    skill_match_score: float,
    semantic_weight: float = 0.80,
    skill_weight: float = 0.20,
) -> float:
    """Combine semantic similarity and skill match scores."""

    total_weight = semantic_weight + skill_weight
    if total_weight <= 0:
        raise ValueError("Score weights must have a positive total.")

    weighted_score = (semantic_weight * semantic_score) + (skill_weight * skill_match_score)
    return weighted_score / total_weight


def match_resume_to_job(
    resume_text: str,
    job_description_text: str,
    model: Any | None = None,
    skill_dictionary: SkillDictionary | None = None,
    evidence_classifier: EvidenceClassifier | None = None,
) -> dict[str, Any]:
    """Run the first complete matching pipeline for one resume-job pair."""

    model = model or load_model()
    skill_dictionary = skill_dictionary or load_skills()

    neural_prediction: dict[str, Any] = {}
    if hasattr(model, "predict_pair"):
        neural_prediction = model.predict_pair(resume_text, job_description_text)
        raw_similarity = float(neural_prediction["cosine_similarity"])
        semantic_score = float(neural_prediction["semantic_score"])
    else:
        raw_similarity = calculate_cosine_similarity(resume_text, job_description_text, model)
        semantic_score = similarity_to_percentage(raw_similarity)

    evidence_result = classify_resume_skill_evidence(
        resume_text,
        skill_dictionary,
        classifier=evidence_classifier,
    )
    resume_skills = evidence_result["positive_resume_skills"]
    job_skills = extract_skills(job_description_text, skill_dictionary)
    comparison = compare_skills(resume_skills, job_skills)
    skill_result = {
        "resume_skills": sorted(resume_skills, key=str.casefold),
        "job_skills": sorted(job_skills, key=str.casefold),
        **comparison,
        "negated_resume_skills": evidence_result["negated_resume_skills"],
        "unclear_resume_skills": evidence_result["unclear_resume_skills"],
        "skill_evidence": evidence_result["skill_evidence"],
        "skill_evidence_model_available": evidence_result["skill_evidence_model_available"],
    }
    unweighted_skill_match_score = calculate_skill_match_score(
        skill_result["matched_skills"],
        skill_result["job_skills"],
        weighted=False,
    )
    skill_match_score = calculate_skill_match_score(
        skill_result["matched_skills"],
        skill_result["job_skills"],
    )
    final_score = calculate_final_score(semantic_score, skill_match_score)
    recommendation = generate_recommendation(
        overall_score=final_score,
        semantic_score=semantic_score,
        skill_match_score=skill_match_score,
        matched_skills=skill_result["matched_skills"],
        missing_skills=skill_result["missing_skills"],
    )

    return {
        "semantic_similarity": round(raw_similarity, 4),
        "semantic_score": round(semantic_score, 2),
        "unweighted_skill_match_score": round(unweighted_skill_match_score, 2),
        "skill_match_score": round(skill_match_score, 2),
        "overall_score": round(final_score, 2),
        "predicted_fit_label": neural_prediction.get("predicted_fit_label"),
        "fit_class_probabilities": neural_prediction.get("fit_class_probabilities", {}),
        "model_temperature": neural_prediction.get("temperature"),
        **skill_result,
        **recommendation,
    }
