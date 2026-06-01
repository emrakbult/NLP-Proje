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
        return "Semantik Eşleşme Güçlü, Beceri Açıkları Var"
    if overall_score >= 70 and semantic_score >= 75 and skill_match_score < 70 and has_key_skill_gaps:
        return "İyi Eşleşme, Beceri Açıkları Var"
    if overall_score >= 80:
        return "Güçlü Eşleşme"
    if overall_score >= 60:
        return "Kısmi Eşleşme"
    if overall_score >= 40:
        return "Zayıf Eşleşme"
    return "Önerilmez"


def _format_skills(skills: list[str], limit: int = 8) -> str:
    if not skills:
        return "yok"

    visible_skills = skills[:limit]
    formatted = ", ".join(visible_skills)
    remaining_count = len(skills) - len(visible_skills)
    if remaining_count > 0:
        formatted += f" ve {remaining_count} tane daha"
    return formatted


def _score_band(score: float) -> str:
    if score >= 80:
        return "yüksek"
    if score >= 60:
        return "orta"
    if score >= 40:
        return "sınırlı"
    return "düşük"


def _gap_severity(skill_match_score: float, missing_skills: list[str]) -> str:
    if not missing_skills:
        return "belirgin beceri açığı yok"
    if skill_match_score >= 75:
        return "küçük beceri açığı"
    if skill_match_score >= 50:
        return "orta düzey beceri açığı"
    return "önemli beceri açığı"


def _recommended_hr_action(category: str, missing_skills: list[str]) -> str:
    if category == "Güçlü Eşleşme":
        return (
            "Önerilen İK aksiyonu: adayı bir sonraki aşamaya taşıyın; yine de listelenen "
            "deneyimin derinliğini mülakatta doğrulayın."
        )
    if category in {"Semantik Eşleşme Güçlü, Beceri Açıkları Var", "İyi Eşleşme, Beceri Açıkları Var"}:
        return (
            "Önerilen İK aksiyonu: adayı değerlendirmede tutun; ancak eksik becerilerin gerçek "
            "açık mı yoksa sadece CV'de belgelenmemiş deneyim mi olduğunu mülakatta doğrulayın."
        )
    if category == "Kısmi Eşleşme":
        return (
            "Önerilen İK aksiyonu: rol belirli bir adaptasyon süresini kaldırabiliyorsa veya "
            "eksik gereksinimler zorunlu değilse adayı ön elemede değerlendirin."
        )
    if category == "Zayıf Eşleşme":
        return (
            "Önerilen İK aksiyonu: aday havuzu sınırlı değilse veya rol gereksinimleri "
            "esnetilemiyorsa adayı önceliklendirmeyin."
        )

    if missing_skills:
        return (
            "Önerilen İK aksiyonu: eksik gereksinimleri karşılayan ek bir kanıt yoksa adayı "
            "reddedin veya başka bir rol için bekletin."
        )
    return "Önerilen İK aksiyonu: ön eleme kararı vermeden önce sonucu manuel olarak inceleyin."


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

    if category == "Semantik Eşleşme Güçlü, Beceri Açıkları Var":
        summary = (
            "Adayın bu rolle semantik uyumu güçlü, "
            "ancak bazı önemli zorunlu beceriler eksik veya belirsiz."
        )
    elif category == "İyi Eşleşme, Beceri Açıkları Var":
        summary = (
            "Aday genel olarak iyi bir eşleşme gibi görünüyor, "
            "ancak özgeçmişte bu rol için dikkate değer beceri açıkları var."
        )
    elif category == "Güçlü Eşleşme":
        summary = "Aday bu rol için güçlü bir eşleşme gibi görünüyor."
    elif category == "Kısmi Eşleşme":
        summary = "Aday bu rol için kısmi bir eşleşme gibi görünüyor."
    elif category == "Zayıf Eşleşme":
        summary = "Aday bu rolle sınırlı düzeyde uyum gösteriyor."
    else:
        summary = "Mevcut özgeçmiş-iş ilanı karşılaştırmasına göre aday bu rol için önerilmez."

    explanation = (
        f"{summary} Genel uygunluk skoru {overall_score:.2f}; bu skor adayı "
        f"'{category}' kategorisine yerleştirir. Semantik benzerlik skoru {semantic_score:.2f}; bu değer "
        f"CV ile iş ilanı arasında {semantic_band} düzeyde metinsel ve rol bağlamı uyumu olduğunu gösterir. "
        f"Ağırlıklı beceri eşleşme skoru {skill_match_score:.2f}; bu skor {skill_band} düzeyde açık beceri "
        f"kapsamı ve çıkarılan iş gereksinimlerine göre {gap_severity} olduğunu gösterir. Eşleşen beceriler: "
        f"{matched_text}."
    )

    if missing_skills:
        explanation += (
            f" Eksik veya belirsiz iş gereksinimleri: {missing_text}. "
            "Bu açıklar otomatik ret kriteri yerine mülakatta doğrulanacak noktalar olarak ele alınmalıdır; "
            "çünkü CV bazı pratik deneyimleri içermiyor olabilir."
        )
    else:
        explanation += (
            " Beceri sözlüğü karşılaştırmasında açık bir eksik beceri bulunmadı; bu nedenle mülakat "
            "deneyim derinliğine, proje sahipliğine ve gerçek kullanım kanıtlarına odaklanmalıdır."
        )

    return f"{explanation} {_recommended_hr_action(category, missing_skills)}"


def generate_interview_focus(missing_skills: list[str], limit: int = 6) -> list[str]:
    """Generate interview topics from missing skills."""

    return [
        f"{skill} konusunda pratik deneyimi doğrulayın."
        for skill in missing_skills[:limit]
    ]


def generate_candidate_suggestions(missing_skills: list[str], limit: int = 6) -> list[str]:
    """Generate candidate-facing improvement suggestions."""

    if not missing_skills:
        return ["Projeleri, araçları ve ölçülebilir sonuçları anlatarak özgeçmişi somut tutun."]

    return [
        f"Aday bu deneyime sahipse {skill} için net kanıt ekleyin; değilse benzer roller için bu beceriyi geliştirin."
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
