import pytest

from src.matcher import match_resume_to_job
from src.skill_evidence import apply_linguistic_safeguard, classify_resume_skill_evidence, split_sentences
from src.skill_extractor import load_skills


class StubEvidenceClassifier:
    def predict(self, sentence: str, skill: str) -> tuple[str, float]:
        sentence_lower = sentence.casefold()
        if "do not" in sentence_lower or "not have" in sentence_lower or "no " in sentence_lower:
            return "negated", 0.97
        if "learning" in sentence_lower or "interested" in sentence_lower:
            return "unclear", 0.88
        return "positive", 0.95


def test_split_sentences_handles_bullets_and_sentences() -> None:
    sentences = split_sentences("- Built APIs with Python.\n- I do not have AWS experience.")

    assert sentences == ["Built APIs with Python.", "I do not have AWS experience."]


def test_split_sentences_splits_contrastive_clauses() -> None:
    sentences = split_sentences("I have experience with Python but I do not have AWS experience.")

    assert sentences == ["I have experience with Python", "I do not have AWS experience."]


def test_split_sentences_splits_turkish_contrastive_clauses() -> None:
    sentences = split_sentences("Python deneyimim var ancak AWS deneyimim yok.")

    assert sentences == ["Python deneyimim var", "AWS deneyimim yok."]


def test_linguistic_safeguard_corrects_obvious_evidence_patterns() -> None:
    assert apply_linguistic_safeguard("I have experience with Python.", "unclear", 0.3) == (
        "positive",
        0.99,
    )
    assert apply_linguistic_safeguard("I do not have AWS experience.", "positive", 0.3) == (
        "negated",
        0.99,
    )
    assert apply_linguistic_safeguard("I am learning Kubernetes.", "positive", 0.3) == (
        "unclear",
        0.99,
    )
    assert apply_linguistic_safeguard("Python ile servisler geliştirdim.", "unclear", 0.3) == (
        "positive",
        0.99,
    )
    assert apply_linguistic_safeguard("AWS deneyimim yok.", "positive", 0.3) == (
        "negated",
        0.99,
    )
    assert apply_linguistic_safeguard("Kubernetes öğreniyorum.", "positive", 0.3) == (
        "unclear",
        0.99,
    )


def test_skill_evidence_classifies_positive_negated_and_unclear_mentions() -> None:
    result = classify_resume_skill_evidence(
        "I built APIs with Python. I do not have AWS experience. I am learning Kubernetes.",
        load_skills(),
        classifier=StubEvidenceClassifier(),
    )

    assert result["positive_resume_skills"] == ["Python"]
    assert result["negated_resume_skills"] == ["AWS"]
    assert result["unclear_resume_skills"] == ["Kubernetes"]
    assert {item["label"] for item in result["skill_evidence"]} == {
        "positive",
        "negated",
        "unclear",
    }


def test_skill_evidence_classifies_turkish_mentions() -> None:
    result = classify_resume_skill_evidence(
        "Python ile servisler geliştirdim. AWS deneyimim yok. Kubernetes öğreniyorum.",
        load_skills(),
        classifier=StubEvidenceClassifier(),
    )

    assert result["positive_resume_skills"] == ["Python"]
    assert result["negated_resume_skills"] == ["AWS"]
    assert result["unclear_resume_skills"] == ["Kubernetes"]
    assert {item["label"] for item in result["skill_evidence"]} == {
        "positive",
        "negated",
        "unclear",
    }


def test_matcher_does_not_count_negated_or_unclear_skills_as_matched(encoder_model) -> None:
    result = match_resume_to_job(
        resume_text="I built APIs with Python. I do not have AWS experience. I am learning Kubernetes.",
        job_description_text="The role requires Python, AWS, and Kubernetes.",
        model=encoder_model,
        skill_dictionary=load_skills(),
        evidence_classifier=StubEvidenceClassifier(),
    )

    assert result["matched_skills"] == ["Python"]
    assert result["missing_skills"] == ["AWS", "Kubernetes"]
    assert result["negated_resume_skills"] == ["AWS"]
    assert result["unclear_resume_skills"] == ["Kubernetes"]
    assert result["unweighted_skill_match_score"] == pytest.approx(33.33)
    assert result["skill_match_score"] == pytest.approx(33.33)
