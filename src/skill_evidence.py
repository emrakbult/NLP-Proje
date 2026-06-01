from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, TypedDict

from src.skill_extractor import SkillDictionary, extract_skill_matches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILL_EVIDENCE_MODEL_PATH = PROJECT_ROOT / "models" / "skill-evidence-minilm-classifier"
SKILL_EVIDENCE_LABELS = ("positive", "negated", "unclear")
CLAUSE_SPLIT_PATTERN = re.compile(
    r"\s+(?:but|however|although|though|ama|ancak|fakat|buna rağmen|buna ragmen)\s+",
    re.IGNORECASE,
)
UNCLEAR_PATTERNS = (
    "learning",
    "interested",
    "basic awareness",
    "basic familiarity",
    "read about",
    "read documentation",
    "tutorial",
    "course",
    "theoretical",
    "want to",
    "plan to",
    "curious",
    "observed",
    "seen",
    "exposure",
    "high level",
    "öğreniyorum",
    "ogreniyorum",
    "öğrenmekteyim",
    "ogrenmekteyim",
    "ilgileniyorum",
    "temel bilgi",
    "temel düzey",
    "temel duzey",
    "aşinalık",
    "asinalik",
    "dokümanlarını okudum",
    "dokumanlarini okudum",
    "eğitim",
    "egitim",
    "teorik",
    "istiyorum",
    "planlıyorum",
    "planliyorum",
    "merak ediyorum",
    "gözlemledim",
    "gozlemledim",
)
NEGATION_PATTERNS = (
    "do not have",
    "don't have",
    "not have",
    "no hands-on",
    "no professional",
    "no experience",
    "not experienced",
    "have not used",
    "not used",
    "never",
    "did not",
    "not worked",
    "not built",
    "not deployed",
    "not configured",
    "without",
    "deneyimim yok",
    "deneyim yok",
    "deneyimim bulunmuyor",
    "deneyim bulunmuyor",
    "tecrübem yok",
    "tecrubem yok",
    "pratik deneyimim yok",
    "profesyonel deneyimim yok",
    "deneyimim eksik",
    "deneyim eksik",
    "kullanmadım",
    "kullanmadim",
    "kullanmadı",
    "kullanmadi",
    "kullanmıyorum",
    "kullanmiyorum",
    "çalışmadım",
    "calismadim",
    "geliştirmedim",
    "gelistirmedim",
    "kurmadım",
    "kurmadim",
    "dağıtmadım",
    "dagitmadim",
    "yapılandırmadım",
    "yapilandirmadim",
    "bilgim yok",
)
POSITIVE_PATTERNS = (
    "built",
    "developed",
    "used",
    "use ",
    "maintained",
    "created",
    "implemented",
    "wrote",
    "managed",
    "configured",
    "deployed",
    "worked with",
    "experience with",
    "hands-on",
    "geliştirdim",
    "gelistirdim",
    "geliştirdi",
    "gelistirdi",
    "kullandım",
    "kullandim",
    "kullandı",
    "kullandi",
    "kullanarak",
    "bakımını yaptım",
    "bakimini yaptim",
    "oluşturdum",
    "olusturdum",
    "oluşturdu",
    "olusturdu",
    "tasarladım",
    "tasarladim",
    "tasarladı",
    "tasarladi",
    "uyguladım",
    "uyguladim",
    "yazdım",
    "yazdim",
    "yönettim",
    "yonettim",
    "yapılandırdım",
    "yapilandirdim",
    "dağıttım",
    "dagittim",
    "destekledim",
    "sürdürdüm",
    "surdurdum",
    "çalıştım",
    "calistim",
    "deneyim",
    "tecrübe",
    "tecrube",
)


class SkillEvidenceItem(TypedDict):
    skill: str
    label: str
    confidence: float
    sentence: str


class SkillEvidenceResult(TypedDict):
    positive_resume_skills: list[str]
    negated_resume_skills: list[str]
    unclear_resume_skills: list[str]
    skill_evidence: list[SkillEvidenceItem]
    skill_evidence_model_available: bool


class EvidenceClassifier(Protocol):
    def predict(self, sentence: str, skill: str) -> tuple[str, float]:
        """Predict evidence label and confidence for a sentence-skill pair."""


def build_classifier_input(sentence: str, skill: str) -> str:
    return f"Skill: {skill}\nSentence: {sentence}"


def split_sentences(text: str) -> list[str]:
    """Split resume text into sentence-like units while preserving bullet content."""

    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences: list[str] = []
    for part in raw_parts:
        cleaned_part = re.sub(r"^\s*[-*•]\s*", "", part.strip())
        for clause in CLAUSE_SPLIT_PATTERN.split(cleaned_part):
            sentence = clause.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def apply_linguistic_safeguard(
    sentence: str,
    label: str,
    confidence: float,
) -> tuple[str, float]:
    """Correct obvious evidence cases that small classifiers can underfit."""

    normalized_sentence = sentence.casefold()
    if any(pattern in normalized_sentence for pattern in UNCLEAR_PATTERNS):
        return "unclear", max(confidence, 0.99)
    if any(pattern in normalized_sentence for pattern in NEGATION_PATTERNS):
        return "negated", max(confidence, 0.99)
    if any(pattern in normalized_sentence for pattern in POSITIVE_PATTERNS):
        return "positive", max(confidence, 0.99)
    return label, confidence


def is_skill_evidence_model_available(
    model_path: str | Path = DEFAULT_SKILL_EVIDENCE_MODEL_PATH,
) -> bool:
    path = Path(model_path)
    return (path / "config.json").exists() and (path / "label_mapping.json").exists()


class TransformerSkillEvidenceClassifier:
    """Encoder-only sequence classifier for sentence-level skill evidence."""

    def __init__(self, model_path: str | Path = DEFAULT_SKILL_EVIDENCE_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not is_skill_evidence_model_available(self.model_path):
            raise FileNotFoundError(f"Skill evidence classifier not found: {self.model_path}")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Beceri kanıtı sınıflandırması için transformers ve torch gereklidir. "
                "Bağımlılıkları şu komutla kurun: pip install -r requirements.txt"
            ) from error

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        label_mapping_path = self.model_path / "label_mapping.json"
        mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
        self.id_to_label = {int(key): value for key, value in mapping["id_to_label"].items()}

    def predict(self, sentence: str, skill: str) -> tuple[str, float]:
        encoded = self.tokenizer(
            build_classifier_input(sentence, skill),
            truncation=True,
            max_length=160,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probabilities = self.torch.softmax(logits, dim=0)
            confidence, label_id = self.torch.max(probabilities, dim=0)

        label = self.id_to_label[int(label_id.item())]
        return label, float(confidence.item())


@lru_cache(maxsize=1)
def load_skill_evidence_classifier() -> TransformerSkillEvidenceClassifier | None:
    if not is_skill_evidence_model_available():
        return None
    return TransformerSkillEvidenceClassifier()


def classify_resume_skill_evidence(
    resume_text: str,
    skill_dictionary: SkillDictionary,
    classifier: EvidenceClassifier | None = None,
) -> SkillEvidenceResult:
    """Validate resume skill mentions as positive, negated, or unclear evidence."""

    if classifier is None:
        classifier = load_skill_evidence_classifier()

    model_available = classifier is not None
    evidence_items: list[SkillEvidenceItem] = []
    positive_skills: set[str] = set()
    negated_skills: set[str] = set()
    unclear_skills: set[str] = set()

    for sentence in split_sentences(resume_text):
        sentence_matches = extract_skill_matches(sentence, skill_dictionary)
        for skill in sentence_matches:
            if classifier is None:
                label, confidence = "positive", 1.0
            else:
                label, confidence = classifier.predict(sentence, skill)
                if label not in SKILL_EVIDENCE_LABELS:
                    label, confidence = "unclear", 0.0
                label, confidence = apply_linguistic_safeguard(sentence, label, confidence)

            evidence_items.append(
                {
                    "skill": skill,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "sentence": sentence,
                }
            )

            if label == "positive":
                positive_skills.add(skill)
            elif label == "negated":
                negated_skills.add(skill)
            else:
                unclear_skills.add(skill)

    negated_only = negated_skills - positive_skills
    unclear_only = unclear_skills - positive_skills - negated_only

    return {
        "positive_resume_skills": sorted(positive_skills, key=str.casefold),
        "negated_resume_skills": sorted(negated_only, key=str.casefold),
        "unclear_resume_skills": sorted(unclear_only, key=str.casefold),
        "skill_evidence": evidence_items,
        "skill_evidence_model_available": model_available,
    }
