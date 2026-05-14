from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from src.preprocessing import normalize_for_matching


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILL_PATH = PROJECT_ROOT / "data" / "skills.json"


SkillDictionary = dict[str, list[str]]


def load_skills(path: str | Path = DEFAULT_SKILL_PATH) -> SkillDictionary:
    """Load and validate the skill dictionary."""

    skill_path = Path(path)
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill dictionary not found: {skill_path}")

    with skill_path.open("r", encoding="utf-8") as file:
        raw_skills = json.load(file)

    if not isinstance(raw_skills, dict):
        raise ValueError("Skill dictionary must be a JSON object.")

    skills: SkillDictionary = {}
    for canonical_name, aliases in raw_skills.items():
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ValueError("Every skill name must be a non-empty string.")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"Aliases for {canonical_name!r} must be a non-empty list.")
        if not all(isinstance(alias, str) and alias.strip() for alias in aliases):
            raise ValueError(f"All aliases for {canonical_name!r} must be non-empty strings.")

        unique_aliases = list(dict.fromkeys([canonical_name, *aliases]))
        skills[canonical_name] = unique_aliases

    return skills


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Build a boundary-aware regex pattern for one normalized alias."""

    normalized_alias = normalize_for_matching(alias)
    escaped_alias = re.escape(normalized_alias)
    return re.compile(rf"(?<![a-z0-9]){escaped_alias}(?![a-z0-9])")


def extract_skill_matches(text: str, skill_dictionary: SkillDictionary) -> dict[str, list[str]]:
    """Return canonical skill names and the aliases found in the text."""

    normalized_text = normalize_for_matching(text)
    matches: dict[str, list[str]] = {}

    for canonical_name, aliases in skill_dictionary.items():
        found_aliases: list[str] = []
        for alias in aliases:
            normalized_alias = normalize_for_matching(alias)
            if not normalized_alias:
                continue
            if _alias_pattern(alias).search(normalized_text):
                found_aliases.append(alias)

        if found_aliases:
            matches[canonical_name] = list(dict.fromkeys(found_aliases))

    return matches


def extract_skills(text: str, skill_dictionary: SkillDictionary) -> list[str]:
    """Extract normalized skill names from text."""

    return list(extract_skill_matches(text, skill_dictionary).keys())


def compare_skills(
    resume_skills: Iterable[str],
    job_skills: Iterable[str],
) -> dict[str, list[str]]:
    """Compare resume skills against job skills."""

    resume_skill_set = set(resume_skills)
    job_skill_set = set(job_skills)

    matched_skills = resume_skill_set & job_skill_set
    missing_skills = job_skill_set - resume_skill_set
    extra_resume_skills = resume_skill_set - job_skill_set

    return {
        "matched_skills": sorted(matched_skills, key=str.casefold),
        "missing_skills": sorted(missing_skills, key=str.casefold),
        "extra_resume_skills": sorted(extra_resume_skills, key=str.casefold),
    }


def extract_and_compare(
    resume_text: str,
    job_description_text: str,
    skill_dictionary: SkillDictionary,
) -> dict[str, list[str]]:
    """Extract skills from both texts and return matched/missing skill groups."""

    resume_skills = extract_skills(resume_text, skill_dictionary)
    job_skills = extract_skills(job_description_text, skill_dictionary)
    comparison = compare_skills(resume_skills, job_skills)

    return {
        "resume_skills": sorted(resume_skills, key=str.casefold),
        "job_skills": sorted(job_skills, key=str.casefold),
        **comparison,
    }
