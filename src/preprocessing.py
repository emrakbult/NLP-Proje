from __future__ import annotations

import re
import unicodedata


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Clean raw resume or job description text without damaging technical terms."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    return normalize_whitespace(normalized)


def normalize_for_matching(text: str) -> str:
    """Prepare text for dictionary and alias matching."""

    return clean_text(text).casefold()
