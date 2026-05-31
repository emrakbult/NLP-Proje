from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ExtractedDocument:
    file_name: str
    extracted_text: str
    character_count: int
    converter: str


def validate_upload(file_name: str, content: bytes) -> str:
    """Validate an uploaded document and return its lowercase extension."""

    extension = Path(file_name).suffix.casefold()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Supported extensions: {supported}.")

    if not content:
        raise ValueError("Uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file is larger than the 10 MB limit.")

    return extension


def _decode_text_file(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


def _convert_with_markitdown(file_name: str, content: bytes) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise RuntimeError(
            "MarkItDown is required for PDF/DOCX extraction. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    suffix = Path(file_name).suffix
    with tempfile.TemporaryDirectory(prefix="resume_job_upload_") as temp_dir:
        temp_path = Path(temp_dir) / f"uploaded{suffix}"
        temp_path.write_bytes(content)

        result = MarkItDown(enable_plugins=False).convert(str(temp_path))
        extracted = getattr(result, "text_content", None) or getattr(result, "markdown", "")
        return str(extracted)


def extract_document_text(file_name: str, content: bytes) -> ExtractedDocument:
    """Extract text from an uploaded PDF, DOCX, TXT, or Markdown document."""

    extension = validate_upload(file_name, content)
    if extension in {".txt", ".md"}:
        extracted_text = _decode_text_file(content)
        converter = "plain-text"
    else:
        extracted_text = _convert_with_markitdown(file_name, content)
        converter = "markitdown"

    extracted_text = extracted_text.strip()
    if not extracted_text:
        raise ValueError("No readable text could be extracted from the uploaded file.")

    return ExtractedDocument(
        file_name=Path(file_name).name,
        extracted_text=extracted_text,
        character_count=len(extracted_text),
        converter=converter,
    )
