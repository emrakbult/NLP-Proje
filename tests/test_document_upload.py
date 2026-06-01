from fastapi.testclient import TestClient

from api import app
from src.document_extraction import MAX_UPLOAD_BYTES


client = TestClient(app)


def test_extract_text_uploads_txt_file() -> None:
    response = client.post(
        "/extract-text",
        files={"file": ("resume.txt", b"Python and SQL experience.", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["file_name"] == "resume.txt"
    assert body["extracted_text"] == "Python and SQL experience."
    assert body["character_count"] == len("Python and SQL experience.")


def test_extract_text_rejects_unsupported_extension() -> None:
    response = client.post(
        "/extract-text",
        files={"file": ("resume.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Desteklenmeyen dosya türü" in response.json()["detail"]


def test_extract_text_rejects_empty_file() -> None:
    response = client.post(
        "/extract-text",
        files={"file": ("resume.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert "boş" in response.json()["detail"].casefold()


def test_extract_text_rejects_oversized_file() -> None:
    response = client.post(
        "/extract-text",
        files={"file": ("resume.txt", b"x" * (MAX_UPLOAD_BYTES + 1), "text/plain")},
    )

    assert response.status_code == 400
    assert "10 MB" in response.json()["detail"]
