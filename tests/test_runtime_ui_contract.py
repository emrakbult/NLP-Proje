from pathlib import Path

from fastapi.testclient import TestClient

from api import app
from src.neural_matcher import DEFAULT_BIENCODER_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_health_reports_optimal_biencoder_runtime_path() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model_path"] == str(DEFAULT_BIENCODER_MODEL_PATH)


def test_compare_models_is_not_exposed_in_runtime_api() -> None:
    client = TestClient(app)
    response = client.post(
        "/compare",
        json={"resume_text": "Python", "job_description_text": "Python"},
    )

    assert response.status_code == 404


def test_frontend_no_longer_renders_model_comparison_workflow() -> None:
    app_source = (PROJECT_ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "Compare Models" not in app_source
    assert "ModelComparison" not in app_source
