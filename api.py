from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.document_extraction import extract_document_text
from src.matcher import match_resume_to_job
from src.similarity import (
    BASE_MODEL_PATH,
    DEFAULT_MODEL_NAME,
    FALLBACK_FINE_TUNED_MODEL_PATH,
    FINAL_FINE_TUNED_MODEL_PATH,
    load_model,
)
from src.skill_evidence import DEFAULT_SKILL_EVIDENCE_MODEL_PATH, is_skill_evidence_model_available
from src.skill_extractor import load_skills


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME_PATH = PROJECT_ROOT / "data" / "examples" / "sample_resume.txt"
SAMPLE_JOB_PATH = PROJECT_ROOT / "data" / "examples" / "sample_job.txt"
MODEL_COMPARISON_SPECS = [
    {
        "model_id": "final",
        "model_label": "Final Model",
        "description": "Selected local fine-tuned MiniLM model",
        "model_name": str(FINAL_FINE_TUNED_MODEL_PATH),
        "requires_path": True,
    },
    {
        "model_id": "previous",
        "model_label": "Previous Model",
        "description": "Earlier local fine-tuned MiniLM model",
        "model_name": str(FALLBACK_FINE_TUNED_MODEL_PATH),
        "requires_path": True,
    },
    {
        "model_id": "baseline",
        "model_label": "Base MiniLM",
        "description": "Local pretrained MiniLM encoder before task adaptation",
        "model_name": str(BASE_MODEL_PATH),
        "requires_path": True,
    },
]


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_description_text: str = Field(..., min_length=1)


class SampleResponse(BaseModel):
    resume_text: str
    job_description_text: str


def compact_model_name(model_name: str) -> str:
    path = Path(model_name)
    return path.name if path.exists() else model_name


@lru_cache(maxsize=1)
def get_resources() -> tuple[Any, Any]:
    """Load model and skill dictionary once for the API process."""

    return get_model(DEFAULT_MODEL_NAME), get_skill_dictionary()


@lru_cache(maxsize=None)
def get_model(model_name: str) -> Any:
    """Load and cache a specific model for comparison requests."""

    return load_model(model_name)


@lru_cache(maxsize=1)
def get_skill_dictionary() -> Any:
    """Load and cache the skill dictionary independently from model loading."""

    return load_skills()


app = FastAPI(
    title="Resume-Job Match Analysis API",
    description="FastAPI backend for the explainable NLP resume-job matching system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": compact_model_name(DEFAULT_MODEL_NAME),
        "model_path": DEFAULT_MODEL_NAME,
        "skill_evidence_classifier_available": is_skill_evidence_model_available(),
        "skill_evidence_classifier_path": str(DEFAULT_SKILL_EVIDENCE_MODEL_PATH),
    }


@app.get("/sample", response_model=SampleResponse)
def sample() -> SampleResponse:
    return SampleResponse(
        resume_text=SAMPLE_RESUME_PATH.read_text(encoding="utf-8"),
        job_description_text=SAMPLE_JOB_PATH.read_text(encoding="utf-8"),
    )


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> dict[str, object]:
    file_name = file.filename or "uploaded"
    content = await file.read()

    try:
        extracted = extract_document_text(file_name, content)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "file_name": extracted.file_name,
        "extracted_text": extracted.extracted_text,
        "character_count": extracted.character_count,
        "converter": extracted.converter,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, object]:
    resume_text = request.resume_text.strip()
    job_description_text = request.job_description_text.strip()

    if not resume_text or not job_description_text:
        raise HTTPException(
            status_code=400,
            detail="Resume text and job description text must not be empty.",
        )

    model, skill_dictionary = get_resources()
    result = match_resume_to_job(
        resume_text=resume_text,
        job_description_text=job_description_text,
        model=model,
        skill_dictionary=skill_dictionary,
    )

    return {
        "model": compact_model_name(DEFAULT_MODEL_NAME),
        **result,
    }


@app.post("/compare")
def compare_models(request: AnalyzeRequest) -> dict[str, object]:
    resume_text = request.resume_text.strip()
    job_description_text = request.job_description_text.strip()

    if not resume_text or not job_description_text:
        raise HTTPException(
            status_code=400,
            detail="Resume text and job description text must not be empty.",
        )

    skill_dictionary = get_skill_dictionary()
    items: list[dict[str, object]] = []

    for spec in MODEL_COMPARISON_SPECS:
        model_name = str(spec["model_name"])
        if spec["requires_path"] and not Path(model_name).exists():
            items.append(
                {
                    "model_id": spec["model_id"],
                    "model_label": spec["model_label"],
                    "description": spec["description"],
                    "model": compact_model_name(model_name),
                    "available": False,
                    "error": "Local model path was not found.",
                }
            )
            continue

        try:
            model = get_model(model_name)
            result = match_resume_to_job(
                resume_text=resume_text,
                job_description_text=job_description_text,
                model=model,
                skill_dictionary=skill_dictionary,
            )
            items.append(
                {
                    "model_id": spec["model_id"],
                    "model_label": spec["model_label"],
                    "description": spec["description"],
                    "model": compact_model_name(model_name),
                    "available": True,
                    "semantic_score": result["semantic_score"],
                    "overall_score": result["overall_score"],
                    "skill_match_score": result["skill_match_score"],
                    "unweighted_skill_match_score": result["unweighted_skill_match_score"],
                    "match_category": result["match_category"],
                }
            )
        except Exception as error:
            items.append(
                {
                    "model_id": spec["model_id"],
                    "model_label": spec["model_label"],
                    "description": spec["description"],
                    "model": compact_model_name(model_name),
                    "available": False,
                    "error": str(error),
                }
            )

    return {"items": items}
