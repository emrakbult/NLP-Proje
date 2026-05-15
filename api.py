from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.matcher import match_resume_to_job
from src.similarity import DEFAULT_MODEL_NAME, load_model
from src.skill_extractor import load_skills


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME_PATH = PROJECT_ROOT / "data" / "examples" / "sample_resume.txt"
SAMPLE_JOB_PATH = PROJECT_ROOT / "data" / "examples" / "sample_job.txt"


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

    return load_model(DEFAULT_MODEL_NAME), load_skills()


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
    }


@app.get("/sample", response_model=SampleResponse)
def sample() -> SampleResponse:
    return SampleResponse(
        resume_text=SAMPLE_RESUME_PATH.read_text(encoding="utf-8"),
        job_description_text=SAMPLE_JOB_PATH.read_text(encoding="utf-8"),
    )


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
