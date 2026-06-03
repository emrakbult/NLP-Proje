from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "base-minilm"
OPTIMAL_BIENCODER_MODEL_PATH = PROJECT_ROOT / "models" / "resume-job-biencoder-optimal"
FINAL_FINE_TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "resume-job-minilm-finetuned-2epoch"


def resolve_default_model_name() -> str:
    """Return the local optimal model used at runtime."""

    return str(OPTIMAL_BIENCODER_MODEL_PATH)


DEFAULT_MODEL_NAME = resolve_default_model_name()


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load the encoder-only Sentence Transformer model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for model loading. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    model_path = Path(model_name)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")

    if model_path.exists():
        from src.neural_matcher import is_biencoder_model_dir, load_biencoder_runtime

        if is_biencoder_model_dir(model_path):
            return load_biencoder_runtime(model_path)

    return SentenceTransformer(model_name)


def _as_float_vector(vector: Any) -> list[float]:
    """Convert numpy arrays, tensors, or sequences into a plain float list."""

    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    if not isinstance(vector, Sequence) or isinstance(vector, str):
        raise TypeError("Embedding vector must be a numeric sequence.")

    if vector and isinstance(vector[0], Sequence) and not isinstance(vector[0], str):
        if len(vector) != 1:
            raise ValueError("Expected a single embedding vector, received multiple vectors.")
        vector = vector[0]

    return [float(value) for value in vector]


def encode_text(text: str, model: Any) -> list[float]:
    """Encode text into an embedding vector using an encoder-only model."""

    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return _as_float_vector(embedding)


def encode_texts(
    texts: Sequence[str],
    model: Any,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    """Encode multiple texts into embedding vectors using the real encoder model."""

    if not texts:
        return []

    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
    )

    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    if not isinstance(embeddings, Sequence) or isinstance(embeddings, str):
        raise TypeError("Batch embeddings must be a sequence of embedding vectors.")

    return [_as_float_vector(embedding) for embedding in embeddings]


def cosine_similarity(vector_a: Any, vector_b: Any) -> float:
    """Calculate cosine similarity between two embedding vectors."""

    a = _as_float_vector(vector_a)
    b = _as_float_vector(vector_b)

    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same length.")

    dot_product = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def pairwise_cosine_similarities(vectors_a: Sequence[Any], vectors_b: Sequence[Any]) -> list[float]:
    """Calculate cosine similarity for aligned vector pairs."""

    if len(vectors_a) != len(vectors_b):
        raise ValueError("Vector collections must have the same length.")

    return [
        cosine_similarity(vector_a, vector_b)
        for vector_a, vector_b in zip(vectors_a, vectors_b)
    ]


def similarity_to_percentage(similarity: float) -> float:
    """Convert cosine similarity into a 0-100 score."""

    clamped_similarity = max(0.0, min(1.0, similarity))
    return clamped_similarity * 100


def calculate_cosine_similarity(resume_text: str, job_description_text: str, model: Any) -> float:
    """Calculate raw cosine similarity between resume and job description texts."""

    resume_embedding = encode_text(resume_text, model)
    job_embedding = encode_text(job_description_text, model)
    return cosine_similarity(resume_embedding, job_embedding)


def calculate_semantic_similarity_score(
    resume_text: str,
    job_description_text: str,
    model: Any,
) -> float:
    """Calculate semantic similarity as a 0-100 score."""

    similarity = calculate_cosine_similarity(resume_text, job_description_text, model)
    return similarity_to_percentage(similarity)
