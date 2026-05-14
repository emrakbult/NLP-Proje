import pytest

from src.similarity import load_model


@pytest.fixture(scope="session")
def encoder_model():
    """Load the real encoder-only Sentence Transformer model once per test session."""

    return load_model()
