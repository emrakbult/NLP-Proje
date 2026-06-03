import math

import pytest

from src.neural_matcher import DEFAULT_BASE_MODEL_PATH, LABELS, ResumeJobBiEncoder


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_biencoder_forward_returns_similarity_and_logits() -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(DEFAULT_BASE_MODEL_PATH))
    model = ResumeJobBiEncoder(base_model_path=DEFAULT_BASE_MODEL_PATH)
    model.eval()

    resume = tokenizer(["Python developer with SQL experience."], return_tensors="pt", padding=True)
    job = tokenizer(["Backend role requiring Python and SQL."], return_tensors="pt", padding=True)

    with torch.no_grad():
        output = model.model(
            resume_input_ids=resume["input_ids"],
            resume_attention_mask=resume["attention_mask"],
            job_input_ids=job["input_ids"],
            job_attention_mask=job["attention_mask"],
        )

    assert output["similarity_score"].shape == (1,)
    assert output["cosine_similarity"].shape == (1,)
    assert output["class_logits"].shape == (1, len(LABELS))
    assert output["resume_embedding"].shape[1] == 128
    assert output["job_embedding"].shape[1] == 128


def test_biencoder_temperature_scales_similarity_score() -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(DEFAULT_BASE_MODEL_PATH))
    model = ResumeJobBiEncoder(base_model_path=DEFAULT_BASE_MODEL_PATH)
    model.eval()

    with torch.no_grad():
        model.model.log_temperature.copy_(torch.tensor(math.log(2.0)))

    resume = tokenizer(["Python developer."], return_tensors="pt", padding=True)
    job = tokenizer(["Python engineer."], return_tensors="pt", padding=True)

    with torch.no_grad():
        output = model.model(
            resume_input_ids=resume["input_ids"],
            resume_attention_mask=resume["attention_mask"],
            job_input_ids=job["input_ids"],
            job_attention_mask=job["attention_mask"],
        )

    expected = torch.sigmoid(output["cosine_similarity"] / output["temperature"])
    assert output["temperature"].item() == pytest.approx(2.0)
    assert torch.allclose(output["similarity_score"], expected)
