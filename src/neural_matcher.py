from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "base-minilm"
DEFAULT_BIENCODER_MODEL_PATH = PROJECT_ROOT / "models" / "resume-job-biencoder-optimal"

LABELS = ("No Fit", "Potential Fit", "Good Fit")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def label_to_similarity_target(label: str) -> float:
    if label == "No Fit":
        return 0.0
    if label == "Potential Fit":
        return 0.5
    if label == "Good Fit":
        return 1.0
    raise ValueError(f"Unknown label: {label}")


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).float()
    masked_embeddings = last_hidden_state * mask
    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed_embeddings / token_counts


class ResumeJobBiEncoder:
    """Custom bi-encoder architecture for resume-job fit prediction."""

    def __init__(
        self,
        base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
        projection_dim: int = 128,
        hidden_projection_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        import torch
        from torch import nn
        from transformers import AutoModel

        self.torch = torch
        self.nn = nn
        self.base_model_path = str(base_model_path)
        self.projection_dim = projection_dim
        self.hidden_projection_dim = hidden_projection_dim
        self.dropout = dropout

        class _Model(nn.Module):
            def __init__(self, outer: "ResumeJobBiEncoder") -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(outer.base_model_path)
                encoder_dim = int(self.encoder.config.hidden_size)
                self.projection = nn.Sequential(
                    nn.Linear(encoder_dim, outer.hidden_projection_dim),
                    nn.GELU(),
                    nn.Dropout(outer.dropout),
                    nn.Linear(outer.hidden_projection_dim, outer.projection_dim),
                )
                pair_dim = outer.projection_dim * 4
                self.classifier = nn.Sequential(
                    nn.Linear(pair_dim, outer.hidden_projection_dim),
                    nn.GELU(),
                    nn.Dropout(outer.dropout),
                    nn.Linear(outer.hidden_projection_dim, len(LABELS)),
                )
                self.log_temperature = nn.Parameter(torch.tensor(0.0))

            def encode_inputs(self, input_ids: Any, attention_mask: Any) -> Any:
                import torch.nn.functional as functional

                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pool(outputs.last_hidden_state, attention_mask)
                projected = self.projection(pooled)
                return functional.normalize(projected, p=2, dim=1)

            def forward(
                self,
                resume_input_ids: Any,
                resume_attention_mask: Any,
                job_input_ids: Any,
                job_attention_mask: Any,
            ) -> dict[str, Any]:
                import torch
                import torch.nn.functional as functional

                resume_embedding = self.encode_inputs(resume_input_ids, resume_attention_mask)
                job_embedding = self.encode_inputs(job_input_ids, job_attention_mask)
                cosine_similarity = functional.cosine_similarity(resume_embedding, job_embedding)
                temperature = torch.exp(self.log_temperature).clamp(min=0.05, max=10.0)
                scaled_similarity = cosine_similarity / temperature
                similarity_score = torch.sigmoid(scaled_similarity)
                pair_features = torch.cat(
                    [
                        resume_embedding,
                        job_embedding,
                        torch.abs(resume_embedding - job_embedding),
                        resume_embedding * job_embedding,
                    ],
                    dim=1,
                )
                class_logits = self.classifier(pair_features)
                return {
                    "resume_embedding": resume_embedding,
                    "job_embedding": job_embedding,
                    "cosine_similarity": cosine_similarity,
                    "similarity_score": similarity_score,
                    "class_logits": class_logits,
                    "temperature": temperature,
                }

        self.model = _Model(self)

    def to(self, device: str) -> "ResumeJobBiEncoder":
        self.model.to(device)
        return self

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def parameters(self) -> Any:
        return self.model.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)

    def save_pretrained(self, output_dir: str | Path, tokenizer: Any | None = None) -> None:
        from safetensors.torch import save_file

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)

        config = {
            "base_model_path": str(Path(self.base_model_path).relative_to(PROJECT_ROOT))
            if Path(self.base_model_path).is_absolute()
            else self.base_model_path,
            "projection_dim": self.projection_dim,
            "hidden_projection_dim": self.hidden_projection_dim,
            "dropout": self.dropout,
            "labels": list(LABELS),
            "architecture": "ResumeJobBiEncoder",
        }
        (output_path / "biencoder_config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        save_file(self.model.state_dict(), output_path / "model.safetensors")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ResumeJobBiEncoder":
        from safetensors.torch import load_file

        model_path = Path(model_dir)
        config = json.loads((model_path / "biencoder_config.json").read_text(encoding="utf-8"))
        base_model_path = Path(config["base_model_path"])
        if not base_model_path.is_absolute():
            base_model_path = PROJECT_ROOT / base_model_path

        model = cls(
            base_model_path=base_model_path,
            projection_dim=int(config["projection_dim"]),
            hidden_projection_dim=int(config["hidden_projection_dim"]),
            dropout=float(config["dropout"]),
        )
        state_dict = load_file(model_path / "model.safetensors")
        model.load_state_dict(state_dict)
        return model


class ResumeJobBiEncoderRuntime:
    """SentenceTransformer-like runtime wrapper for the custom bi-encoder."""

    def __init__(self, model_dir: str | Path = DEFAULT_BIENCODER_MODEL_PATH) -> None:
        import torch
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.biencoder = ResumeJobBiEncoder.from_pretrained(self.model_dir).to(self.device)
        self.biencoder.eval()

    def _tokenize(self, texts: list[str], max_length: int = 256) -> dict[str, Any]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def encode(
        self,
        sentences: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        **_: Any,
    ) -> Any:
        del show_progress_bar

        single_input = isinstance(sentences, str)
        texts = [sentences] if single_input else list(sentences)
        embeddings: list[np.ndarray] = []

        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self._tokenize(batch)
                batch_embeddings = self.biencoder.model.encode_inputs(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                )
                embeddings.append(batch_embeddings.detach().cpu().numpy())

        stacked = np.vstack(embeddings) if embeddings else np.empty((0, 128), dtype=np.float32)
        if single_input:
            stacked = stacked[0]
        if convert_to_numpy:
            return stacked
        return stacked.tolist()

    def predict_pair(self, resume_text: str, job_description_text: str) -> dict[str, Any]:
        with self.torch.no_grad():
            resume = self._tokenize([resume_text])
            job = self._tokenize([job_description_text])
            output = self.biencoder.model(
                resume_input_ids=resume["input_ids"],
                resume_attention_mask=resume["attention_mask"],
                job_input_ids=job["input_ids"],
                job_attention_mask=job["attention_mask"],
            )
            probabilities = self.torch.softmax(output["class_logits"], dim=1)[0]
            predicted_id = int(self.torch.argmax(probabilities).item())

        return {
            "cosine_similarity": float(output["cosine_similarity"][0].detach().cpu().item()),
            "semantic_score": float(output["similarity_score"][0].detach().cpu().item() * 100),
            "predicted_fit_label": ID_TO_LABEL[predicted_id],
            "fit_class_probabilities": {
                ID_TO_LABEL[index]: float(probabilities[index].detach().cpu().item())
                for index in range(len(LABELS))
            },
            "temperature": float(output["temperature"].detach().cpu().item()),
        }


def is_biencoder_model_dir(path: str | Path) -> bool:
    return (Path(path) / "biencoder_config.json").exists()


def load_biencoder_runtime(model_dir: str | Path = DEFAULT_BIENCODER_MODEL_PATH) -> ResumeJobBiEncoderRuntime:
    return ResumeJobBiEncoderRuntime(model_dir)
