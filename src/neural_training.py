from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation import pearson_correlation, spearman_correlation
from src.neural_matcher import LABEL_TO_ID, label_to_similarity_target


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_cosine_loss: float
    train_classification_loss: float
    train_accuracy: float
    train_pearson: float
    train_spearman: float
    validation_total_loss: float
    validation_cosine_loss: float
    validation_classification_loss: float
    validation_accuracy: float
    validation_pearson: float
    validation_spearman: float


class ResumeJobPairDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, str]],
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self.records = list(records)
        self.resume_tokens = tokenizer(
            [record["resume_text"] for record in self.records],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.job_tokens = tokenizer(
            [record["job_description_text"] for record in self.records],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.similarity_targets = torch.tensor(
            [label_to_similarity_target(record["label"]) for record in self.records],
            dtype=torch.float32,
        )
        self.class_labels = torch.tensor(
            [LABEL_TO_ID[record["label"]] for record in self.records],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "resume_input_ids": self.resume_tokens["input_ids"][index],
            "resume_attention_mask": self.resume_tokens["attention_mask"][index],
            "job_input_ids": self.job_tokens["input_ids"][index],
            "job_attention_mask": self.job_tokens["attention_mask"][index],
            "similarity_target": self.similarity_targets[index],
            "class_label": self.class_labels[index],
        }


def stratified_train_validation_split(
    records: Sequence[dict[str, str]],
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    rng = random.Random(seed)
    train_records: list[dict[str, str]] = []
    validation_records: list[dict[str, str]] = []

    for label in sorted(grouped):
        group = grouped[label][:]
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * validation_ratio))
        validation_records.extend(group[:validation_count])
        train_records.extend(group[validation_count:])

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    return train_records, validation_records


def create_dataloader(
    records: Sequence[dict[str, str]],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ResumeJobPairDataset(records, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def compute_combined_loss(
    output: dict[str, torch.Tensor],
    similarity_target: torch.Tensor,
    class_label: torch.Tensor,
    cosine_weight: float = 0.6,
    classification_weight: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine_loss = torch.nn.functional.mse_loss(output["similarity_score"], similarity_target)
    classification_loss = torch.nn.functional.cross_entropy(output["class_logits"], class_label)
    total_loss = (cosine_weight * cosine_loss) + (classification_weight * classification_loss)
    return total_loss, cosine_loss, classification_loss


def _safe_correlation(predictions: list[float], targets: list[float], kind: str) -> float:
    if not predictions or len(set(targets)) <= 1:
        return 0.0
    if kind == "pearson":
        return pearson_correlation(predictions, targets)
    return spearman_correlation(predictions, targets)


def run_epoch(
    model: Any,
    dataloader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    cosine_weight: float = 0.6,
    classification_weight: float = 0.4,
) -> dict[str, float]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    cosine_loss_sum = 0.0
    classification_loss_sum = 0.0
    correct = 0
    count = 0
    predicted_scores: list[float] = []
    target_scores: list[float] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model.model(
                resume_input_ids=batch["resume_input_ids"],
                resume_attention_mask=batch["resume_attention_mask"],
                job_input_ids=batch["job_input_ids"],
                job_attention_mask=batch["job_attention_mask"],
            )
            total_loss, cosine_loss, classification_loss = compute_combined_loss(
                output,
                batch["similarity_target"],
                batch["class_label"],
                cosine_weight=cosine_weight,
                classification_weight=classification_weight,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            batch_size = int(batch["class_label"].shape[0])
            total_loss_sum += float(total_loss.item()) * batch_size
            cosine_loss_sum += float(cosine_loss.item()) * batch_size
            classification_loss_sum += float(classification_loss.item()) * batch_size
            predictions = torch.argmax(output["class_logits"], dim=1)
            correct += int((predictions == batch["class_label"]).sum().item())
            count += batch_size
            predicted_scores.extend(output["similarity_score"].detach().cpu().tolist())
            target_scores.extend(batch["similarity_target"].detach().cpu().tolist())

    return {
        "total_loss": total_loss_sum / max(count, 1),
        "cosine_loss": cosine_loss_sum / max(count, 1),
        "classification_loss": classification_loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
        "pearson": _safe_correlation(predicted_scores, target_scores, "pearson"),
        "spearman": _safe_correlation(predicted_scores, target_scores, "spearman"),
    }


def select_best_epoch(history: Sequence[dict[str, float]]) -> int:
    if not history:
        raise ValueError("Training history must not be empty.")
    best_row = min(history, key=lambda row: row["validation_total_loss"])
    return int(best_row["epoch"])


def write_history_csv(path: Path, history: Sequence[dict[str, float]]) -> None:
    if not history:
        raise ValueError("Training history must not be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
