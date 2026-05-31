from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.skill_evidence import build_classifier_input


LABEL_TO_ID = {"positive": 0, "negated": 1, "unclear": 2}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class SkillEvidenceDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], tokenizer: Any, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            build_classifier_input(record["sentence"], record["skill"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[record["label"]], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a skill evidence classifier.")
    parser.add_argument(
        "--dataset",
        default="data/skill_evidence/skill_evidence_dataset.csv",
        help="CSV dataset with sentence, skill, label, split, notes columns.",
    )
    parser.add_argument(
        "--model",
        default="microsoft/MiniLM-L12-H384-uncased",
        help="Base encoder model for sequence classification.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/skill-evidence-minilm-classifier",
        help="Directory where the classifier will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--max-length", type=int, default=160, help="Maximum token length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. Auto uses CUDA when available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested_device


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    required_columns = {"sentence", "skill", "label", "split", "notes"}
    if not records:
        raise ValueError("Skill evidence dataset is empty.")
    if set(records[0]) != required_columns:
        raise ValueError(f"Dataset columns must be: {sorted(required_columns)}")

    for record in records:
        if record["label"] not in LABEL_TO_ID:
            raise ValueError(f"Unknown label: {record['label']}")
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {record['split']}")

    return records


def evaluate(
    model: Any,
    dataloader: DataLoader,
    device: str,
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            batch_predictions = torch.argmax(outputs.logits, dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    report = classification_report(
        labels,
        predictions,
        target_names=[ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)],
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": accuracy_score(labels, predictions),
        "classification_report": report,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_path = PROJECT_ROOT / args.dataset
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(dataset_path)
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }

    device = resolve_device(args.device)
    print(f"Base model: {args.model}")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    for split, split_data in split_records.items():
        print(f"{split}: {len(split_data)} {dict(Counter(record['label'] for record in split_data))}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    train_loader = DataLoader(
        SkillEvidenceDataset(split_records["train"], tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        SkillEvidenceDataset(split_records["validation"], tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        SkillEvidenceDataset(split_records["test"], tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_accuracy = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        validation_metrics = evaluate(model, validation_loader, device)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"validation_accuracy={validation_metrics['accuracy']:.4f}"
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics = evaluate(model, validation_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
    }
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2),
        encoding="utf-8",
    )

    summary = {
        "base_model": args.model,
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "split_sizes": {split: len(data) for split, data in split_records.items()},
        "label_distribution": {
            split: dict(Counter(record["label"] for record in data))
            for split, data in split_records.items()
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Saved classifier to: {output_dir}")


if __name__ == "__main__":
    main()
