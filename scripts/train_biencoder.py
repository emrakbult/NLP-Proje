from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_split_path, load_records
from src.neural_matcher import DEFAULT_BASE_MODEL_PATH, ResumeJobBiEncoder
from src.neural_training import (
    create_dataloader,
    run_epoch,
    select_best_epoch,
    stratified_train_validation_split,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom resume-job bi-encoder.")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL_PATH), help="Local pretrained MiniLM path.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum token length.")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Stratified validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cosine-weight", type=float, default=0.6, help="Cosine MSE loss weight.")
    parser.add_argument("--classification-weight", type=float, default=0.4, help="CE loss weight.")
    parser.add_argument("--train-limit", type=int, default=0, help="Use 0 for the full train split.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. Auto uses CUDA when available.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="models/resume-job-biencoder-checkpoints",
        help="Directory for per-epoch checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/resume-job-biencoder-optimal",
        help="Directory for the selected optimal model.",
    )
    parser.add_argument("--reports-dir", default="reports", help="Directory for training artifacts.")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested_device


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_model_path = Path(args.base_model)
    if not base_model_path.is_absolute():
        base_model_path = PROJECT_ROOT / base_model_path

    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    output_dir = PROJECT_ROOT / args.output_dir
    reports_dir = PROJECT_ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(get_split_path("train"), limit=None)
    if args.train_limit > 0:
        records = records[: args.train_limit]

    train_records, validation_records = stratified_train_validation_split(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    model = ResumeJobBiEncoder(base_model_path=base_model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loader = create_dataloader(
        train_records,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )
    validation_loader = create_dataloader(
        validation_records,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    config: dict[str, Any] = {
        "architecture": "ResumeJobBiEncoder",
        "base_model": relative(base_model_path),
        "output_dir": relative(output_dir),
        "checkpoint_dir": relative(checkpoint_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "cosine_weight": args.cosine_weight,
        "classification_weight": args.classification_weight,
        "device": device,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_label_distribution": dict(Counter(record["label"] for record in train_records)),
        "validation_label_distribution": dict(Counter(record["label"] for record in validation_records)),
    }
    (reports_dir / "biencoder_training_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(config, indent=2))

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            cosine_weight=args.cosine_weight,
            classification_weight=args.classification_weight,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            cosine_weight=args.cosine_weight,
            classification_weight=args.classification_weight,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_cosine_loss": train_metrics["cosine_loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_pearson": train_metrics["pearson"],
            "train_spearman": train_metrics["spearman"],
            "validation_total_loss": validation_metrics["total_loss"],
            "validation_cosine_loss": validation_metrics["cosine_loss"],
            "validation_classification_loss": validation_metrics["classification_loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_pearson": validation_metrics["pearson"],
            "validation_spearman": validation_metrics["spearman"],
        }
        history.append(row)

        epoch_checkpoint = checkpoint_dir / f"epoch-{epoch:02d}"
        model.save_pretrained(epoch_checkpoint, tokenizer=tokenizer)

        if row["validation_total_loss"] < best_validation_loss:
            best_validation_loss = row["validation_total_loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        write_history_csv(reports_dir / "biencoder_training_history.csv", history)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={row['train_total_loss']:.4f} "
            f"val_loss={row['validation_total_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.4f} "
            f"val_pearson={row['validation_pearson']:.4f}"
        )

    selected_epoch = select_best_epoch(history)
    if selected_epoch != best_epoch:
        best_epoch = selected_epoch

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    model.load_state_dict(best_state)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    model.save_pretrained(output_dir, tokenizer=tokenizer)

    selection = {
        "best_epoch": best_epoch,
        "selection_metric": "validation_total_loss",
        "best_validation_total_loss": best_validation_loss,
        "optimal_model_dir": relative(output_dir),
    }
    (reports_dir / "biencoder_optimal_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(f"Saved optimal model to: {output_dir}")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
