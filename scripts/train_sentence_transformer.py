from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_split_path, load_records
from src.fine_tuning import (
    records_to_input_examples,
    records_to_similarity_columns,
    select_records,
)
from src.similarity import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Sentence Transformer for resume-job fit scoring."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Base Sentence Transformer model.")
    parser.add_argument("--train-split", default="train", choices=["train", "test"], help="Training split.")
    parser.add_argument("--eval-split", default="test", choices=["train", "test"], help="Evaluation split.")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=900,
        help="Training records to use. Use 0 for the full split.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=300,
        help="Evaluation records to use. Use 0 for the full split.",
    )
    parser.add_argument(
        "--sample-mode",
        default="balanced",
        choices=["balanced", "first", "random"],
        help="How records are selected when a limit is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for training steps.")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum token length.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. Auto uses CUDA when available.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use mixed precision training. Leave disabled on GTX 1060 unless tested.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/resume-job-minilm-finetuned",
        help="Directory where the fine-tuned model will be saved.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return requested_device


def clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        cleaned[key] = float(value)
    return cleaned


def build_evaluator(records: list[dict[str, str]], batch_size: int):
    try:
        from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
    except ImportError as error:
        raise ImportError("sentence-transformers is required for fine-tuning.") from error

    resumes, jobs, scores = records_to_similarity_columns(records)
    return EmbeddingSimilarityEvaluator(
        resumes,
        jobs,
        scores,
        batch_size=batch_size,
        name="resume-job-fit",
        show_progress_bar=True,
        write_csv=False,
    )


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(get_split_path(args.train_split), limit=None)
    eval_records = load_records(get_split_path(args.eval_split), limit=None)

    train_records = select_records(train_records, args.train_limit, args.sample_mode, args.seed)
    eval_records = select_records(eval_records, args.eval_limit, args.sample_mode, args.seed)

    device = resolve_device(args.device)

    print(f"Loading base model: {args.model}")
    print(f"Training device: {device}")
    print(f"Train records: {len(train_records)} {dict(Counter(record['label'] for record in train_records))}")
    print(f"Eval records: {len(eval_records)} {dict(Counter(record['label'] for record in eval_records))}")

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import losses

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    train_examples = records_to_input_examples(train_records)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = build_evaluator(eval_records, args.batch_size) if eval_records else None

    baseline_metrics = clean_metrics(evaluator(model)) if evaluator else {}
    if baseline_metrics:
        print("Baseline evaluator metrics:")
        for key, value in baseline_metrics.items():
            print(f"  {key}: {value:.4f}")

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)

    print(f"Training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output_dir),
        save_best_model=evaluator is not None,
        use_amp=args.use_amp,
        show_progress_bar=True,
    )

    model.save(str(output_dir))
    final_metrics = clean_metrics(evaluator(model, output_path=str(output_dir))) if evaluator else {}

    summary = {
        "base_model": args.model,
        "output_dir": str(output_dir),
        "device": device,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_label_distribution": dict(Counter(record["label"] for record in train_records)),
        "eval_label_distribution": dict(Counter(record["label"] for record in eval_records)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
    }

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if final_metrics:
        print("Final evaluator metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")

    print(f"Saved fine-tuned model to: {output_dir}")
    print(f"Saved training summary to: {summary_path}")


if __name__ == "__main__":
    main()
