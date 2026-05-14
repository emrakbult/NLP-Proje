from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_matching import select_records
from src.data_loader import get_split_path, load_records
from src.evaluation import score_records
from src.similarity import DEFAULT_MODEL_NAME, load_model
from src.skill_extractor import load_skills


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze high-confidence matching errors.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split.")
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Number of records to analyze. Use 0 to analyze the whole split.",
    )
    parser.add_argument(
        "--sample-mode",
        default="balanced",
        choices=["balanced", "first", "random"],
        help="How records are selected when --limit is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoder batch size.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence Transformer model name.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of examples per error group.")
    return parser.parse_args()


def shorten(text: str, max_length: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        shortened = compact
    else:
        shortened = compact[: max_length - 3] + "..."
    return shortened.encode("ascii", errors="ignore").decode("ascii")


def print_case(title: str, rows: list[dict[str, object]], top_n: int) -> None:
    print(title)
    if not rows:
        print("  No examples found.")
        print()
        return

    for index, row in enumerate(rows[:top_n], start=1):
        print(f"  Case {index}")
        print(f"    label: {row['label']}")
        print(f"    predicted label: {row['predicted_label']}")
        print(f"    semantic score: {float(row['semantic_score']):.2f}")
        print(f"    unweighted skill match score: {float(row['unweighted_skill_match_score']):.2f}")
        print(f"    skill match score: {float(row['skill_match_score']):.2f}")
        print(f"    overall score: {float(row['overall_score']):.2f}")
        print(f"    matched skills: {', '.join(row['matched_skills']) or '-'}")
        print(f"    missing skills: {', '.join(row['missing_skills']) or '-'}")
        print(f"    resume preview: {shorten(str(row['resume_text']))}")
        print(f"    job preview: {shorten(str(row['job_description_text']))}")
        print()


def main() -> None:
    args = parse_args()
    record_limit = None if args.limit == 0 else args.limit
    all_records = load_records(get_split_path(args.split))
    records = select_records(
        all_records,
        limit=record_limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    print(f"Analyzing split: {args.split}")
    print(f"Records: {len(records)}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Encoder model: {args.model}")
    print()

    model = load_model(args.model)
    skills = load_skills()
    rows = score_records(
        records,
        model=model,
        skill_dictionary=skills,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    good_fit_low_score = sorted(
        [row for row in rows if row["label"] == "Good Fit"],
        key=lambda row: float(row["overall_score"]),
    )
    no_fit_high_score = sorted(
        [row for row in rows if row["label"] == "No Fit"],
        key=lambda row: float(row["overall_score"]),
        reverse=True,
    )
    potential_extreme_low = sorted(
        [row for row in rows if row["label"] == "Potential Fit"],
        key=lambda row: float(row["overall_score"]),
    )
    potential_extreme_high = sorted(
        [row for row in rows if row["label"] == "Potential Fit"],
        key=lambda row: float(row["overall_score"]),
        reverse=True,
    )

    print_case("Good Fit examples with the lowest overall scores", good_fit_low_score, args.top_n)
    print_case("No Fit examples with the highest overall scores", no_fit_high_score, args.top_n)
    print_case("Potential Fit examples with the lowest overall scores", potential_extreme_low, args.top_n)
    print_case("Potential Fit examples with the highest overall scores", potential_extreme_high, args.top_n)


if __name__ == "__main__":
    main()
