from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_split_path, load_records
from src.matcher import match_resume_to_job
from src.similarity import DEFAULT_MODEL_NAME, load_model
from src.skill_extractor import load_skills


def shorten(text: str, max_length: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real encoder-based matching on dataset samples.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to sample.")
    parser.add_argument("--limit", type=int, default=3, help="Number of rows to evaluate.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence Transformer model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_path = get_split_path(args.split)
    records = load_records(split_path, limit=args.limit)

    print(f"Loading encoder model: {args.model}")
    model = load_model(args.model)
    skills = load_skills()
    print()

    for index, record in enumerate(records, start=1):
        result = match_resume_to_job(
            resume_text=record["resume_text"],
            job_description_text=record["job_description_text"],
            model=model,
            skill_dictionary=skills,
        )

        print(f"Sample {index}")
        print(f"  dataset label: {record['label']}")
        print(f"  semantic score: {result['semantic_score']}")
        print(f"  skill match score: {result['skill_match_score']}")
        print(f"  overall score: {result['overall_score']}")
        print(f"  match category: {result['match_category']}")
        print(f"  matched skills: {', '.join(result['matched_skills']) or '-'}")
        print(f"  missing skills: {', '.join(result['missing_skills']) or '-'}")
        print(f"  HR evaluation: {result['hr_evaluation']}")
        print(f"  interview focus: {' | '.join(result['interview_focus']) or '-'}")
        print(f"  resume preview: {shorten(record['resume_text'])}")
        print(f"  job preview: {shorten(record['job_description_text'])}")
        print()


if __name__ == "__main__":
    main()
