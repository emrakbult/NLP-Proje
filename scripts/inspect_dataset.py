from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_DATASET_DIR, get_sample, get_split_path, summarize_dataset


def shorten(text: str, max_length: int = 240) -> str:
    """Return a compact one-line preview."""

    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def main() -> None:
    print(f"Dataset directory: {DEFAULT_DATASET_DIR}")
    print()

    summaries = summarize_dataset()
    for summary in summaries:
        print(f"{summary.name}.csv")
        print(f"  path: {summary.path}")
        print(f"  rows: {summary.row_count}")
        print(f"  columns: {', '.join(summary.columns)}")
        print("  labels:")
        for label, count in sorted(summary.label_distribution.items()):
            print(f"    {label}: {count}")
        print()

    sample = get_sample(get_split_path("train"))
    print("Sample training record")
    print(f"  label: {sample['label']}")
    print(f"  resume: {shorten(sample['resume_text'])}")
    print(f"  job: {shorten(sample['job_description_text'])}")


if __name__ == "__main__":
    main()
