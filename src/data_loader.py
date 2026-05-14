from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "cnamuangtoun_resume_job_description_fit"
REQUIRED_COLUMNS = ("resume_text", "job_description_text", "label")


@dataclass(frozen=True)
class SplitSummary:
    """Summary information for one dataset split."""

    name: str
    path: Path
    row_count: int
    columns: tuple[str, ...]
    label_distribution: dict[str, int]


def get_split_path(split: str, dataset_dir: Path = DEFAULT_DATASET_DIR) -> Path:
    """Return the CSV path for a split name such as train or test."""

    return dataset_dir / f"{split}.csv"


def validate_dataset_file(path: Path) -> None:
    """Validate that a dataset CSV exists and contains the required columns."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {path.name}: {missing}")


def iter_records(path: Path) -> Iterable[dict[str, str]]:
    """Yield records from a dataset CSV file."""

    validate_dataset_file(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield {
                "resume_text": row["resume_text"],
                "job_description_text": row["job_description_text"],
                "label": row["label"],
            }


def load_records(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load records from a split into memory."""

    records: list[dict[str, str]] = []
    for index, record in enumerate(iter_records(path)):
        if limit is not None and index >= limit:
            break
        records.append(record)
    return records


def summarize_split(path: Path, split_name: str | None = None) -> SplitSummary:
    """Create a row count and label distribution summary for one split."""

    validate_dataset_file(path)

    label_counts: Counter[str] = Counter()
    row_count = 0
    columns: tuple[str, ...] = ()

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        for row in reader:
            row_count += 1
            label_counts[row["label"]] += 1

    return SplitSummary(
        name=split_name or path.stem,
        path=path,
        row_count=row_count,
        columns=columns,
        label_distribution=dict(label_counts),
    )


def summarize_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> list[SplitSummary]:
    """Summarize all expected dataset splits."""

    return [
        summarize_split(get_split_path("train", dataset_dir), "train"),
        summarize_split(get_split_path("test", dataset_dir), "test"),
    ]


def get_sample(path: Path) -> dict[str, str]:
    """Return the first record from a dataset split."""

    for record in iter_records(path):
        return record
    raise ValueError(f"Dataset split is empty: {path}")
