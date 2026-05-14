from __future__ import annotations

import pytest

from src.fine_tuning import label_to_similarity, records_to_similarity_columns, select_records


def make_record(label: str, index: int) -> dict[str, str]:
    return {
        "resume_text": f"Resume text {index}",
        "job_description_text": f"Job description text {index}",
        "label": label,
    }


def test_label_to_similarity_maps_fit_labels() -> None:
    assert label_to_similarity("No Fit") == 0.0
    assert label_to_similarity("Potential Fit") == 0.5
    assert label_to_similarity("Good Fit") == 1.0


def test_label_to_similarity_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        label_to_similarity("Excellent Fit")


def test_select_records_balanced_returns_mixed_labels() -> None:
    records = []
    for index in range(6):
        records.append(make_record("No Fit", index))
        records.append(make_record("Potential Fit", index))
        records.append(make_record("Good Fit", index))

    selected = select_records(records, limit=9, sample_mode="balanced", seed=7)
    label_counts = {
        label: sum(record["label"] == label for record in selected)
        for label in ("No Fit", "Potential Fit", "Good Fit")
    }

    assert len(selected) == 9
    assert label_counts == {"No Fit": 3, "Potential Fit": 3, "Good Fit": 3}


def test_records_to_similarity_columns_cleans_and_maps_scores() -> None:
    records = [
        {
            "resume_text": " Python\n\nSQL ",
            "job_description_text": " Python role\twith SQL ",
            "label": "Good Fit",
        }
    ]

    resumes, jobs, scores = records_to_similarity_columns(records)

    assert resumes == ["Python SQL"]
    assert jobs == ["Python role with SQL"]
    assert scores == [1.0]
