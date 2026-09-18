import pytest

from dataset_splits import split_real_rows_by_time


def test_time_group_split_keeps_groups_together_and_excludes_synthetic_rows():
    rows = [
        {"row_id": "a1", "source_group_id": "a", "invoice_date": "2024-01-01", "is_synthetic": False},
        {"row_id": "a2", "source_group_id": "a", "invoice_date": "2024-01-02", "is_synthetic": False},
        {"row_id": "b", "source_group_id": "b", "invoice_date": "2024-02-01", "is_synthetic": False},
        {"row_id": "c", "source_group_id": "c", "invoice_date": "2024-03-01", "is_synthetic": False},
        {"row_id": "s", "source_group_id": "a", "invoice_date": "2024-01-03", "is_synthetic": True},
    ]

    splits = split_real_rows_by_time(rows, development_fraction=0.5, calibration_fraction=0.25)

    assert {row["row_id"] for row in splits["development"]} == {"a1", "a2"}
    assert {row["row_id"] for row in splits["calibration"]} == {"b"}
    assert {row["row_id"] for row in splits["test"]} == {"c"}


def test_time_group_split_rejects_missing_group_or_date():
    with pytest.raises(ValueError, match="source_group_id"):
        split_real_rows_by_time([{"row_id": "x", "invoice_date": "2024-01-01", "is_synthetic": False}])


def test_n_flag_is_real_data():
    rows = [
        {"row_id": "a", "source_group_id": "a", "invoice_date": "2024-01-01", "is_synthetic": "N"},
        {"row_id": "b", "source_group_id": "b", "invoice_date": "2024-02-01", "is_synthetic": "N"},
        {"row_id": "c", "source_group_id": "c", "invoice_date": "2024-03-01", "is_synthetic": "N"},
    ]
    splits = split_real_rows_by_time(rows)
    assert sum(len(value) for value in splits.values()) == 3
