"""Validation for Finance-labelled invoice-distribution data used in certification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

try:
    from .segment3_taxonomy import ACCOUNT_TYPE_TO_SEGMENT3
except ImportError:
    from segment3_taxonomy import ACCOUNT_TYPE_TO_SEGMENT3


_REQUIRED_FIELDS = frozenset(
    {
        "invoice_distribution_id",
        "invoice_date",
        "source_group_id",
        "vendor_id",
        "vendor_name",
        "vendor_site_id",
        "business_unit",
        "legal_entity",
        "ledger",
        "chart_of_accounts",
        "line_type",
        "line_description",
        "line_amount",
        "currency",
        "segment3",
        "final_posted",
        "is_synthetic",
    }
)


def _is_synthetic(value: Any) -> bool:
    """Normalize booleans and common spreadsheet flags without truthiness bugs."""
    if isinstance(value, str):
        return value.strip().upper() in {"Y", "YES", "TRUE", "1"}
    return bool(value)


def validate_finance_rows(
    rows: Iterable[Mapping[str, Any]], *, strict: bool = False
) -> dict[str, int]:
    """Validate governed rows and count rows eligible for real-data certification.

    Args:
        rows: Finance-labelled invoice distribution records.
        strict: Require the complete posted-distribution contract. The legacy
            POC workbook uses the compatibility default; certification loaders
            must pass ``strict=True``.

    Returns:
        Counts split between real and synthetic records.

    Raises:
        ValueError: If a row lacks a required value or repeats a distribution id.
    """
    materialized = [dict(row) for row in rows]
    required_fields = set(_REQUIRED_FIELDS)
    if strict:
        required_fields.update(
            {
                "segment1",
                "segment2",
                "segment4",
                "segment5",
                "segment6",
                "natural_account_description",
            }
        )
    seen_ids: set[str] = set()
    real_count = 0
    synthetic_count = 0
    for index, row in enumerate(materialized):
        missing = sorted(
            field
            for field in required_fields
            if row.get(field) is None or (isinstance(row.get(field), str) and not row[field].strip())
        )
        if missing:
            raise ValueError(f"Finance row {index} missing required fields: {missing}")
        if strict and not _is_synthetic(row["is_synthetic"]):
            if str(row.get("final_posted")).strip().upper() not in {"Y", "YES", "TRUE", "1"} and row.get("final_posted") is not True:
                raise ValueError(f"Finance row {index} is not a final posted distribution")
            segment3 = str(row.get("segment3") or "").strip()
            valid_segments = set(ACCOUNT_TYPE_TO_SEGMENT3.values())
            if segment3 not in valid_segments:
                raise ValueError(f"Finance row {index} has an invalid Segment 3 value: {segment3!r}")
        distribution_id = str(row["invoice_distribution_id"]).strip()
        if distribution_id in seen_ids:
            raise ValueError(f"Duplicate invoice_distribution_id: {distribution_id}")
        seen_ids.add(distribution_id)
        if _is_synthetic(row["is_synthetic"]):
            synthetic_count += 1
        else:
            real_count += 1
    return {
        "row_count": len(materialized),
        "real_row_count": real_count,
        "synthetic_row_count": synthetic_count,
    }
