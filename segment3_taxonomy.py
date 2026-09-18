"""Shared approved Segment 3 taxonomy and normalization rules."""

from __future__ import annotations

import re
import unicodedata
import json
from pathlib import Path
from typing import Mapping, Any


ACCOUNT_TYPE_TO_SEGMENT3 = {
    "Accounts Payable Clearing": "22190",
    "Accrued Expenses": "24220",
    "Accrued Receipts": "22210",
    "Airfare": "60512",
    "Asset Clearing": "15910",
    "Car Mileage": "60514",
    "Contractor Expenses": "65600",
    "Hotel / Accomodation": "60530",
    "Meals": "60521",
    "Miscellaneous": "60540",
    "Operating Lease Expense - Company Labor": "63611",
    "Operating Lease Expense - Equipment Rental": "63612",
    "Operating Lease Expense - Electricity": "63613",
    "Project Labor Cost": "59110",
    "Supplies": "60520",
    "Withholding Tax Payable": "25400",
}


def normalize_account_type(account_type: str) -> str:
    """Normalize an account type without changing its approved taxonomy meaning."""
    value = unicodedata.normalize("NFKC", str(account_type or ""))
    value = value.replace("\ufffd", "-").replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"\s*[-]\s*", " - ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_taxonomy_artifact(path: Path | None) -> dict[str, str]:
    """Load Finance's account-type-to-Segment-3 map when configured.

    The embedded 16-label map is a demo fallback only. A production artifact
    may be either ``{"Account name": "60520"}`` or
    ``{"mappings": [{"account_type": ..., "segment3": ...}]}``.
    """
    if path is None or not path.is_file():
        return dict(ACCOUNT_TYPE_TO_SEGMENT3)
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and isinstance(value.get("mappings"), list):
        value = {
            str(item.get("account_type")): str(item.get("segment3"))
            for item in value["mappings"]
            if isinstance(item, Mapping) and item.get("account_type") and item.get("segment3")
        }
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Segment 3 taxonomy artifact must be a non-empty object")
    return {
        normalize_account_type(str(account_type)): str(segment3).strip()
        for account_type, segment3 in value.items()
        if str(account_type).strip() and str(segment3).strip()
    }
