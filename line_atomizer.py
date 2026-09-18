"""Review-safe suggestions for descriptions containing multiple account natures."""

from __future__ import annotations

import re
from typing import Any


_NATURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("asset", re.compile(r"\b(?:asset|capital|workstation|equipment acquisition|monitor|desktop)\b", re.I)),
    ("lease", re.compile(r"\b(?:lease|rental)\b", re.I)),
    ("tax", re.compile(r"\b(?:withholding|wht|gst|vat|tax)\b", re.I)),
    ("travel", re.compile(r"\b(?:air travel|airfare|hotel|lodging|mileage)\b", re.I)),
    ("meals", re.compile(r"\b(?:meal|meals|catering)\b", re.I)),
    ("labor", re.compile(r"\b(?:contractor|consultant|labou?r|professional services)\b", re.I)),
    ("supplies", re.compile(r"\b(?:supplies|paper|toner|stationery)\b", re.I)),
)
_CLAUSE_BOUNDARY = re.compile(r"\s*(?:,|;|\band\b|\bor\b)\s*", re.I)


def _natures(text: str) -> list[str]:
    """Return recognized account natures in stable priority order."""
    return [name for name, pattern in _NATURE_PATTERNS if pattern.search(text)]


def atomize_line_description(description: str, *, line_amount: Any = None) -> dict[str, Any]:
    """Return only review-required child suggestions; never infer Segment 3 values.

    Splitting is a clerical aid, not an accounting decision. Children intentionally
    carry no account type or Segment 3 value until a reviewer confirms them.
    """
    text = str(description or "").strip()
    clauses = [clause.strip() for clause in _CLAUSE_BOUNDARY.split(text) if clause.strip()]
    children: list[dict[str, Any]] = []
    seen: set[str] = set()
    for clause in clauses:
        for nature in _natures(clause):
            if nature in seen:
                continue
            seen.add(nature)
            children.append(
                {
                    "description": clause,
                    "account_nature": nature,
                    "account_type": None,
                    "segment3": None,
                    "decision": "REVIEW_REQUIRED",
                    "amount": None,
                    "amount_allocation": "REVIEW_REQUIRED",
                }
            )
    if len(children) < 2:
        return {"status": "not_compound", "requires_review": False, "children": []}
    return {
        "status": "split_suggested",
        "requires_review": True,
        "children": children,
        "source_amount": line_amount,
        "amount_allocation": "REVIEW_REQUIRED",
        "note": "Amounts are intentionally not allocated automatically; Finance must approve the split.",
    }
