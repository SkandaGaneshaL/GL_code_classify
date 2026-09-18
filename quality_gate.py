"""Input-quality checks for safe, one-natural-account classification."""

from __future__ import annotations

import re
from typing import Any, Mapping


_ACCOUNT_NATURE_PATTERNS = {
    "asset": re.compile(r"\b(?:asset|capital|workstation|equipment acquisition|monitor|desktop)\b", re.I),
    "lease": re.compile(r"\b(?:lease|rental)\b", re.I),
    "tax": re.compile(r"\b(?:withholding|wht|gst|vat|tax)\b", re.I),
    "travel": re.compile(r"\b(?:air travel|airfare|hotel|lodging|mileage)\b", re.I),
    "meals": re.compile(r"\b(?:meal|meals|catering)\b", re.I),
    "labor": re.compile(r"\b(?:contractor|consultant|labou?r|professional services)\b", re.I),
    "supplies": re.compile(r"\b(?:supplies|paper|toner|stationery)\b", re.I),
}
_CLAUSE_BOUNDARY = re.compile(r"(?:,|;|\band\b|\bor\b)", re.I)


def _account_natures(description: str) -> set[str]:
    return {name for name, pattern in _ACCOUNT_NATURE_PATTERNS.items() if pattern.search(description)}


def assess_line_quality(feature: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit review reasons without pretending that they are probabilities.

    A source line with multiple recognizable accounting natures separated by
    clauses cannot truthfully receive one natural-account prediction. Missing
    vendor or line type makes the evidence incomplete, so it is review-only but
    can still be scored for an analyst suggestion.
    """
    description = str(feature.get("line_description") or "").strip()
    review_reasons: list[str] = []
    if not description:
        review_reasons.append("missing_line_description")
    if not str(feature.get("vendor_name_norm") or "").strip():
        review_reasons.append("missing_vendor")
    if str(feature.get("line_type_norm") or "UNKNOWN").upper() == "UNKNOWN":
        review_reasons.append("missing_line_type")

    natures = _account_natures(description)
    has_clause_boundary = bool(_CLAUSE_BOUNDARY.search(description))
    compound = len(natures) >= 2 and has_clause_boundary
    if compound:
        review_reasons.append("compound_account_nature")

    blocks_classification = not description or compound
    return {
        "decision": "REVIEW_REQUIRED" if review_reasons else "CONTINUE",
        "blocks_classification": blocks_classification,
        "review_reasons": review_reasons,
        "quality_status": "sufficient" if not review_reasons else "insufficient",
    }
