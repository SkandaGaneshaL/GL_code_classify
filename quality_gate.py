"""Input-quality checks for safe, one-natural-account classification."""

from __future__ import annotations

import os
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
_MIN_EXTRACTION_QUALITY = float(os.getenv("GL_MIN_EXTRACTION_QUALITY", "0.80"))


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
    if str(feature.get("line_type_norm") or "").upper() == "TAX":
        review_reasons.append("unsupported_tax")
    extraction_quality = feature.get("extraction_quality")
    try:
        quality_value = float(extraction_quality) if extraction_quality is not None else None
    except (TypeError, ValueError):
        quality_value = None
    if quality_value is not None and quality_value < _MIN_EXTRACTION_QUALITY:
        review_reasons.append("low_extraction_quality")
    if feature.get("extraction_status") in {"failed", "partial", "low_quality"}:
        review_reasons.append("low_extraction_quality")

    natures = _account_natures(description)
    has_clause_boundary = bool(_CLAUSE_BOUNDARY.search(description))
    compound = len(natures) >= 2 and has_clause_boundary
    if compound:
        review_reasons.append("compound_account_nature")

    # Missing context is not a lower-confidence prediction; it is an
    # abstention condition.  Finance coding must not guess a natural account
    # when the supplier or line type was not extracted.
    description_available = bool(description)
    llm_eligible = description_available and not compound
    singleton_account_eligible = (
        llm_eligible
        and bool(str(feature.get("vendor_name_norm") or "").strip())
        and str(feature.get("line_type_norm") or "UNKNOWN").upper() != "UNKNOWN"
        and "low_extraction_quality" not in review_reasons
        and "unsupported_tax" not in review_reasons
    )
    segment3_output_eligible = singleton_account_eligible
    blocks_classification = not singleton_account_eligible
    return {
        "decision": "REVIEW_REQUIRED" if review_reasons else "CONTINUE",
        "blocks_classification": blocks_classification,
        "description_available": description_available,
        "llm_eligible": llm_eligible,
        "blocks_llm": not llm_eligible,
        "singleton_account_eligible": singleton_account_eligible,
        "segment3_output_eligible": segment3_output_eligible,
        "review_reasons": review_reasons,
        "quality_status": "sufficient" if not review_reasons else "insufficient",
    }
