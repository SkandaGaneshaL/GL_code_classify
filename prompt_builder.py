from __future__ import annotations

import json
from typing import Any, Iterable


def _historical_evidence(cases: Iterable[dict[str, Any]]) -> str:
    """Serialize precedents without retrieval scores or non-allowlisted fields."""
    evidence = [
        {
            "vendor": case.get("vendor_name_norm") or case.get("vendor_name"),
            "line_type": case.get("line_type_norm") or case.get("line_type"),
            "description": case.get("line_description"),
            "approved_account_type": case.get("account_type"),
        }
        for case in cases
    ]
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def _current_line(feature: dict[str, Any]) -> str:
    header = feature.get("classification_header") or {}
    bounded_header = {
        key: str(header[key])[:240]
        for key in (
            "VendorAddress",
            "BillToAddress",
            "ShipToAddress",
            "InvoiceNetAmount",
            "TaxAmount",
            "InvoiceGrossAmount",
            "InvoiceCurrency",
            "PONumber",
            "ProjectProtocolNumber",
            "InvoicingCountry",
            "BuyerNameOriginal",
        )
        if header.get(key) is not None
    }
    payload = {
        "line_index": feature.get("line_index"),
        "vendor": feature.get("vendor_name_norm") or None,
        "line_type": feature.get("line_type_norm") or "UNKNOWN",
        "description": feature.get("line_description") or "",
        "quantity_invoiced": feature.get("quantity_invoiced"),
        "unit_price": feature.get("unit_price"),
        "line_amount": feature.get("line_amount"),
        "tax_rate": feature.get("tax_rate"),
        "tax_amount": feature.get("tax_amount"),
        "invoice_context": bounded_header,
        "project_reference_present": bool(feature.get("project_reference_present")),
        "invoicing_country": feature.get("invoicing_country") or None,
        "invoice_currency": feature.get("invoice_currency") or None,
        "sibling_descriptions": list(feature.get("sibling_descriptions") or [])[:3],
        "numeric_consistency": feature.get("numeric_consistency"),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_decision_prompt(
    current_line: dict[str, Any],
    historical_cases: list[dict[str, Any]],
    allowed_account_types: list[str],
    evidence_mode: str = "qualified",
) -> str:
    allowed = list(dict.fromkeys([*allowed_account_types, "Unknown"]))
    enum_text = json.dumps(allowed, ensure_ascii=False)
    mode_text = (
        "These are strong retrieval candidates; they are evidence, not proof."
        if evidence_mode == "qualified"
        else "These are weak nearest references; do not force a match."
    )
    return f"""You are an enterprise Accounts Payable natural-account classifier.

Classify the CURRENT invoice line into exactly one allowed account type. Use the line's business meaning, accounting nature, vendor context, line type, and numeric consistency. Historical cases are approved precedents, but do not use retrieval rank, similarity scores, or majority voting as the decision.

Allowed account_type values: {enum_text}

Current line (only approved invoice fields):
{_current_line(current_line)}

Historical cases (one per competing type where available):
{_historical_evidence(historical_cases)}

Evidence policy: {mode_text}

Policy constraints:
- Do not emit GL codes, segment values, POs, invoice IDs, country, or any other payload fields.
- Software, SaaS, subscriptions, operating-system licenses, Linux/RHEL, and unsupported tax natures must be Unknown unless an explicitly allowed approved type genuinely supports that nature.
- Asset Clearing and Miscellaneous require clear accounting evidence; do not use them as generic fallbacks.
- Missing numeric values are unknown, not zero. Use amounts only for consistency, unit-value, capitalization, and bulk-purchase reasoning; they are not currency bands.
- If no allowed type is clearly supported, set account_type to Unknown and inferred_account_type to the best allowed type or null.
- If account_type is not Unknown, inferred_account_type must be null.
- Confidence is accounting certainty from 0.0 to 1.0, never a retrieval score.

Return ONLY a JSON object with exactly these fields:
{{"line_description":"...","account_type":"...","inferred_account_type":null,"reason":"...","confidence":0.0}}
No Markdown or text outside the JSON object.
"""


def _legacy_feature(line_description: str) -> dict[str, Any]:
    return {
        "line_index": 0,
        "vendor_name_norm": None,
        "line_type_norm": "UNKNOWN",
        "line_description": line_description,
        "quantity_invoiced": None,
        "unit_price": None,
        "line_amount": None,
        "sibling_descriptions": [],
    }


def _allowed_from_cases(cases: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(case.get("account_type")) for case in cases if case.get("account_type")))


def build_qualified_cases_prompt(line_description: str, historical_cases: list[dict[str, Any]]) -> str:
    return build_decision_prompt(
        _legacy_feature(line_description),
        historical_cases,
        _allowed_from_cases(historical_cases),
        evidence_mode="qualified",
    )


def build_weak_cases_prompt(line_description: str, historical_cases: list[dict[str, Any]]) -> str:
    return build_decision_prompt(
        _legacy_feature(line_description),
        historical_cases,
        _allowed_from_cases(historical_cases),
        evidence_mode="weak",
    )


def build_account_type_prompt(line_description: str, historical_cases: list[dict[str, Any]]) -> str:
    """Backward-compatible alias for the qualified-evidence prompt."""
    return build_qualified_cases_prompt(line_description, historical_cases)
