from __future__ import annotations

import hashlib
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


ALLOWED_INVOICE_KEYS = frozenset({"VendorName", "PayeeName", "LineItems"})
ALLOWED_LINE_KEYS = frozenset(
    {"LineDescription", "LineType", "LineAmount", "UnitPrice", "QuantityInvoiced"}
)
ALLOWED_CONTEXT_HEADER_KEYS = frozenset(
    {
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
    }
)
ALLOWED_CONTEXT_LINE_KEYS = frozenset({"TaxRate", "TaxAmount"})

_LEGAL_SUFFIXES = re.compile(
    r"\b(?:PVT\.?\s+LTD\.?|PRIVATE\s+LIMITED|LIMITED|LTD\.?|INC\.?|LLC|CO\.?|CORP(?:ORATION)?\.?)\b",
    re.IGNORECASE,
)
_SOFTWARE_TERMS = re.compile(
    r"\b(?:software|saas|subscription|license|licence|linux|windows|rhel|enterprise\s+linux|os)\b",
    re.IGNORECASE,
)
_HARDWARE_TERMS = re.compile(
    r"\b(?:chassis|rack\s+server|dimm|workstation|monitor|printer|desktop|laptop|keyboard|hardware)\b",
    re.IGNORECASE,
)


def _scalar(value: Any) -> Any:
    """Extract common IDP scalar wrappers without accepting arbitrary fields."""
    if isinstance(value, Mapping):
        for key in ("value", "text", "content", "rawValue"):
            if key in value:
                return value[key]
    return value


def _text(value: Any) -> str:
    value = _scalar(value)
    if value is None:
        return ""
    return str(value).strip()


def normalize_vendor_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).upper()
    text = _LEGAL_SUFFIXES.sub(" ", text)
    text = re.sub(r"\s*[,.]+\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_description(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value))
    text = text.replace("\ufffd", "-")
    return re.sub(r"\s+", " ", text).strip()


def normalize_line_type(value: Any) -> str:
    text = normalize_description(value).upper()
    if text in {"ITEM", "ITEMS", "GOODS", "PRODUCT", "PRODUCTS"}:
        return "ITEM"
    if text in {"TAX", "GST", "IGST", "CGST", "SGST", "VAT", "WHT", "WITHHOLDING"}:
        return "TAX"
    if text in {"FREIGHT", "SHIPPING", "DELIVERY"}:
        return "FREIGHT"
    if text in {"MISC", "MISCELLANEOUS"}:
        return "MISC"
    return text or "UNKNOWN"


def parse_decimal(value: Any) -> Decimal | None:
    value = _scalar(value)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Numeric invoice value is invalid: {value!r}") from exc


def decimal_to_json(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def effective_vendor(payload: Mapping[str, Any]) -> str:
    return _text(payload.get("VendorName")) or _text(payload.get("PayeeName"))


def amount_band(line_amount: Decimal | None) -> str:
    """Return a non-currency band only when a deployment later configures one."""
    del line_amount
    return "UNKNOWN"


def is_software_like(description: str) -> bool:
    return bool(_SOFTWARE_TERMS.search(description)) and not bool(_HARDWARE_TERMS.search(description))


def is_hardware_like(description: str) -> bool:
    return bool(_HARDWARE_TERMS.search(description))


def numeric_consistency(line_amount: Decimal | None, unit_price: Decimal | None, quantity: Decimal | None) -> dict[str, Any]:
    """Use numeric fields for consistency only; never infer currency bands."""
    if line_amount is None or unit_price is None or quantity is None:
        return {"available": False, "consistent": None, "expected_amount": None, "bulk_purchase": False}
    expected = unit_price * quantity
    tolerance = max(Decimal("0.01"), abs(line_amount) * Decimal("0.01"))
    return {
        "available": True,
        "consistent": abs(line_amount - expected) <= tolerance,
        "expected_amount": decimal_to_json(expected),
        "bulk_purchase": quantity >= Decimal("10"),
    }


def lexical_aliases(description: str) -> list[str]:
    aliases = [description]
    lower = description.lower()
    if "rhel" in lower or "enterprise linux" in lower or "blue hat" in lower:
        aliases.extend(["RHEL", "Red Hat Enterprise Linux", "Blue Hat Enterprise Linux", "software license"])
    return list(dict.fromkeys(aliases))


def _identity_key(vendor_norm: str, description_norm: str, line_type_norm: str) -> str:
    raw = "|".join((vendor_norm, description_norm.upper(), line_type_norm))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_line_feature(
    line: Mapping[str, Any],
    vendor: str,
    line_index: int,
    sibling_descriptions: list[str] | None = None,
    header_context: Mapping[str, Any] | None = None,
    line_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unknown = set(line) - ALLOWED_LINE_KEYS
    if unknown:
        raise ValueError(f"Unsupported line payload keys: {sorted(unknown)}")
    description = normalize_description(line.get("LineDescription"))
    vendor_norm = normalize_vendor_name(vendor)
    line_type_norm = normalize_line_type(line.get("LineType"))
    line_amount = parse_decimal(line.get("LineAmount"))
    unit_price = parse_decimal(line.get("UnitPrice"))
    quantity = parse_decimal(line.get("QuantityInvoiced"))
    numeric = numeric_consistency(line_amount, unit_price, quantity)
    header_context = {
        key: _scalar(value)
        for key, value in (header_context or {}).items()
        if key in ALLOWED_CONTEXT_HEADER_KEYS and _scalar(value) is not None
    }
    line_context = {
        key: _scalar(value)
        for key, value in (line_context or {}).items()
        if key in ALLOWED_CONTEXT_LINE_KEYS and _scalar(value) is not None
    }
    project_reference = header_context.get("ProjectProtocolNumber") or header_context.get("PONumber")
    country = normalize_description(header_context.get("InvoicingCountry"))
    currency = normalize_description(header_context.get("InvoiceCurrency"))
    return {
        "line_index": line_index,
        "vendor_name": _text(vendor),
        "vendor_name_norm": vendor_norm,
        "line_type": _text(line.get("LineType")),
        "line_type_norm": line_type_norm,
        "line_description": description,
        "line_description_norm": description.upper(),
        "line_amount": decimal_to_json(line_amount),
        "unit_price": decimal_to_json(unit_price),
        "quantity_invoiced": decimal_to_json(quantity),
        "numeric_consistency": numeric,
        "tax_rate": line_context.get("TaxRate"),
        "tax_amount": line_context.get("TaxAmount"),
        "classification_header": header_context,
        "project_reference_present": bool(project_reference),
        "invoicing_country": country,
        "invoice_currency": currency,
        "amount_band": amount_band(line_amount),
        "retrieval_text": (
            f"vendor: {vendor_norm or 'UNKNOWN'}\n"
            f"line_type: {line_type_norm}\n"
            f"description: {description}\n"
            f"country: {country or 'UNKNOWN'}\n"
            f"project_reference: {'PRESENT' if project_reference else 'ABSENT'}"
        ),
        "identity_key": _identity_key(vendor_norm, description, line_type_norm),
        "is_software_like": is_software_like(description),
        "is_hardware_like": is_hardware_like(description),
        "sibling_descriptions": [
            normalize_description(value) for value in (sibling_descriptions or []) if normalize_description(value)
        ][:3],
    }


def build_invoice_features(
    payload: Mapping[str, Any],
    classification_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    unknown = set(payload) - ALLOWED_INVOICE_KEYS
    if unknown:
        raise ValueError(f"Unsupported invoice payload keys: {sorted(unknown)}")
    lines = payload.get("LineItems")
    if not isinstance(lines, list):
        raise ValueError("Invoice payload must contain a LineItems array")
    vendor = effective_vendor(payload)
    classification_context = classification_context or {}
    header_context = classification_context.get("header") or {}
    context_lines = classification_context.get("line_items") or []
    descriptions = [normalize_description(line.get("LineDescription")) for line in lines if isinstance(line, Mapping)]
    features = []
    for index, line in enumerate(lines):
        if not isinstance(line, Mapping):
            raise ValueError(f"LineItems[{index}] must be an object")
        siblings = [value for position, value in enumerate(descriptions) if position != index]
        line_context = context_lines[index] if index < len(context_lines) and isinstance(context_lines[index], Mapping) else {}
        features.append(
            build_line_feature(
                line,
                vendor,
                index,
                siblings,
                header_context=header_context,
                line_context=line_context,
            )
        )
    return features


def build_oracle_text_query(feature: Mapping[str, Any]) -> str:
    """Create a safe word query; never pass raw punctuation/operators to CONTAINS."""
    tokens = re.findall(r"[\w]+", str(feature.get("line_description") or ""), flags=re.UNICODE)
    vendor_tokens = re.findall(r"[\w]+", str(feature.get("vendor_name_norm") or ""), flags=re.UNICODE)
    reserved = {"AND", "OR", "NOT", "NEAR", "ACCUM", "MINUS", "WITHIN", "ABOUT", "SYN"}
    tokens = [token for token in [*vendor_tokens, *tokens] if token.upper() not in reserved]
    tokens = list(dict.fromkeys(tokens))
    return " OR ".join(tokens[:64]) or "NO_MATCH"
