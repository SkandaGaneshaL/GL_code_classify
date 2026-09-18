from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CANONICAL_LINE_KEYS = (
    "LineDescription",
    "LineType",
    "LineAmount",
    "UnitPrice",
    "QuantityInvoiced",
)
CLASSIFICATION_HEADER_KEYS = frozenset(
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
CLASSIFICATION_LINE_KEYS = frozenset({"TaxRate", "TaxAmount"})
EXTRACTED_TOP_LEVEL_KEYS = frozenset({"VendorName", "PayeeName", "LineItems", *CLASSIFICATION_HEADER_KEYS})
EXTRACTED_LINE_KEYS = frozenset({*CANONICAL_LINE_KEYS, "LineItemDescription", "QuantityBilled"})
EXTRACTED_LINE_KEYS = frozenset({*EXTRACTED_LINE_KEYS, *CLASSIFICATION_LINE_KEYS})
LEGACY_ALIASES = {
    "LineDescription": "LineItemDescription",
    "QuantityInvoiced": "QuantityBilled",
}
_NULL_TEXT = frozenset({"", "not present", "no value", "n/a", "null"})


class ExtractionProjectionError(ValueError):
    """Raised when AP_INVOICE output cannot be safely projected."""


@dataclass(frozen=True)
class ProjectedInvoice:
    payload: dict[str, Any]
    vendor_source: str | None
    line_count: int
    warnings: tuple[str, ...] = ()
    classification_context: dict[str, Any] | None = None


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def _scalar(value: Any) -> str | None:
    value = _unwrap(value)
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in _NULL_TEXT else text


def _line_value(item: Mapping[str, Any], canonical_key: str) -> str | None:
    value = item.get(canonical_key)
    if value is None and canonical_key in LEGACY_ALIASES:
        value = item.get(LEGACY_ALIASES[canonical_key])
    return _scalar(value)


def project_extracted_payload(extracted: Mapping[str, Any]) -> ProjectedInvoice:
    """Project AP_INVOICE output into the classifier's strict six-field payload."""
    if not isinstance(extracted, Mapping):
        raise ExtractionProjectionError("AP_INVOICE extraction must be a JSON object")
    unknown_top_level = set(extracted) - EXTRACTED_TOP_LEVEL_KEYS
    if unknown_top_level:
        raise ExtractionProjectionError(
            f"Unexpected AP extraction fields: {sorted(unknown_top_level)}"
        )
    if "LineItems" not in extracted:
        raise ExtractionProjectionError("AP_INVOICE extraction did not return LineItems")
    line_items = extracted["LineItems"]
    if not isinstance(line_items, list):
        raise ExtractionProjectionError("AP_INVOICE LineItems must be an array")

    vendor = _scalar(extracted.get("VendorName"))
    payee = _scalar(extracted.get("PayeeName"))
    effective_vendor = vendor or payee
    vendor_source = "VendorName" if vendor else "PayeeName" if payee else None
    projected_lines: list[dict[str, str | None]] = []
    context_lines: list[dict[str, str | None]] = []
    warnings: list[str] = []
    for index, item in enumerate(line_items):
        if not isinstance(item, Mapping):
            raise ExtractionProjectionError(f"LineItems[{index}] must be an object")
        unknown_line_keys = set(item) - EXTRACTED_LINE_KEYS
        if unknown_line_keys:
            raise ExtractionProjectionError(
                f"Unexpected LineItems[{index}] fields: {sorted(unknown_line_keys)}"
            )
        line = {key: _line_value(item, key) for key in CANONICAL_LINE_KEYS}
        context_lines.append(
            {
                "TaxRate": _line_value(item, "TaxRate"),
                "TaxAmount": _line_value(item, "TaxAmount"),
            }
        )
        if not line["LineDescription"]:
            warnings.append(f"LineItems[{index}] has no LineDescription")
        projected_lines.append(line)

    return ProjectedInvoice(
        payload={"VendorName": effective_vendor, "LineItems": projected_lines},
        vendor_source=vendor_source,
        line_count=len(projected_lines),
        warnings=tuple(warnings),
        classification_context={
            "header": {
                key: _scalar(extracted.get(key))
                for key in CLASSIFICATION_HEADER_KEYS
                if _scalar(extracted.get(key)) is not None
            },
            "line_items": context_lines,
        },
    )
