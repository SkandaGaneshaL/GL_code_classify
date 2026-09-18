from decimal import Decimal

import pytest

from feature_pack import build_invoice_features, build_line_feature, normalize_line_type, normalize_vendor_name


def test_feature_pack_uses_vendor_fallback_and_six_fields_only():
    payload = {
        "PayeeName": "Acme, Inc.",
        "LineItems": [
            {
                "LineDescription": "  RHEL 9 license SKU-42  ",
                "LineType": "goods",
                "LineAmount": "1,200.50",
                "UnitPrice": "600.25",
                "QuantityInvoiced": "2",
            }
        ],
    }
    feature = build_invoice_features(payload)[0]
    assert feature["vendor_name_norm"] == "ACME,"
    assert feature["line_type_norm"] == "ITEM"
    assert feature["line_amount"] == "1200.50"
    assert feature["quantity_invoiced"] == "2"
    assert "rhel" in feature["retrieval_text"].lower()
    assert "LineAmount" not in feature["retrieval_text"]


def test_non_allowlisted_payload_is_rejected():
    with pytest.raises(ValueError, match="Unsupported invoice payload keys"):
        build_invoice_features({"PONumber": "PO-1", "LineItems": []})


def test_line_fields_are_rejected_when_extra_field_is_present():
    with pytest.raises(ValueError, match="Unsupported line payload keys"):
        build_line_feature({"LineDescription": "paper", "VendorName": "bad"}, "Acme", 0)


def test_normalization_and_null_preservation():
    assert normalize_vendor_name("  Globex   Pvt. Ltd. ") == "GLOBEX"
    assert normalize_line_type("VAT") == "TAX"
    feature = build_line_feature({"LineDescription": "paper"}, "Acme", 0)
    assert feature["line_amount"] is None
    assert feature["unit_price"] is None
    assert feature["quantity_invoiced"] is None


def test_internal_context_adds_project_and_tax_signals_without_changing_public_payload():
    feature = build_invoice_features(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "consulting"}]},
        classification_context={
            "header": {
                "PONumber": "PO-42",
                "InvoicingCountry": "IN",
                "BankAccountNumber": "must-not-be-used",
            },
            "line_items": [{"TaxRate": "18", "TaxAmount": "18.00"}],
        },
    )[0]
    assert feature["project_reference_present"] is True
    assert feature["invoicing_country"] == "IN"
    assert feature["tax_rate"] == "18"
    assert "BankAccountNumber" not in feature["classification_header"]
