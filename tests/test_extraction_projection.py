import pytest

from extraction_projection import project_extracted_payload


def _line(**overrides):
    values = {
        "LineDescription": {"value": "Network printer", "Page": 1},
        "LineType": {"value": "ITEM", "Page": 1},
        "LineAmount": {"value": "1250.00", "Page": 1},
        "UnitPrice": {"value": "625.00", "Page": 1},
        "QuantityInvoiced": {"value": "2", "Page": 1},
    }
    values.update(overrides)
    return values


def test_projection_drops_scalar_metadata_and_uses_canonical_fields():
    result = project_extracted_payload(
        {
            "VendorName": {"value": "Example Vendor", "Page": 1, "evidence": "hidden"},
            "PayeeName": {"value": "Other Vendor", "Page": 1},
            "LineItems": [_line()],
        }
    )

    assert result.payload == {
        "VendorName": "Example Vendor",
        "LineItems": [
            {
                "LineDescription": "Network printer",
                "LineType": "ITEM",
                "LineAmount": "1250.00",
                "UnitPrice": "625.00",
                "QuantityInvoiced": "2",
            }
        ],
    }


def test_vendor_name_precedes_payee_and_payee_is_fallback():
    preferred = project_extracted_payload(
        {"VendorName": {"value": "Vendor"}, "PayeeName": {"value": "Payee"}, "LineItems": []}
    )
    fallback = project_extracted_payload({"VendorName": {"value": None}, "PayeeName": {"value": "Payee"}, "LineItems": []})
    assert preferred.payload["VendorName"] == "Vendor"
    assert fallback.payload["VendorName"] == "Payee"
    assert fallback.vendor_source == "PayeeName"


def test_legacy_aliases_are_projected_and_missing_numbers_remain_null():
    result = project_extracted_payload(
        {
            "VendorName": {"value": "Vendor"},
            "LineItems": [
                {
                    "LineItemDescription": {"value": "Paper"},
                    "QuantityBilled": {"value": "10"},
                    "LineType": None,
                    "LineAmount": None,
                    "UnitPrice": None,
                }
            ],
        }
    )
    assert result.payload["LineItems"][0] == {
        "LineDescription": "Paper",
        "LineType": None,
        "LineAmount": None,
        "UnitPrice": None,
        "QuantityInvoiced": "10",
    }


def test_context_fields_are_captured_but_not_publicly_projected():
    result = project_extracted_payload(
        {"PONumber": {"value": "PO-1"}, "LineItems": []}
    )
    assert result.classification_context["header"]["PONumber"] == "PO-1"
    assert "PONumber" not in result.payload


def test_unapproved_line_fields_are_rejected():
    with pytest.raises(ValueError, match="Unexpected LineItems"):
        project_extracted_payload({"VendorName": "V", "LineItems": [{"PONumber": "PO-1"}]})
