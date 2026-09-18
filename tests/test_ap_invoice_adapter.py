from pathlib import Path

import pytest

from ap_invoice_adapter import APInvoiceAdapterError, APInvoiceExtractionAdapter


AP_ROOT = Path(r"C:\Users\Skanda Ganesha L\Downloads\AP_INVOICE")


def test_missing_ap_root_fails_closed(tmp_path):
    with pytest.raises(APInvoiceAdapterError, match="does not exist"):
        APInvoiceExtractionAdapter(tmp_path / "missing")


class FakeClient:
    class Response:
        text = """{
          "VendorName": {"value": "Example Vendor", "Page": 1},
          "PayeeName": {"value": null, "Page": null},
          "LineItems": [{
            "LineDescription": {"value": "Paper", "Page": 1},
            "LineType": {"value": "ITEM", "Page": 1},
            "LineAmount": {"value": "12.00", "Page": 1},
            "UnitPrice": {"value": "6.00", "Page": 1},
            "QuantityInvoiced": {"value": "2", "Page": 1}
          }]
        }"""
        finish_reason = "STOP"
        request_id = "fake-request"
        usage = None

    def extract(self, document_bytes, *, filename, prompt):
        assert document_bytes.startswith(b"%PDF")
        assert filename == "invoice.pdf"
        assert "LineDescription" in prompt and "QuantityInvoiced" in prompt
        assert "Do not include PONumber" in prompt
        return self.Response()


def test_adapter_uses_isolated_ap_engine_and_scoped_rules():
    adapter = APInvoiceExtractionAdapter(AP_ROOT, client=FakeClient(), model_id="test-model")
    result = adapter.extract(b"%PDF-1.7 test", filename=r"C:\secret\invoice.pdf")
    assert result.extracted_json["VendorName"]["value"] == "Example Vendor"
    assert result.page_count in {None, 1}
    assert result.model_id == "test-model"
