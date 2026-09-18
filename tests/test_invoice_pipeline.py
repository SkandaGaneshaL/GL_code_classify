import pytest

from ap_invoice_adapter import APExtractionResult
from invoice_pipeline import InvoiceClassificationPipeline, InvoicePipelineError


def _extraction(payload):
    return APExtractionResult(
        extracted_json=payload,
        page_count=1,
        diagnostics={"attempts": 1},
        model_id="extract-test",
    )


def test_invalid_pdf_is_rejected_before_extractor_call():
    class ExplodingExtractor:
        def extract(self, *args, **kwargs):
            raise AssertionError("must not be called")

    with pytest.raises(InvoicePipelineError, match="valid PDF"):
        InvoiceClassificationPipeline(extractor=ExplodingExtractor()).extract_and_classify(b"not-pdf", filename="x.pdf")


def test_pdf_size_limit_is_enforced_before_extractor_call():
    class ExplodingExtractor:
        def extract(self, *args, **kwargs):
            raise AssertionError("must not be called")

    with pytest.raises(InvoicePipelineError, match="size limit"):
        InvoiceClassificationPipeline(extractor=ExplodingExtractor(), max_bytes=4).extract_and_classify(
            b"%PDF-1.7", filename="x.pdf"
        )


def test_only_projected_json_reaches_classifier_and_empty_lines_skip_classifier():
    captured = []

    def fake_classifier(payload, embedder, **kwargs):
        captured.append(payload)
        return {"line_count": 1, "lines": [{"line_index": 0}], "classifier_version": "test"}

    extractor = type(
        "Extractor",
        (),
        {
            "extract": lambda self, document_bytes, *, filename: _extraction(
                {
                    "VendorName": {"value": "Vendor", "Page": 1},
                    "PayeeName": {"value": "Payee", "Page": 1},
                    "LineItems": [
                        {
                            "LineItemDescription": {"value": "Paper", "Page": 1},
                            "QuantityBilled": {"value": "2", "Page": 1},
                            "LineType": {"value": "ITEM", "Page": 1},
                        }
                    ],
                }
            )
        },
    )()
    result = InvoiceClassificationPipeline(extractor=extractor, embedder=object(), classifier=fake_classifier).extract_and_classify(
        b"%PDF-1.7", filename="invoice.pdf"
    )
    assert captured == [
        {
            "VendorName": "Vendor",
            "LineItems": [
                {
                    "LineDescription": "Paper",
                    "LineType": "ITEM",
                    "LineAmount": None,
                    "UnitPrice": None,
                    "QuantityInvoiced": "2",
                }
            ],
        }
    ]
    assert "PONumber" not in result["sanitized_extracted_json"]


def test_empty_line_items_return_warning_without_classification():
    calls = []
    extractor = type("Extractor", (), {"extract": lambda self, *args, **kwargs: _extraction({"LineItems": []})})()
    result = InvoiceClassificationPipeline(
        extractor=extractor,
        embedder=object(),
        classifier=lambda *args, **kwargs: calls.append(True),
    ).extract_and_classify(b"%PDF-1.7", filename="invoice.pdf")
    assert result["status"] == "NO_LINE_ITEMS"
    assert calls == []
