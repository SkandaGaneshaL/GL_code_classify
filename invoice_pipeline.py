from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .ap_invoice_adapter import APInvoiceExtractionAdapter
    from .classification import classify_invoice
    from .config import AP_EXTRACTION_MAX_BYTES, AP_INVOICE_ROOT
    from .embeddings import create_query_embedder, load_oci_settings
    from .extraction_projection import project_extracted_payload
except ImportError:  # Supports running the file directly from this folder.
    from ap_invoice_adapter import APInvoiceExtractionAdapter
    from classification import classify_invoice
    from config import AP_EXTRACTION_MAX_BYTES, AP_INVOICE_ROOT
    from embeddings import create_query_embedder, load_oci_settings
    from extraction_projection import project_extracted_payload


class InvoicePipelineError(RuntimeError):
    def __init__(self, message: str, *, stage: str, code: str):
        super().__init__(message)
        self.stage = stage
        self.code = code


class InvoiceClassificationPipeline:
    """Coordinate AP_INVOICE extraction and the six-field classifier."""

    def __init__(
        self,
        *,
        extractor: Any | None = None,
        embedder: Any | None = None,
        settings: Any | None = None,
        classifier: Callable[..., dict[str, Any]] = classify_invoice,
        classifier_kwargs: Mapping[str, Any] | None = None,
        max_bytes: int | None = None,
        stage_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.extractor = extractor
        self.embedder = embedder
        self.settings = settings
        self.classifier = classifier
        self.classifier_kwargs = dict(classifier_kwargs or {})
        self.max_bytes = max_bytes or AP_EXTRACTION_MAX_BYTES
        self.stage_callback = stage_callback

    def _stage(self, value: str) -> None:
        if self.stage_callback:
            self.stage_callback(value)

    def _ensure_runtime(self) -> None:
        if self.extractor is None:
            self.extractor = APInvoiceExtractionAdapter(
                AP_INVOICE_ROOT,
                include_context_fields=True,
            )
        if self.embedder is None:
            self.settings = self.settings or load_oci_settings()
            self.embedder = create_query_embedder(self.settings)

    def extract_and_classify(self, document_bytes: bytes, *, filename: str = "invoice.pdf") -> dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(document_bytes, (bytes, bytearray)) or not document_bytes.startswith(b"%PDF"):
            raise InvoicePipelineError("Uploaded file is not a valid PDF", stage="validation", code="PDF_REQUIRED")
        if len(document_bytes) > self.max_bytes:
            raise InvoicePipelineError("Uploaded PDF exceeds the configured size limit", stage="validation", code="PDF_TOO_LARGE")
        safe_filename = Path(filename or "invoice.pdf").name or "invoice.pdf"
        self._ensure_runtime()

        extraction_started = time.perf_counter()
        self._stage("extracting")
        try:
            extraction = self.extractor.extract(bytes(document_bytes), filename=safe_filename)
        except Exception as exc:
            if isinstance(exc, InvoicePipelineError):
                raise
            raise InvoicePipelineError("AP_INVOICE extraction failed", stage="extraction", code="EXTRACTION_FAILED") from exc
        extraction_seconds = time.perf_counter() - extraction_started

        projection_started = time.perf_counter()
        self._stage("projecting")
        try:
            projected = project_extracted_payload(extraction.extracted_json)
        except Exception as exc:
            raise InvoicePipelineError("Extracted JSON failed the six-field contract", stage="projection", code="INVALID_EXTRACTED_JSON") from exc
        projection_seconds = time.perf_counter() - projection_started

        classification_seconds = 0.0
        if projected.line_count:
            classification_started = time.perf_counter()
            self._stage("classifying")
            try:
                classifier_options = dict(self.classifier_kwargs)
                classifier_options.setdefault("stage_callback", self.stage_callback)
                classification = self.classifier(
                    projected.payload,
                    self.embedder,
                    settings=self.settings,
                    classification_context=projected.classification_context,
                    **classifier_options,
                )
            except Exception as exc:
                raise InvoicePipelineError("Segment 3 classification failed", stage="classification", code="CLASSIFICATION_FAILED") from exc
            classification_seconds = time.perf_counter() - classification_started
            lines = classification.get("lines", [])
            classifier_version = classification.get("classifier_version")
        else:
            classification = {
                "line_count": 0,
                "lines": [],
                "classifier_version": None,
                "classifier_capability": None,
                "classification_usage": {
                    "calls": 0,
                    "input_tokens": None,
                    "output_tokens": None,
                    "cached_tokens": None,
                    "reasoning_tokens": None,
                    "total_tokens": None,
                    "reported_calls": 0,
                    "unknown_calls": 0,
                    "missing_categories": {},
                    "reasoning_tokens_reported": False,
                    "reasoning_tokens_status": "not_applicable",
                    "output_tokens_semantics": "provider_reported_output_total",
                    "total_latency_ms": 0.0,
                    "model": None,
                    "route": None,
                    "request_id": None,
                    "finish_reason": None,
                    "attempt": 0,
                    "latency": 0.0,
                },
                "approved_fields": [
                    "LineDescription",
                    "LineType",
                    "VendorName/PayeeName",
                    "LineAmount",
                    "UnitPrice",
                    "QuantityInvoiced",
                ],
            }
            lines = []
            classifier_version = None
        self._stage("completed")

        return {
            "status": "completed" if projected.line_count else "NO_LINE_ITEMS",
            "filename": safe_filename,
            "page_count": extraction.page_count,
            "extraction_model": extraction.model_id,
            "extraction_diagnostics": extraction.diagnostics,
            "sanitized_extracted_json": projected.payload,
            "projection_diagnostics": {
                "vendor_source": projected.vendor_source,
                "line_count": projected.line_count,
                "warnings": list(projected.warnings),
            },
            "line_count": len(lines),
            "lines": lines,
            "classifier_version": classifier_version,
            "classifier_capability": classification.get("classifier_capability"),
            "classification_usage": classification.get("classification_usage") or {},
            "timing_seconds": {
                "extraction": extraction_seconds,
                "projection": projection_seconds,
                "classification": classification_seconds,
                "total": time.perf_counter() - started,
            },
        }
