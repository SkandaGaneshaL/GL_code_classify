from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dotenv import dotenv_values

try:
    from .config import (
        AP_EXTRACTION_MAX_PAGES,
        AP_EXTRACTION_MODEL_ID,
        AP_EXTRACTION_SPLIT_PASSES,
        AP_INVOICE_ROOT,
        OCI_CONFIG_PATH,
        OCI_PROFILE,
    )
except ImportError:  # Supports running the file directly from this folder.
    from config import (
        AP_EXTRACTION_MAX_PAGES,
        AP_EXTRACTION_MODEL_ID,
        AP_EXTRACTION_SPLIT_PASSES,
        AP_INVOICE_ROOT,
        OCI_CONFIG_PATH,
        OCI_PROFILE,
    )


@dataclass(frozen=True)
class APExtractionResult:
    extracted_json: dict[str, Any]
    page_count: int | None
    diagnostics: dict[str, Any]
    model_id: str


class APInvoiceAdapterError(RuntimeError):
    """Raised when the external AP_INVOICE extraction source cannot be used."""


_AP_PACKAGE_CACHE: dict[str, str] = {}
_AP_ENV_PREFIXES = ("OCI_EXTRACTION_",)
_AP_ENV_KEYS = {
    "OCI_EXC_COMPARTMENT_ID",
    "OCI_EXC_REGION",
    "OCI_EXTRACTION_PROJECT_ID",
    "OCI_CONFIG_FILE",
    "OCI_PROFILE",
}


def _validate_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    app_dir = root / "app"
    if not root.exists():
        raise APInvoiceAdapterError(f"AP_INVOICE_ROOT does not exist: {root}")
    if not root.is_dir() or not (app_dir / "__init__.py").is_file():
        raise APInvoiceAdapterError(f"AP_INVOICE_ROOT is not a valid AP_INVOICE repository: {root}")
    return root


def _load_external_package(root: Path) -> str:
    """Load AP_INVOICE's app package under a collision-resistant alias."""
    cache_key = str(root)
    if cache_key in _AP_PACKAGE_CACHE:
        return _AP_PACKAGE_CACHE[cache_key]
    package_name = f"_ap_invoice_external_{abs(hash(cache_key)):x}"
    package_path = root / "app" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_path,
        submodule_search_locations=[str(root / "app")],
    )
    if spec is None or spec.loader is None:
        raise APInvoiceAdapterError(f"Unable to load AP_INVOICE package from {package_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    _AP_PACKAGE_CACHE[cache_key] = package_name
    return package_name


@contextmanager
def _ap_environment(root: Path) -> Iterator[None]:
    """Temporarily expose only AP extraction configuration to AP_INVOICE."""
    env_path = root / ".env"
    values = dotenv_values(env_path) if env_path.is_file() else {}
    selected: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        if key in _AP_ENV_KEYS or any(key.startswith(prefix) for prefix in _AP_ENV_PREFIXES):
            selected[key] = str(value)
    # AP_INVOICE's OciPdfClient defaults to the process-relative `.oci/config`.
    # Bridge it to the GL application's canonical absolute path for both
    # construction and extraction, without modifying AP_INVOICE's environment.
    selected["OCI_CONFIG_FILE"] = str(OCI_CONFIG_PATH)
    selected["OCI_PROFILE"] = os.getenv("OCI_PROFILE", OCI_PROFILE).strip() or OCI_PROFILE
    if AP_EXTRACTION_MAX_PAGES > 0:
        selected["MAX_PDF_PAGES"] = str(AP_EXTRACTION_MAX_PAGES)
    if AP_EXTRACTION_SPLIT_PASSES:
        selected["OCI_EXTRACTION_SPLIT_PASSES"] = AP_EXTRACTION_SPLIT_PASSES

    previous: dict[str, str | None] = {}
    try:
        for key, value in selected.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _scoped_rules(models_module: Any, *, include_context_fields: bool = False) -> list[Any]:
    RuleRecord = models_module.RuleRecord
    line_rules = [
        "Extract one object for every physical invoice item or service row, preserving invoice order.",
        (
            "Each LineItems object must contain exactly LineDescription, LineType, LineAmount, UnitPrice, "
            "QuantityInvoiced, TaxRate, and TaxAmount."
            if include_context_fields
            else "Each LineItems object must contain exactly LineDescription, LineType, LineAmount, UnitPrice, and QuantityInvoiced."
        ),
        "Extract each value only when explicitly present on the same line; return null for missing or unreadable values.",
        "LineDescription is the product or service description. Normalize LineType only to ITEM, TAX, FREIGHT, or MISC when the invoice explicitly identifies its nature; otherwise return the source line type or null.",
        "Do not calculate LineAmount, UnitPrice, or QuantityInvoiced. Do not create rows for subtotal, total, tax summary, VAT summary, or payment rows.",
        "Every child field must use the public scalar shape {value, Page}; Page is an integer or null.",
        (
            "TaxRate and TaxAmount are optional and must be null when not explicitly aligned to that line."
            if include_context_fields
            else "Do not include PONumber or any field other than the five approved line fields."
        ),
    ]
    rules = [
        RuleRecord(
            ID="segment3-vendor",
            FIELD_KEY="VendorName",
            DISPLAY_LABEL="Supplier Name",
            SHORT_RULE="seller/vendor/supplier name exactly as written",
            DETAILED_RULE=[
                "Extract the seller, vendor, or supplier legal entity name exactly as written.",
                "Do not use the buyer, bill-to, ship-to, contact, address, tax identifier, or purchase-order value.",
            ],
        ),
        RuleRecord(
            ID="segment3-payee",
            FIELD_KEY="PayeeName",
            DISPLAY_LABEL="Payee Name",
            SHORT_RULE="payee or beneficiary name",
            DETAILED_RULE=[
                "Extract Payee, Beneficiary, Remit To, Account Name, or Payable To exactly as written when explicitly present.",
                "Do not substitute a buyer or unrelated account holder.",
            ],
        ),
        RuleRecord(
            ID="segment3-line-items",
            FIELD_KEY="LineItems",
            DISPLAY_LABEL="Invoice Line Items",
            SHORT_RULE="approved five-field line item objects",
            VALUE_TYPE="list",
            DETAILED_RULE=line_rules,
        ),
    ]
    if include_context_fields:
        context_fields = {
            "VendorAddress": "seller or supplier address",
            "BillToAddress": "bill-to entity or address",
            "ShipToAddress": "ship-to entity or address",
            "InvoiceNetAmount": "invoice net amount",
            "TaxAmount": "invoice tax amount",
            "InvoiceGrossAmount": "invoice gross amount",
            "InvoiceCurrency": "invoice currency",
            "PONumber": "purchase order number",
            "ProjectProtocolNumber": "project or protocol reference",
            "InvoicingCountry": "invoice country",
            "BuyerNameOriginal": "buyer name as written",
        }
        for field_key, label in context_fields.items():
            rules.append(
                RuleRecord(
                    ID=f"segment3-context-{field_key.lower()}",
                    FIELD_KEY=field_key,
                    DISPLAY_LABEL=label,
                    SHORT_RULE=f"{label}, only when explicitly present",
                    DETAILED_RULE=[
                        f"Extract {label} exactly as written when explicitly present.",
                        "Return null when absent or unreadable; never infer or calculate it.",
                    ],
                )
            )
    return rules


class APInvoiceExtractionAdapter:
    """Reuse AP_INVOICE's extraction engine with a local six-field contract."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        client: Any = None,
        model_id: str | None = None,
        include_context_fields: bool = False,
    ) -> None:
        self.root = _validate_root(Path(root) if root is not None else AP_INVOICE_ROOT)
        with _ap_environment(self.root):
            package_name = _load_external_package(self.root)
            pdf_module = importlib.import_module(f"{package_name}.pdf_extractor")
            models_module = importlib.import_module(f"{package_name}.models")
            repository_module = importlib.import_module(f"{package_name}.rule_repository")
            self._error_type = pdf_module.PdfExtractionError
            self._extractor = pdf_module.OciPdfExtractor(
                repository_module.InMemoryRuleRepository(
                    _scoped_rules(models_module, include_context_fields=include_context_fields)
                ),
                client=client,
                model_id=model_id or AP_EXTRACTION_MODEL_ID or None,
            )
        self.model_id = str(self._extractor.model_id)

    @property
    def extractor(self) -> Any:
        return self._extractor

    def extract(self, document_bytes: bytes, *, filename: str = "invoice.pdf") -> APExtractionResult:
        safe_filename = Path(filename or "invoice.pdf").name or "invoice.pdf"
        with _ap_environment(self.root):
            try:
                extracted, page_count, diagnostics = self._extractor.extract(
                    document_bytes,
                    filename=safe_filename,
                )
            except self._error_type:
                raise
            except Exception as exc:
                raise APInvoiceAdapterError("AP_INVOICE extraction failed") from exc
        if not isinstance(extracted, dict):
            raise APInvoiceAdapterError("AP_INVOICE returned a non-object extraction result")
        return APExtractionResult(
            extracted_json=extracted,
            page_count=page_count,
            diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
            model_id=self.model_id,
        )
