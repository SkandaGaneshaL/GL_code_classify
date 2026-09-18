from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from . import db_utils
    from .config import BASE_DIR, EMBEDDING_MODEL
    from .embeddings import create_document_embedder, embed_documents, load_oci_settings
    from .feature_pack import build_line_feature
    from .finance_dataset import validate_finance_rows
except ImportError:  # Supports running the file directly from this folder.
    import db_utils
    from config import BASE_DIR, EMBEDDING_MODEL
    from embeddings import create_document_embedder, embed_documents, load_oci_settings
    from feature_pack import build_line_feature
    from finance_dataset import validate_finance_rows


LOGGER = logging.getLogger(__name__)
DEFAULT_EXCEL_PATH = BASE_DIR / "gl_account_history_poc_120.xlsx"
DEFAULT_SHEET_NAME = "POC Dataset"
EXPECTED_COLUMNS = [
    "INVOICE_ID",
    "INVOICE_NUM",
    "INVOICE_DATE",
    "INVOICE_DISTRIBUTION_ID",
    "INVOICE_LINE_NUMBER",
    "DISTRIBUTION_LINE_NUMBER",
    "DIST_CODE_COMBINATION_ID",
    "LINE_DESCRIPTION",
    "LINE_TYPE",
    "VENDOR_ID",
    "VENDOR_SITE_ID",
    "BUSINESS_UNIT_ID",
    "LEDGER_ID",
    "CHART_OF_ACCOUNTS_ID",
    "SEGMENT1",
    "SEGMENT2",
    "SEGMENT3",
    "SEGMENT3_DESCRIPTION",
    "SEGMENT4",
    "SEGMENT5",
    "SEGMENT6",
    "GL_CODE",
    "DATASET_TYPE",
    "IS_SYNTHETIC",
    "SOURCE_INVOICE_DISTRIBUTION_ID",
    "SYNTHETIC_TYPE",
]

# Governed Finance exports may include these columns. They are optional for
# the legacy 120-row POC workbook, but certification loads require them via
# ``validate_finance_rows(..., strict=True)`` before insertion.
FINANCE_OPTIONAL_COLUMNS = [
    "VENDOR_NAME",
    "LEGAL_ENTITY_ID",
    "LINE_AMOUNT",
    "CURRENCY_CODE",
    "UNIT_PRICE",
    "QUANTITY_INVOICED",
    "FINAL_POSTED",
    "NATURAL_ACCOUNT_DESCRIPTION",
    "SOURCE_GROUP_ID",
]


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _integer(value: Any) -> int | None:
    text = _clean(value)
    return int(text) if text else None


def _source_key(row: dict[str, str | None]) -> str:
    columns = [*EXPECTED_COLUMNS, *(column for column in FINANCE_OPTIONAL_COLUMNS if column in row)]
    payload = "\x1f".join(row.get(column) or "" for column in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_batch(
    excel_path: Path,
    batch_number: int,
    batch_size: int,
    sheet_name: str = DEFAULT_SHEET_NAME,
    dataset_type: str | None = "HISTORY",
    strict_finance: bool = False,
) -> list[dict[str, str | None]]:
    if batch_number < 1:
        raise ValueError("batch_number must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not excel_path.is_file():
        raise FileNotFoundError(f"Excel dataset was not found: {excel_path}")

    # Read values as objects and convert them to strings ourselves. This avoids
    # pandas inferring identifiers, codes, or ISO date strings as numeric/date values.
    frame = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        dtype=object,
        keep_default_na=False,
    )
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Excel dataset is missing columns: {missing}")

    if dataset_type is not None:
        requested_type = str(dataset_type).strip().upper()
        frame = frame.loc[frame["DATASET_TYPE"].astype(str).str.strip().str.upper() == requested_type].copy()
        if frame.empty:
            raise ValueError(f"Excel dataset contains no rows with DATASET_TYPE={requested_type!r}")

    start = (batch_number - 1) * batch_size
    end = min(start + batch_size, len(frame))
    if start >= len(frame):
        raise ValueError(
            f"Batch {batch_number} is empty. Dataset has {len(frame)} rows with batch size {batch_size}."
        )

    selected_columns = [*EXPECTED_COLUMNS, *(column for column in FINANCE_OPTIONAL_COLUMNS if column in frame.columns)]
    selected = frame.iloc[start:end][selected_columns]
    rows: list[dict[str, str | None]] = []
    for excel_row_number, (_, values) in enumerate(selected.iterrows(), start=start + 2):
        row = {column: _clean(values[column]) for column in selected_columns}
        if not row["LINE_DESCRIPTION"]:
            raise ValueError(f"Excel row {excel_row_number} has an empty LINE_DESCRIPTION")
        for required in ("INVOICE_NUM", "GL_CODE", "SEGMENT3_DESCRIPTION"):
            if not row[required]:
                raise ValueError(f"Excel row {excel_row_number} has an empty {required}")
        rows.append(row)

    if strict_finance:
        validate_finance_rows(
            [
                {
                    "invoice_distribution_id": row.get("INVOICE_DISTRIBUTION_ID"),
                    "invoice_date": row.get("INVOICE_DATE"),
                    "source_group_id": row.get("SOURCE_GROUP_ID") or row.get("INVOICE_ID") or row.get("INVOICE_DISTRIBUTION_ID"),
                    "vendor_id": row.get("VENDOR_ID"),
                    "vendor_name": row.get("VENDOR_NAME"),
                    "vendor_site_id": row.get("VENDOR_SITE_ID"),
                    "business_unit": row.get("BUSINESS_UNIT_ID"),
                    "legal_entity": row.get("LEGAL_ENTITY_ID"),
                    "ledger": row.get("LEDGER_ID"),
                    "chart_of_accounts": row.get("CHART_OF_ACCOUNTS_ID"),
                    "line_type": row.get("LINE_TYPE"),
                    "line_description": row.get("LINE_DESCRIPTION"),
                    "natural_account_description": row.get("NATURAL_ACCOUNT_DESCRIPTION") or row.get("SEGMENT3_DESCRIPTION"),
                    "line_amount": row.get("LINE_AMOUNT"),
                    "currency": row.get("CURRENCY_CODE"),
                    "segment1": row.get("SEGMENT1"),
                    "segment2": row.get("SEGMENT2"),
                    "segment3": row.get("SEGMENT3"),
                    "segment4": row.get("SEGMENT4"),
                    "segment5": row.get("SEGMENT5"),
                    "segment6": row.get("SEGMENT6"),
                    "final_posted": row.get("FINAL_POSTED"),
                    "source_invoice_distribution_id": row.get("SOURCE_INVOICE_DISTRIBUTION_ID"),
                    "is_synthetic": row.get("IS_SYNTHETIC"),
                }
                for row in rows
            ],
            strict=True,
        )

    LOGGER.info("Selected batch %d: Excel data rows %d-%d", batch_number, start + 1, end)
    return rows


def to_db_row(
    row: dict[str, str | None], embedding: list[float], feature: dict[str, Any]
) -> dict[str, Any]:
    return {
        "source_key": _source_key(row),
        "invoice_id": row["INVOICE_ID"],
        "invoice_num": row["INVOICE_NUM"],
        "invoice_date": row["INVOICE_DATE"],
        "invoice_distribution_id": row["INVOICE_DISTRIBUTION_ID"],
        "invoice_line_number": _integer(row["INVOICE_LINE_NUMBER"]),
        "distribution_line_number": _integer(row["DISTRIBUTION_LINE_NUMBER"]),
        "dist_code_combination_id": row["DIST_CODE_COMBINATION_ID"],
        "line_description": row["LINE_DESCRIPTION"],
        "line_type": row["LINE_TYPE"],
        "vendor_id": row["VENDOR_ID"],
        "vendor_site_id": row["VENDOR_SITE_ID"],
        "business_unit_id": row["BUSINESS_UNIT_ID"],
        "ledger_id": row["LEDGER_ID"],
        "chart_of_accounts_id": row["CHART_OF_ACCOUNTS_ID"],
        "account_type": row["SEGMENT3_DESCRIPTION"],
        "gl_code": row["GL_CODE"],
        "segment1": row["SEGMENT1"],
        "segment2": row["SEGMENT2"],
        "segment3": row["SEGMENT3"],
        "segment4": row["SEGMENT4"],
        "segment5": row["SEGMENT5"],
        "segment6": row["SEGMENT6"],
        "retrieval_text": feature["retrieval_text"],
        "vendor_name_norm": feature["vendor_name_norm"] or None,
        "line_type_norm": feature["line_type_norm"],
        "identity_key": feature["identity_key"],
        "amount_band": feature["amount_band"],
        "line_amount": row.get("LINE_AMOUNT"),
        "unit_price": row.get("UNIT_PRICE"),
        "quantity_invoiced": row.get("QUANTITY_INVOICED"),
        "vendor_name": row.get("VENDOR_NAME"),
        "legal_entity_id": row.get("LEGAL_ENTITY_ID"),
        "currency_code": row.get("CURRENCY_CODE"),
        "final_posted": row.get("FINAL_POSTED"),
        "natural_account_description": row.get("NATURAL_ACCOUNT_DESCRIPTION") or row.get("SEGMENT3_DESCRIPTION"),
        "source_group_id": row.get("SOURCE_GROUP_ID") or row.get("INVOICE_ID") or row.get("INVOICE_DISTRIBUTION_ID"),
        "embedding_model": EMBEDDING_MODEL,
        "dataset_type": row["DATASET_TYPE"],
        "is_synthetic": row["IS_SYNTHETIC"],
        "source_invoice_distribution_id": row["SOURCE_INVOICE_DISTRIBUTION_ID"],
        "synthetic_type": row["SYNTHETIC_TYPE"],
        "embedding": embedding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and load one Excel history batch into Oracle")
    parser.add_argument("batch_number", type=int, help="1-based batch number; batch 1 is rows 1-60 by default")
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument(
        "--dataset-type",
        default="HISTORY",
        help="Only load this dataset split into retrieval history; use 'all' to disable filtering.",
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--compartment-id", default=None)
    parser.add_argument("--service-endpoint", default=None)
    parser.add_argument(
        "--strict-finance",
        action="store_true",
        help="Require the complete Finance-posted distribution contract before loading",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    dataset_type = None if str(args.dataset_type).strip().lower() == "all" else args.dataset_type
    rows = read_batch(
        args.excel_path,
        args.batch_number,
        args.batch_size,
        args.sheet_name,
        dataset_type,
        strict_finance=args.strict_finance,
    )
    settings = load_oci_settings(
        profile=args.profile,
        compartment_id=args.compartment_id,
        service_endpoint=args.service_endpoint,
    )
    embedder = create_document_embedder(settings)
    features = [
        build_line_feature(
            {
                "LineDescription": row["LINE_DESCRIPTION"],
                "LineType": row["LINE_TYPE"],
                "LineAmount": row.get("LINE_AMOUNT"),
                "UnitPrice": row.get("UNIT_PRICE"),
                "QuantityInvoiced": row.get("QUANTITY_INVOICED"),
            },
            vendor=row.get("VENDOR_NAME") or row.get("VENDOR_ID") or "",
            line_index=index,
        )
        for index, row in enumerate(rows)
    ]
    embeddings = embed_documents(features, embedder)
    inserted_or_updated = db_utils.bulk_upsert_history(
        to_db_row(row, embedding, feature)
        for row, embedding, feature in zip(rows, embeddings, features)
    )
    first_row = (args.batch_number - 1) * args.batch_size + 1
    last_row = first_row + len(rows) - 1
    print(
        f"Batch {args.batch_number} completed: rows {first_row}-{last_row}; "
        f"{inserted_or_updated} rows inserted/updated in {db_utils.TABLE_NAME}"
    )


if __name__ == "__main__":
    main()
