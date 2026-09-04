from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from . import db_utils
    from .config import BASE_DIR
    from .embeddings import create_document_embedder, embed_documents, load_oci_settings
except ImportError:  # Supports running the file directly from this folder.
    import db_utils
    from config import BASE_DIR
    from embeddings import create_document_embedder, embed_documents, load_oci_settings


LOGGER = logging.getLogger(__name__)
DEFAULT_EXCEL_PATH = BASE_DIR / "gl_account_history_poc_120.xlsx"
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


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _integer(value: Any) -> int | None:
    text = _clean(value)
    return int(text) if text else None


def _source_key(row: dict[str, str | None]) -> str:
    payload = "\x1f".join(row.get(column) or "" for column in EXPECTED_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_batch(excel_path: Path, batch_number: int, batch_size: int) -> list[dict[str, str | None]]:
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
        sheet_name=0,
        dtype=object,
        keep_default_na=False,
    )
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Excel dataset is missing columns: {missing}")

    start = (batch_number - 1) * batch_size
    end = min(start + batch_size, len(frame))
    if start >= len(frame):
        raise ValueError(
            f"Batch {batch_number} is empty. Dataset has {len(frame)} rows with batch size {batch_size}."
        )

    selected = frame.iloc[start:end][EXPECTED_COLUMNS]
    rows: list[dict[str, str | None]] = []
    for excel_row_number, (_, values) in enumerate(selected.iterrows(), start=start + 2):
        row = {column: _clean(values[column]) for column in EXPECTED_COLUMNS}
        if not row["LINE_DESCRIPTION"]:
            raise ValueError(f"Excel row {excel_row_number} has an empty LINE_DESCRIPTION")
        for required in ("INVOICE_NUM", "GL_CODE", "SEGMENT3_DESCRIPTION"):
            if not row[required]:
                raise ValueError(f"Excel row {excel_row_number} has an empty {required}")
        rows.append(row)

    LOGGER.info("Selected batch %d: Excel data rows %d-%d", batch_number, start + 1, end)
    return rows


def to_db_row(row: dict[str, str | None], embedding: list[float]) -> dict[str, Any]:
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
    parser.add_argument("--profile", default=None)
    parser.add_argument("--compartment-id", default=None)
    parser.add_argument("--service-endpoint", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    rows = read_batch(args.excel_path, args.batch_number, args.batch_size)
    settings = load_oci_settings(
        profile=args.profile,
        compartment_id=args.compartment_id,
        service_endpoint=args.service_endpoint,
    )
    embedder = create_document_embedder(settings)
    embeddings = embed_documents(rows, embedder)
    inserted_or_updated = db_utils.bulk_upsert_history(
        to_db_row(row, embedding)
        for row, embedding in zip(rows, embeddings)
    )
    first_row = (args.batch_number - 1) * args.batch_size + 1
    last_row = first_row + len(rows) - 1
    print(
        f"Batch {args.batch_number} completed: rows {first_row}-{last_row}; "
        f"{inserted_or_updated} rows inserted/updated in {db_utils.TABLE_NAME}"
    )


if __name__ == "__main__":
    main()

