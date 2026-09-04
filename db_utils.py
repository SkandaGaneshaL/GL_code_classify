from __future__ import annotations

import array
import os
from pathlib import Path
from typing import Any, Iterable

import oracledb
from dotenv import load_dotenv

try:
    from .config import BASE_DIR, POC_DIR
except ImportError:  # Supports running the files directly from this folder.
    from config import BASE_DIR, POC_DIR


load_dotenv(POC_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)
TABLE_NAME = "AG_GL_ACCOUNT_HISTORY_CASES"


def _resolve(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    return str(path if path.is_absolute() else POC_DIR / path)


def get_db_connection() -> oracledb.Connection:
    """Open an Oracle connection using the existing POC environment pattern."""
    return oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dsn=os.getenv("DB_DSN"),
        config_dir=_resolve(os.getenv("DB_CONFIG_DIR")),
        wallet_location=_resolve(os.getenv("DB_WALLET_LOCATION")),
        wallet_password=os.getenv("DB_WALLET_PASSWORD"),
    )


def _vector_value(value: Iterable[float]) -> array.array:
    vector = list(value)
    if len(vector) != 1024:
        raise ValueError(f"embedding must contain exactly 1024 dimensions, got {len(vector)}")
    return array.array("f", (float(item) for item in vector))


def bulk_upsert_history(rows: Iterable[dict[str, Any]]) -> int:
    """Insert or refresh rows by source_key in one transaction."""
    binds = []
    for row in rows:
        item = dict(row)
        item["embedding"] = _vector_value(item["embedding"])
        binds.append(item)
    if not binds:
        return 0

    sql = f"""
        MERGE INTO {TABLE_NAME} target
        USING (SELECT :source_key AS source_key FROM dual) incoming
           ON (target.SOURCE_KEY = incoming.source_key)
        WHEN MATCHED THEN UPDATE SET
            INVOICE_ID = :invoice_id,
            INVOICE_NUM = :invoice_num,
            INVOICE_DATE = :invoice_date,
            INVOICE_DISTRIBUTION_ID = :invoice_distribution_id,
            INVOICE_LINE_NUMBER = :invoice_line_number,
            DISTRIBUTION_LINE_NUMBER = :distribution_line_number,
            DIST_CODE_COMBINATION_ID = :dist_code_combination_id,
            LINE_DESCRIPTION = :line_description,
            LINE_TYPE = :line_type,
            VENDOR_ID = :vendor_id,
            VENDOR_SITE_ID = :vendor_site_id,
            BUSINESS_UNIT_ID = :business_unit_id,
            LEDGER_ID = :ledger_id,
            CHART_OF_ACCOUNTS_ID = :chart_of_accounts_id,
            ACCOUNT_TYPE = :account_type,
            GL_CODE = :gl_code,
            SEGMENT1 = :segment1,
            SEGMENT2 = :segment2,
            SEGMENT3 = :segment3,
            SEGMENT4 = :segment4,
            SEGMENT5 = :segment5,
            SEGMENT6 = :segment6,
            EMBEDDING = :embedding,
            DATASET_TYPE = :dataset_type,
            IS_SYNTHETIC = :is_synthetic,
            SOURCE_INVOICE_DISTRIBUTION_ID = :source_invoice_distribution_id,
            SYNTHETIC_TYPE = :synthetic_type
        WHEN NOT MATCHED THEN INSERT (
            SOURCE_KEY, INVOICE_ID, INVOICE_NUM, INVOICE_DATE, INVOICE_DISTRIBUTION_ID,
            INVOICE_LINE_NUMBER, DISTRIBUTION_LINE_NUMBER, DIST_CODE_COMBINATION_ID,
            LINE_DESCRIPTION, LINE_TYPE, VENDOR_ID, VENDOR_SITE_ID, BUSINESS_UNIT_ID,
            LEDGER_ID, CHART_OF_ACCOUNTS_ID, ACCOUNT_TYPE,
            GL_CODE, SEGMENT1, SEGMENT2, SEGMENT3, SEGMENT4, SEGMENT5, SEGMENT6,
            EMBEDDING, DATASET_TYPE, IS_SYNTHETIC, SOURCE_INVOICE_DISTRIBUTION_ID,
            SYNTHETIC_TYPE
        ) VALUES (
            :source_key, :invoice_id, :invoice_num, :invoice_date, :invoice_distribution_id,
            :invoice_line_number, :distribution_line_number, :dist_code_combination_id,
            :line_description, :line_type, :vendor_id, :vendor_site_id, :business_unit_id,
            :ledger_id, :chart_of_accounts_id, :account_type,
            :gl_code, :segment1, :segment2, :segment3, :segment4, :segment5, :segment6,
            :embedding, :dataset_type, :is_synthetic, :source_invoice_distribution_id,
            :synthetic_type
        )
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, binds)
        conn.commit()
        return len(binds)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_account_types() -> list[str]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT ACCOUNT_TYPE FROM {TABLE_NAME} ORDER BY ACCOUNT_TYPE")
            return [str(row[0]) for row in cur.fetchall() if row[0]]
    finally:
        conn.close()


def search_similar_history(
    query_embedding: Iterable[float], top_k: int = 5, target_accuracy: int = 95
) -> list[dict[str, Any]]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if not 1 <= target_accuracy <= 100:
        raise ValueError("target_accuracy must be between 1 and 100")

    sql = f"""
        SELECT ID, INVOICE_NUM, INVOICE_DATE, INVOICE_LINE_NUMBER,
               DISTRIBUTION_LINE_NUMBER, LINE_DESCRIPTION, LINE_TYPE,
               ACCOUNT_TYPE, GL_CODE, SEGMENT1, SEGMENT2, SEGMENT3,
               SEGMENT4, SEGMENT5, SEGMENT6,
               VECTOR_DISTANCE(EMBEDDING, :query_vector, COSINE) AS DISTANCE
          FROM {TABLE_NAME}
         ORDER BY VECTOR_DISTANCE(EMBEDDING, :query_vector, COSINE)
         FETCH APPROX FIRST {top_k} ROWS ONLY
         WITH TARGET ACCURACY {target_accuracy}
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"query_vector": _vector_value(query_embedding)})
            rows = cur.fetchall()
        results = []
        for row in rows:
            distance = float(row[15]) if row[15] is not None else None
            results.append(
                {
                    "id": int(row[0]) if row[0] is not None else None,
                    "invoice_num": row[1],
                    "invoice_date": row[2],
                    "invoice_line_number": row[3],
                    "distribution_line_number": row[4],
                    "line_description": row[5],
                    "line_type": row[6],
                    "account_type": row[7],
                    "gl_code": row[8],
                    "segment1": row[9],
                    "segment2": row[10],
                    "segment3": row[11],
                    "segment4": row[12],
                    "segment5": row[13],
                    "segment6": row[14],
                    "cosine_distance": distance,
                    "similarity_score": None if distance is None else max(0.0, min(1.0, 1.0 - distance)),
                }
            )
        return results
    finally:
        conn.close()


def map_account_type_to_gl(account_type: str) -> dict[str, Any] | None:
    """Return the most frequent GL mapping for an LLM-selected account type."""
    if not account_type or account_type == "Unknown":
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT GL_CODE, SEGMENT1, SEGMENT2, SEGMENT3, SEGMENT4, SEGMENT5, SEGMENT6,
                       COUNT(*) AS ROW_COUNT
                  FROM {TABLE_NAME}
                 WHERE ACCOUNT_TYPE = :account_type
                 GROUP BY GL_CODE, SEGMENT1, SEGMENT2, SEGMENT3, SEGMENT4, SEGMENT5, SEGMENT6
                 ORDER BY ROW_COUNT DESC, GL_CODE
                 FETCH FIRST 1 ROW ONLY
                """,
                {"account_type": account_type},
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "gl_code": row[0],
            "segment1": row[1],
            "segment2": row[2],
            "segment3": row[3],
            "segment4": row[4],
            "segment5": row[5],
            "segment6": row[6],
            "mapping_row_count": int(row[7]),
            "mapping_source": "account_type_history",
        }
    finally:
        conn.close()
