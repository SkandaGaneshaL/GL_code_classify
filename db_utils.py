from __future__ import annotations

import array
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable

import oracledb
from dotenv import load_dotenv

try:
    from .config import BASE_DIR, EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIMENSIONS, POC_DIR
    from .feature_pack import build_oracle_text_query
    from .retrieval import reciprocal_rank_fusion
except ImportError:  # Supports running the files directly from this folder.
    from config import BASE_DIR, EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIMENSIONS, POC_DIR
    from feature_pack import build_oracle_text_query
    from retrieval import reciprocal_rank_fusion


load_dotenv(POC_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)
TABLE_NAME = "AG_GL_ACCOUNT_HISTORY_CASES"
OVERRIDE_TABLE_NAME = "AG_GL_ACCOUNT_CLASSIFICATION_OVERRIDES"


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
    if len(vector) != EMBEDDING_OUTPUT_DIMENSIONS:
        raise ValueError(
            f"embedding must contain exactly {EMBEDDING_OUTPUT_DIMENSIONS} dimensions, got {len(vector)}"
        )
    return array.array("f", (float(item) for item in vector))


def bulk_upsert_history(rows: Iterable[dict[str, Any]]) -> int:
    """Insert or refresh history rows by source_key in one transaction."""
    binds = []
    for row in rows:
        item = dict(row)
        for key in (
            "invoice_id",
            "invoice_distribution_id",
            "dist_code_combination_id",
            "vendor_id",
            "vendor_site_id",
            "business_unit_id",
            "ledger_id",
            "chart_of_accounts_id",
            "vendor_name_norm",
            "line_type_norm",
            "identity_key",
            "amount_band",
            "line_amount",
            "unit_price",
            "quantity_invoiced",
        ):
            item.setdefault(key, None)
        item.setdefault("embedding_model", EMBEDDING_MODEL)
        item.setdefault("dataset_type", "HISTORY")
        item.setdefault("is_synthetic", "N")
        item.setdefault("source_invoice_distribution_id", None)
        item.setdefault("synthetic_type", None)
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
            RETRIEVAL_TEXT = :retrieval_text,
            VENDOR_NAME_NORM = :vendor_name_norm,
            LINE_TYPE_NORM = :line_type_norm,
            IDENTITY_KEY = :identity_key,
            AMOUNT_BAND = :amount_band,
            LINE_AMOUNT = :line_amount,
            UNIT_PRICE = :unit_price,
            QUANTITY_INVOICED = :quantity_invoiced,
            EMBEDDING_MODEL = :embedding_model,
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
            RETRIEVAL_TEXT, VENDOR_NAME_NORM, LINE_TYPE_NORM, IDENTITY_KEY, AMOUNT_BAND,
            LINE_AMOUNT, UNIT_PRICE, QUANTITY_INVOICED, EMBEDDING_MODEL, EMBEDDING,
            DATASET_TYPE, IS_SYNTHETIC, SOURCE_INVOICE_DISTRIBUTION_ID, SYNTHETIC_TYPE
        ) VALUES (
            :source_key, :invoice_id, :invoice_num, :invoice_date, :invoice_distribution_id,
            :invoice_line_number, :distribution_line_number, :dist_code_combination_id,
            :line_description, :line_type, :vendor_id, :vendor_site_id, :business_unit_id,
            :ledger_id, :chart_of_accounts_id, :account_type,
            :gl_code, :segment1, :segment2, :segment3, :segment4, :segment5, :segment6,
            :retrieval_text, :vendor_name_norm, :line_type_norm, :identity_key, :amount_band,
            :line_amount, :unit_price, :quantity_invoiced, :embedding_model, :embedding,
            :dataset_type, :is_synthetic, :source_invoice_distribution_id, :synthetic_type
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


def _case_from_row(row: tuple[Any, ...], distance_index: int | None = None) -> dict[str, Any]:
    distance = None
    if distance_index is not None and len(row) > distance_index and row[distance_index] is not None:
        distance = float(row[distance_index])
    return {
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
        "vendor_name_norm": row[16] if len(row) > 16 else None,
        "line_type_norm": row[17] if len(row) > 17 else None,
        "is_synthetic": row[18] if len(row) > 18 else None,
    }


def _history_filters(
    *,
    dataset_types: Iterable[str] | None,
    exclude_ids: Iterable[int | str] | None,
    exclude_invoice_distribution_ids: Iterable[str] | None,
    exclude_source_invoice_distribution_ids: Iterable[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Build bind-only filters that keep evaluation data out of retrieval."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if dataset_types is not None:
        normalized_types = list(dict.fromkeys(str(value).strip().upper() for value in dataset_types if str(value).strip()))
        if normalized_types:
            placeholders = []
            for index, value in enumerate(normalized_types):
                name = f"dataset_type_{index}"
                placeholders.append(f":{name}")
                params[name] = value
            # Legacy rows with no DATASET_TYPE are treated as production history.
            clauses.append(f"NVL(UPPER(DATASET_TYPE), 'HISTORY') IN ({', '.join(placeholders)})")
    for prefix, column, values in (
        ("exclude_id", "ID", exclude_ids),
        ("exclude_distribution", "INVOICE_DISTRIBUTION_ID", exclude_invoice_distribution_ids),
        ("exclude_source_distribution", "SOURCE_INVOICE_DISTRIBUTION_ID", exclude_source_invoice_distribution_ids),
    ):
        normalized_values = list(dict.fromkeys(value for value in (values or []) if value is not None and str(value).strip()))
        if not normalized_values:
            continue
        placeholders = []
        for index, value in enumerate(normalized_values):
            name = f"{prefix}_{index}"
            placeholders.append(f":{name}")
            params[name] = value
        clauses.append(f"({column} IS NULL OR {column} NOT IN ({', '.join(placeholders)}))")
    return clauses, params


def search_dense_history(
    query_embedding: Iterable[float],
    top_k: int = 15,
    target_accuracy: int = 95,
    line_type_norm: str | None = None,
    dataset_types: Iterable[str] | None = ("HISTORY",),
    exclude_ids: Iterable[int | str] | None = None,
    exclude_invoice_distribution_ids: Iterable[str] | None = None,
    exclude_source_invoice_distribution_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if not 1 <= target_accuracy <= 100:
        raise ValueError("target_accuracy must be between 1 and 100")

    filter_clauses, filter_params = _history_filters(
        dataset_types=dataset_types,
        exclude_ids=exclude_ids,
        exclude_invoice_distribution_ids=exclude_invoice_distribution_ids,
        exclude_source_invoice_distribution_ids=exclude_source_invoice_distribution_ids,
    )
    where_clauses = ["EMBEDDING IS NOT NULL", *filter_clauses]
    sql = f"""
        SELECT ID, INVOICE_NUM, INVOICE_DATE, INVOICE_LINE_NUMBER,
               DISTRIBUTION_LINE_NUMBER, LINE_DESCRIPTION, LINE_TYPE,
               ACCOUNT_TYPE, GL_CODE, SEGMENT1, SEGMENT2, SEGMENT3,
               SEGMENT4, SEGMENT5, SEGMENT6,
               VECTOR_DISTANCE(EMBEDDING, :query_vector, COSINE) AS DISTANCE,
               VENDOR_NAME_NORM, LINE_TYPE_NORM, IS_SYNTHETIC
          FROM {TABLE_NAME}
         WHERE {' AND '.join(where_clauses)}
         ORDER BY VECTOR_DISTANCE(EMBEDDING, :query_vector, COSINE)
         FETCH APPROX FIRST {top_k} ROWS ONLY
         WITH TARGET ACCURACY {target_accuracy}
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            params: dict[str, Any] = {"query_vector": _vector_value(query_embedding), **filter_params}
            cur.execute(sql, params)
            return [_case_from_row(row, 15) for row in cur.fetchall()]
    finally:
        conn.close()


def search_similar_history(
    query_embedding: Iterable[float],
    top_k: int = 5,
    target_accuracy: int = 95,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Backward-compatible dense retrieval wrapper."""
    return search_dense_history(query_embedding, top_k=top_k, target_accuracy=target_accuracy, **filters)


def search_sparse_history(
    feature: dict[str, Any],
    top_k: int = 15,
    line_type_norm: str | None = None,
    dataset_types: Iterable[str] | None = ("HISTORY",),
    exclude_ids: Iterable[int | str] | None = None,
    exclude_invoice_distribution_ids: Iterable[str] | None = None,
    exclude_source_invoice_distribution_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    query_text = build_oracle_text_query(feature)
    if not query_text:
        return []
    filter_clauses, filter_params = _history_filters(
        dataset_types=dataset_types,
        exclude_ids=exclude_ids,
        exclude_invoice_distribution_ids=exclude_invoice_distribution_ids,
        exclude_source_invoice_distribution_ids=exclude_source_invoice_distribution_ids,
    )
    where_clauses = ["CONTAINS(RETRIEVAL_TEXT, :query_text, 1) > 0", *filter_clauses]
    sql = f"""
        SELECT ID, INVOICE_NUM, INVOICE_DATE, INVOICE_LINE_NUMBER,
               DISTRIBUTION_LINE_NUMBER, LINE_DESCRIPTION, LINE_TYPE,
               ACCOUNT_TYPE, GL_CODE, SEGMENT1, SEGMENT2, SEGMENT3,
               SEGMENT4, SEGMENT5, SEGMENT6,
               SCORE(1) AS TEXT_SCORE, VENDOR_NAME_NORM, LINE_TYPE_NORM, IS_SYNTHETIC
          FROM {TABLE_NAME}
         WHERE {' AND '.join(where_clauses)}
         ORDER BY SCORE(1) DESC
         FETCH FIRST {top_k} ROWS ONLY
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            params: dict[str, Any] = {"query_text": query_text, **filter_params}
            cur.execute(sql, params)
            results = []
            for row in cur.fetchall():
                case = _case_from_row(row, None)
                case["sparse_score"] = float(row[15]) if row[15] is not None else 0.0
                results.append(case)
            return results
    finally:
        conn.close()


def search_hybrid_history(
    feature: dict[str, Any],
    query_embedding: Iterable[float],
    dense_top_k: int = 15,
    sparse_top_k: int = 15,
    rrf_k: int = 60,
    target_accuracy: int = 95,
    dataset_types: Iterable[str] | None = ("HISTORY",),
    exclude_ids: Iterable[int | str] | None = None,
    exclude_invoice_distribution_ids: Iterable[str] | None = None,
    exclude_source_invoice_distribution_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    dense = search_dense_history(
        query_embedding,
        top_k=dense_top_k,
        target_accuracy=target_accuracy,
        line_type_norm=feature.get("line_type_norm"),
        dataset_types=dataset_types,
        exclude_ids=exclude_ids,
        exclude_invoice_distribution_ids=exclude_invoice_distribution_ids,
        exclude_source_invoice_distribution_ids=exclude_source_invoice_distribution_ids,
    )
    sparse = search_sparse_history(
        feature,
        top_k=sparse_top_k,
        line_type_norm=feature.get("line_type_norm"),
        dataset_types=dataset_types,
        exclude_ids=exclude_ids,
        exclude_invoice_distribution_ids=exclude_invoice_distribution_ids,
        exclude_source_invoice_distribution_ids=exclude_source_invoice_distribution_ids,
    )
    return reciprocal_rank_fusion(dense, sparse, k=rrf_k)


def get_identity_match(identity_key: str | None) -> dict[str, Any] | None:
    if not identity_key:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT ACCOUNT_TYPE, COUNT(*)
                 FROM {TABLE_NAME}
                 WHERE IDENTITY_KEY = :identity_key
                   AND NVL(UPPER(DATASET_TYPE), 'HISTORY') = 'HISTORY'
                   AND NVL(IS_SYNTHETIC, 'N') <> 'Y'
                 GROUP BY ACCOUNT_TYPE
                 ORDER BY COUNT(*) DESC, ACCOUNT_TYPE
                """,
                {"identity_key": identity_key},
            )
            counts = [(row[0], int(row[1])) for row in cur.fetchall() if row[0]]
        total = sum(count for _, count in counts)
        types = [account_type for account_type, _ in counts]
        return {
            "identity_key": identity_key,
            "account_types": types,
            "counts": dict(counts),
            "real_post_count": total,
            "conflict": len(types) > 1,
            "account_type": types[0] if len(types) == 1 else None,
        }
    finally:
        conn.close()


def get_vendor_prior(vendor_name_norm: str | None) -> dict[str, Any] | None:
    if not vendor_name_norm:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT ACCOUNT_TYPE, COUNT(*)
                 FROM {TABLE_NAME}
                 WHERE VENDOR_NAME_NORM = :vendor_name_norm
                   AND NVL(UPPER(DATASET_TYPE), 'HISTORY') = 'HISTORY'
                   AND NVL(IS_SYNTHETIC, 'N') <> 'Y'
                   AND ACCOUNT_TYPE IS NOT NULL
                 GROUP BY ACCOUNT_TYPE
                 ORDER BY COUNT(*) DESC, ACCOUNT_TYPE
                """,
                {"vendor_name_norm": vendor_name_norm},
            )
            counts = [(account_type, int(count)) for account_type, count in cur.fetchall() if account_type]
        total = sum(count for _, count in counts)
        if not total:
            return None
        dominant, dominant_count = counts[0]
        dominance = dominant_count / total
        entropy = -sum((count / total) * math.log(count / total) for _, count in counts)
        return {
            "vendor_name_norm": vendor_name_norm,
            "counts": dict(counts),
            "total": total,
            "dominant_account_type": dominant,
            "dominance": dominance,
            "entropy": entropy,
            "eligible": total >= 5 and dominance >= 0.90,
        }
    finally:
        conn.close()


def record_override(
    *,
    identity_key: str | None,
    line_description: str,
    vendor_name_norm: str | None,
    line_type_norm: str | None,
    predicted_account_type: str | None,
    corrected_account_type: str | None,
    confidence: float | None,
    reason: str | None,
    source: str = "streamlit",
) -> None:
    """Persist a human correction for audit and later history ingestion."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {OVERRIDE_TABLE_NAME} (
                    IDENTITY_KEY, LINE_DESCRIPTION, VENDOR_NAME_NORM, LINE_TYPE_NORM,
                    PREDICTED_ACCOUNT_TYPE, CORRECTED_ACCOUNT_TYPE, CONFIDENCE,
                    REASON, SOURCE_SYSTEM
                ) VALUES (
                    :identity_key, :line_description, :vendor_name_norm, :line_type_norm,
                    :predicted_account_type, :corrected_account_type, :confidence,
                    :reason, :source_system
                )
                """,
                {
                    "identity_key": identity_key,
                    "line_description": line_description,
                    "vendor_name_norm": vendor_name_norm,
                    "line_type_norm": line_type_norm,
                    "predicted_account_type": predicted_account_type,
                    "corrected_account_type": corrected_account_type,
                    "confidence": confidence,
                    "reason": reason,
                    "source_system": source,
                },
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def promote_correction_to_history(
    feature: dict[str, Any],
    corrected_account_type: str,
    embedding: Iterable[float],
    mapping: dict[str, Any],
) -> int:
    """Insert a human-accepted correction as a real, non-synthetic history row."""
    if not corrected_account_type or corrected_account_type == "Unknown":
        return 0
    source_key = hashlib.sha256(
        f"HITL|{feature.get('identity_key')}|{corrected_account_type}".encode("utf-8")
    ).hexdigest()
    row = {
        "source_key": source_key,
        "invoice_id": None,
        "invoice_num": f"HITL-{source_key[:24]}",
        "invoice_date": None,
        "invoice_distribution_id": None,
        "invoice_line_number": feature.get("line_index"),
        "distribution_line_number": None,
        "dist_code_combination_id": None,
        "line_description": feature.get("line_description") or "",
        "line_type": feature.get("line_type") or None,
        "vendor_id": None,
        "vendor_site_id": None,
        "business_unit_id": None,
        "ledger_id": None,
        "chart_of_accounts_id": None,
        "account_type": corrected_account_type,
        "gl_code": mapping.get("gl_code"),
        "segment1": mapping.get("segment1"),
        "segment2": mapping.get("segment2"),
        "segment3": mapping.get("segment3"),
        "segment4": mapping.get("segment4"),
        "segment5": mapping.get("segment5"),
        "segment6": mapping.get("segment6"),
        "retrieval_text": feature.get("retrieval_text"),
        "vendor_name_norm": feature.get("vendor_name_norm") or None,
        "line_type_norm": feature.get("line_type_norm"),
        "identity_key": feature.get("identity_key"),
        "amount_band": feature.get("amount_band"),
        "line_amount": feature.get("line_amount"),
        "unit_price": feature.get("unit_price"),
        "quantity_invoiced": feature.get("quantity_invoiced"),
        "embedding_model": EMBEDDING_MODEL,
        "dataset_type": "HISTORY",
        "is_synthetic": "N",
        "source_invoice_distribution_id": None,
        "synthetic_type": None,
        "embedding": list(embedding),
    }
    return bulk_upsert_history([row])


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
