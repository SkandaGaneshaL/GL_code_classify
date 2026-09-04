from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

try:
    from . import db_utils
    from .config import CANDIDATE_TOP_K, DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
    from .embeddings import embed_query
    from .llm_client import OCIResponsesClient
    from .prompt_builder import (
        build_qualified_cases_prompt,
        build_weak_cases_prompt,
    )
    from .response_parser import parse_account_type_response
except ImportError:  # Supports running the files directly from this folder.
    import db_utils
    from config import CANDIDATE_TOP_K, DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
    from embeddings import embed_query
    from llm_client import OCIResponsesClient
    from prompt_builder import build_qualified_cases_prompt, build_weak_cases_prompt
    from response_parser import parse_account_type_response


ACCOUNT_TYPE_TO_SEGMENT3 = {
    "Accounts Payable Clearing": "22190",
    "Accrued Expenses": "24220",
    "Accrued Receipts": "22210",
    "Airfare": "60512",
    "Asset Clearing": "15910",
    "Car Mileage": "60514",
    "Contractor Expenses": "65600",
    "Hotel / Accomodation": "60530",
    "Meals": "60521",
    "Miscellaneous": "60540",
    "Operating Lease Expense - Company Labor": "63611",
    "Operating Lease Expense - Equipment Rental": "63612",
    # Keep one canonical spelling. LLMs and source files may use an en dash,
    # hyphen, or the Unicode replacement character for the separator.
    "Operating Lease Expense - Electricity": "63613",
    "Project Labor Cost": "59110",
    "Supplies": "60520",
    "Withholding Tax Payable": "25400",
}


def normalize_account_type(account_type: str) -> str:
    """Normalize harmless Unicode/spacing variations from the LLM output."""
    value = unicodedata.normalize("NFKC", str(account_type or ""))
    value = value.replace("\ufffd", "-")  # mojibake/replacement character
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"\s*[-]\s*", " - ", value)
    return re.sub(r"\s+", " ", value).strip()


def map_account_type_to_segments(account_type: str) -> dict[str, Any] | None:
    """Map the LLM account type through the fixed POC account taxonomy."""
    normalized_type = normalize_account_type(account_type)
    normalized_mapping = {
        normalize_account_type(key): value
        for key, value in ACCOUNT_TYPE_TO_SEGMENT3.items()
    }
    segment3 = normalized_mapping.get(normalized_type)
    if not segment3:
        return None
    # For this POC, the chart combination uses fixed S1/S2/S4/S5/S6 values.
    return {
        "gl_code": f"101.10.{segment3}.000.000.000",
        "segment1": "101",
        "segment2": "10",
        "segment3": segment3,
        "segment4": "000",
        "segment5": "000",
        "segment6": "000",
        "mapping_source": "deterministic_account_type_mapping",
        "mapping_row_count": None,
    }


def classify_line(
    line_description: str,
    embedder: object,
    settings: object,
    top_k: int = DEFAULT_TOP_K,
    min_similarity_score: float = MIN_SIMILARITY_SCORE,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    line_description = line_description.strip()
    if not line_description:
        raise ValueError("Line description cannot be empty")

    if stage_callback:
        stage_callback("embedding_running")
    query_embedding = embed_query(line_description, embedder)
    if stage_callback:
        stage_callback("embedding_success")
        stage_callback("retrieval_running")
    if not 0.0 <= min_similarity_score <= 1.0:
        raise ValueError("min_similarity_score must be between 0.0 and 1.0")
    if not 1 <= top_k <= CANDIDATE_TOP_K:
        raise ValueError(f"top_k must be between 1 and {CANDIDATE_TOP_K}")
    # Retrieve five candidates first. Threshold, ranking, and configurable
    # top_k selection happen before the prompt is built.
    candidate_cases = db_utils.search_similar_history(
        query_embedding, top_k=CANDIDATE_TOP_K, target_accuracy=95
    )
    qualifying_cases = [
        case
        for case in candidate_cases
        if float(case.get("similarity_score") or 0.0) >= min_similarity_score
    ]
    qualifying_cases.sort(
        key=lambda case: float(case.get("similarity_score") or 0.0),
        reverse=True,
    )
    if qualifying_cases:
        # Strong evidence: only the highest-scoring qualifying cases, up to
        # the configured top_k (3 by default), are sent to the LLM.
        llm_cases = qualifying_cases[:top_k]
        display_cases = llm_cases
        evidence_mode = "qualified"
        build_prompt = build_qualified_cases_prompt
    else:
        # Weak evidence: send all five candidates to the backup prompt. They
        # are labeled for the UI, but the prompt must judge semantic fit rather
        # than copy the nearest account type.
        llm_cases = sorted(
            candidate_cases,
            key=lambda case: float(case.get("similarity_score") or 0.0),
            reverse=True,
        )[:CANDIDATE_TOP_K]
        display_cases = [dict(case, lower_similarity=True) for case in llm_cases]
        evidence_mode = "weak"
        build_prompt = build_weak_cases_prompt
    if stage_callback:
        stage_callback("retrieval_success")

    prompt = build_prompt(line_description, llm_cases)
    if stage_callback:
        stage_callback("llm_running")
    client = OCIResponsesClient()
    result = parse_account_type_response(client.call(prompt), line_description)
    if stage_callback:
        stage_callback("llm_success")
        stage_callback("mapping_running")
    mapping = map_account_type_to_segments(result["account_type"])
    if mapping:
        result.update(mapping)
    else:
        result.update(
            {
                "gl_code": None,
                **{f"segment{i}": None for i in range(1, 7)},
                "mapping_source": "unmapped",
                "mapping_row_count": 0,
            }
        )
    if stage_callback:
        stage_callback("mapping_success")
    result["retrieved_cases"] = display_cases
    result["candidate_case_count"] = len(candidate_cases)
    result["qualifying_case_count"] = len(qualifying_cases)
    result["loaded_case_count"] = len(llm_cases)
    result["candidate_top_k"] = CANDIDATE_TOP_K
    result["retrieved_case_count"] = len(llm_cases)
    result["llm_case_count"] = len(llm_cases)
    result["evidence_mode"] = evidence_mode
    result["minimum_similarity_score"] = min_similarity_score
    return result
