from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

try:
    from .provider_response import extract_provider_text
except ImportError:
    from provider_response import extract_provider_text


def extract_response_text(raw_response: Any) -> str:
    try:
        return extract_provider_text(raw_response)
    except ValueError as exc:
        raise ValueError("OCI provider returned no text output") from exc


def parse_account_type_response(
    raw_response: Any,
    expected_line_description: str,
    allowed_account_types: list[str] | None = None,
) -> dict[str, Any]:
    text = extract_response_text(raw_response)
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")

    required = {
        "line_description",
        "account_type",
        "inferred_account_type",
        "reason",
        "confidence",
    }
    if set(result) != required:
        raise ValueError(f"LLM response fields must be exactly: {sorted(required)}")
    for field in ("line_description", "account_type", "reason"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    inferred = result["inferred_account_type"]
    if inferred is not None and (not isinstance(inferred, str) or not inferred.strip()):
        raise ValueError("inferred_account_type must be null or a non-empty string")
    result["account_type"] = _normalize_account_type(result["account_type"])
    if result["inferred_account_type"] is not None:
        result["inferred_account_type"] = _normalize_account_type(result["inferred_account_type"])
    allowed = {_normalize_account_type(value) for value in (allowed_account_types or [])}
    allowed.add("Unknown")
    if allowed_account_types and result["account_type"] not in allowed:
        raise ValueError("account_type is not one of the allowed taxonomy values")
    if result["inferred_account_type"] is not None and allowed_account_types:
        if result["inferred_account_type"] not in allowed:
            raise ValueError("inferred_account_type is not one of the allowed taxonomy values")
    if result["account_type"].lower() == "unknown":
        result["account_type"] = "Unknown"
    elif inferred is not None:
        raise ValueError("inferred_account_type must be null when account_type is not Unknown")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number between 0 and 1")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    result["line_description"] = expected_line_description
    result["confidence"] = float(confidence)
    return result


def _normalize_account_type(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\ufffd", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s*-\s*", " - ", text)
    return re.sub(r"\s+", " ", text).strip()
