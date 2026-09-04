from __future__ import annotations

import json
import re
from typing import Any


def extract_response_text(raw_response: Any) -> str:
    output_text = getattr(raw_response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    parts: list[str] = []
    for item in getattr(raw_response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    if parts:
        return "".join(parts).strip()
    raise ValueError("OCI Responses API returned no text output")


def parse_account_type_response(
    raw_response: Any, expected_line_description: str
) -> dict[str, Any]:
    text = extract_response_text(raw_response)
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {text}") from exc
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
    if result["account_type"].strip().lower() == "unknown":
        result["account_type"] = "Unknown"
        if inferred is None:
            raise ValueError("inferred_account_type is required when account_type is Unknown")
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
