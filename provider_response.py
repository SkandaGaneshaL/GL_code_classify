"""Provider response normalization for OCI Responses and native Chat APIs.

The OCI OpenAI-compatible APIs and the OCI native SDK return materially
different object graphs.  This module creates a small provider-neutral envelope
so parsing and logprob analysis do not need to know which contract produced a
response.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from .usage import TokenUsage, normalize_provider_usage
except ImportError:
    from usage import TokenUsage, normalize_provider_usage


@dataclass(frozen=True)
class TokenEvidence:
    token: str
    logprob: float
    top_logprobs: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True)
class NormalizedProviderResponse:
    """Canonical response envelope consumed by parsing and confidence code."""

    text: str
    selected_tokens: tuple[TokenEvidence, ...] = ()
    candidate_logprobs: dict[str, float] = field(default_factory=dict)
    selected_account_type: str | None = None
    provider: str = "unknown"
    route: str = "unknown"
    model: str | None = None
    finish_reason: str | None = None
    reasoning_text: str | None = None
    evidence_scope_hint: str = "selected_only"
    requires_completion_retry: bool = False
    usage: TokenUsage | None = None
    request_id: str | None = None
    attempt: int = 1
    latency_ms: float | None = None


COMPACT_CODE_MAP_VERSION = "compact-code-v1"


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            text = _get(item, "text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    text = _get(content, "text")
    return str(text) if text else ""


def _choices_container(raw_response: Any) -> Any:
    for candidate in (raw_response, _get(raw_response, "data")):
        nested = _get(candidate, "chat_response")
        if nested is not None:
            return nested
    return raw_response


def _usage_payload(raw_response: Any) -> tuple[Any, str | None]:
    candidates = (
        (raw_response, "response.usage"),
        (_get(raw_response, "data"), "data.usage"),
        (_get(raw_response, "chat_response"), "chat_response.usage"),
        (_get(_get(raw_response, "data"), "chat_response"), "data.chat_response.usage"),
        (_get(raw_response, "response"), "response.usage"),
    )
    for candidate, location in candidates:
        payload = _get(candidate, "usage")
        if payload is not None:
            return payload, location
    return None, None


def _request_id(raw_response: Any) -> str | None:
    headers = _get(raw_response, "headers")
    if isinstance(headers, dict):
        return headers.get("opc-request-id") or headers.get("x-request-id")
    return _get(raw_response, "request_id") or _get(raw_response, "id")


def _response_usage(
    raw_response: Any,
    *,
    model: str | None,
    provider: str,
    route: str,
    request_id: str | None,
    attempt: int,
    latency_ms: float | None,
    finish_reason: str | None,
) -> TokenUsage:
    payload, location = _usage_payload(raw_response)
    return normalize_provider_usage(
        payload,
        model=model,
        provider=provider,
        route=route,
        request_id=request_id,
        attempt=attempt,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        usage_location=location,
    )


def extract_provider_text(raw_response: Any) -> str:
    """Extract final assistant text from all supported provider shapes."""
    if isinstance(raw_response, NormalizedProviderResponse):
        text = raw_response.text
        if text:
            return text.strip()

    output_text = _get(raw_response, "output_text")
    if output_text:
        return str(output_text).strip()

    choices = _get(_choices_container(raw_response), "choices") or []
    if choices:
        message = _get(choices[0], "message")
        content = _get(message, "content")
        text = _content_text(content)
        if text:
            return text.strip()

    parts: list[str] = []
    output = _get(raw_response, "output") or []
    for item in output:
        parts.append(_content_text(_get(item, "content")))
    text = "".join(parts).strip()
    if text:
        return text
    raise ValueError("OCI provider returned no final text output")


def _top_logprob_map(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for token, score in value.items():
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                result[str(token)] = score
        return result
    result = {}
    for item in value or []:
        token = _get(item, "token")
        score = _get(item, "logprob")
        if token is None or score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            result[str(token)] = score
    return result


def _native_logprob_arrays(raw_response: Any) -> tuple[list[str], list[float], list[dict[str, float]]]:
    container = _choices_container(raw_response)
    choices = _get(container, "choices") or []
    if not choices:
        return [], [], []
    logprobs = _get(choices[0], "logprobs")
    if logprobs is None:
        return [], [], []
    tokens = [str(token) for token in (_get(logprobs, "tokens") or [])]
    values: list[float] = []
    for value in _get(logprobs, "token_logprobs") or []:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    tops = [_top_logprob_map(value) for value in (_get(logprobs, "top_logprobs") or [])]
    return tokens, values, tops


def _native_metadata(raw_response: Any) -> tuple[str | None, str | None, str | None]:
    container = _choices_container(raw_response)
    choices = _get(container, "choices") or []
    if not choices:
        return None, None, None
    choice = choices[0]
    message = _get(choice, "message")
    reasoning = _get(message, "reasoning_content")
    return (
        str(reasoning) if reasoning else None,
        _get(choice, "finish_reason"),
        _get(raw_response, "model_id") or _get(raw_response, "model")
        or _get(_get(raw_response, "data"), "model_id")
        or _get(_get(raw_response, "data"), "model"),
    )


def _walk_generic(value: Any, seen: set[int] | None = None):
    seen = seen or set()
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_generic(child, seen)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_generic(child, seen)
        return
    for name in ("output", "choices", "content", "message", "logprobs", "top_logprobs"):
        yield from _walk_generic(_get(value, name), seen)


def _generic_token_records(raw_response: Any) -> list[TokenEvidence]:
    candidates: list[list[TokenEvidence]] = []
    for node in _walk_generic(raw_response):
        logprobs = _get(node, "logprobs")
        values = logprobs if isinstance(logprobs, (list, tuple)) else _get(logprobs, "content")
        if not isinstance(values, (list, tuple)):
            continue
        records: list[TokenEvidence] = []
        for item in values:
            token = _get(item, "token")
            score = _get(item, "logprob")
            if token is None or score is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                top = _top_logprob_map(_get(item, "top_logprobs"))
                records.append(
                    TokenEvidence(
                        token=str(token),
                        logprob=score,
                        top_logprobs=(top,) if top else (),
                    )
                )
        if records:
            candidates.append(records)
    return max(candidates, key=len, default=[])


def _explicit_candidate_scores(raw_response: Any) -> dict[str, float]:
    for node in _walk_generic(raw_response):
        scores = _get(node, "candidate_logprobs")
        if not isinstance(scores, dict):
            continue
        result: dict[str, float] = {}
        for name, value in scores.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                result[str(name)] = value
        if result:
            return result
    return {}


def _align_final_records(
    final_text: str,
    tokens: list[str],
    values: list[float],
    tops: list[dict[str, float]],
) -> list[TokenEvidence]:
    """Align final-channel text to the provider's complete token stream.

    gpt-oss native responses include Harmony reasoning/channel tokens in the
    logprob arrays while exposing final text separately.  The rightmost exact
    match selects the final channel and avoids assigning reasoning evidence to
    the JSON answer.  If alignment is impossible, return no evidence rather
    than guessing.
    """
    token_text = "".join(tokens)
    start = token_text.rfind(final_text)
    if start < 0:
        return []
    end = start + len(final_text)
    records: list[TokenEvidence] = []
    offset = 0
    for index, token in enumerate(tokens):
        next_offset = offset + len(token)
        if offset < end and next_offset > start and index < len(values):
            value = values[index]
            if math.isfinite(value):
                records.append(
                    TokenEvidence(
                        token=token,
                        logprob=value,
                        top_logprobs=(tops[index],) if index < len(tops) and tops[index] else (),
                    )
                )
        offset = next_offset
    return records


def _code_span(text: str) -> tuple[str, int, int] | None:
    match = re.search(r'"account_type_code"\s*:\s*"([^"\\]+)"', text)
    if not match:
        return None
    return match.group(1), match.start(1), match.end(1)


def _code_records(
    records: list[TokenEvidence],
    text: str,
    value_start: int,
    value_end: int,
) -> list[TokenEvidence]:
    selected: list[TokenEvidence] = []
    offset = 0
    for record in records:
        next_offset = offset + len(record.token)
        if offset < value_end and next_offset > value_start:
            selected.append(record)
        offset = next_offset
    return selected


def normalize_native_chat_response(
    raw_response: Any,
    *,
    allowed_account_types: list[str],
    code_map: dict[str, str] | None = None,
    provider: str = "oci_native_chat",
    model: str | None = None,
    request_id: str | None = None,
    attempt: int = 1,
    latency_ms: float | None = None,
) -> NormalizedProviderResponse:
    """Normalize an OCI native Chat response, optionally decoding compact codes."""
    text = extract_provider_text(raw_response)
    reasoning_text, finish_reason, response_model = _native_metadata(raw_response)
    model = model or response_model
    request_id = request_id or _request_id(raw_response)
    usage = _response_usage(
        raw_response,
        model=model,
        provider=provider,
        route=provider,
        request_id=request_id,
        attempt=attempt,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )
    tokens, values, tops = _native_logprob_arrays(raw_response)
    aligned = _align_final_records(text, tokens, values, tops)
    requires_retry = str(finish_reason or "").lower() == "length"

    if not code_map:
        selected_account_type = None
        selected_tokens: list[TokenEvidence] = []
        try:
            payload = json.loads(text)
            selected_account_type = str(payload.get("account_type") or "").strip() or None
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        label_match = re.search(r'"account_type"\s*:\s*"([^"\\]+)"', text)
        if label_match:
            selected_tokens = _code_records(
                aligned,
                text,
                label_match.start(1),
                label_match.end(1),
            )
        return NormalizedProviderResponse(
            text=text,
            selected_tokens=tuple(selected_tokens),
            selected_account_type=selected_account_type,
            provider=provider,
            route=provider,
            model=model,
            finish_reason=finish_reason,
            reasoning_text=reasoning_text,
            requires_completion_retry=requires_retry,
            usage=usage,
            request_id=request_id,
            attempt=attempt,
            latency_ms=latency_ms,
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Native OCI response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Native OCI response must be a JSON object")

    code_value = str(payload.get("account_type_code") or "").strip()
    if code_value not in code_map:
        raise ValueError("Native OCI response returned an unknown account_type_code")
    selected_account_type = code_map[code_value]
    payload.pop("account_type_code", None)
    payload["account_type"] = selected_account_type
    inferred = payload.get("inferred_account_type")
    if inferred in code_map:
        payload["inferred_account_type"] = code_map[inferred]
    canonical_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    span = _code_span(text)
    selected_code_records = _code_records(aligned, text, span[1], span[2]) if span else []
    selected_code_score = selected_code_records[0].logprob if selected_code_records else None
    observed: dict[str, float] = {}
    top_values: dict[str, float] = {}
    for record in selected_code_records:
        for token, score in (record.top_logprobs[0] if record.top_logprobs else {}).items():
            normalized_token = token.strip()
            if normalized_token in code_map:
                top_values[normalized_token] = score
    if selected_code_score is not None:
        top_values.setdefault(code_value, selected_code_score)
    for code, score in top_values.items():
        observed[code_map[code]] = score

    complete = bool(code_map) and set(top_values) >= set(code_map)
    selected_tokens = tuple(
        TokenEvidence(
            token=code_value,
            logprob=selected_code_score,
            top_logprobs=(
                {
                    code_map[token]: score
                    for token, score in top_values.items()
                },
            )
            if top_values
            else (),
        )
        for _ in [0]
        if selected_code_score is not None
    )
    return NormalizedProviderResponse(
        text=canonical_text,
        selected_tokens=selected_tokens,
        candidate_logprobs=observed,
        selected_account_type=selected_account_type,
        provider=provider,
        route=provider,
        model=model,
        finish_reason=finish_reason,
        reasoning_text=reasoning_text,
        evidence_scope_hint="candidate_set" if complete else "selected_only",
        requires_completion_retry=requires_retry,
        usage=usage,
        request_id=request_id,
        attempt=attempt,
        latency_ms=latency_ms,
    )


def normalize_provider_response(
    raw_response: Any,
    *,
    allowed_account_types: list[str] | None = None,
    provider: str = "unknown",
    route: str | None = None,
    model: str | None = None,
    request_id: str | None = None,
    attempt: int = 1,
    latency_ms: float | None = None,
) -> NormalizedProviderResponse:
    """Normalize Responses/Chat output while preserving selected token evidence."""
    if isinstance(raw_response, NormalizedProviderResponse):
        return raw_response
    text = extract_provider_text(raw_response)
    records = _generic_token_records(raw_response)
    aligned = _align_final_records(
        text,
        [record.token for record in records],
        [record.logprob for record in records],
        [record.top_logprobs[0] if record.top_logprobs else {} for record in records],
    )
    selected_account_type = None
    selected_tokens: list[TokenEvidence] = []
    match = re.search(r'"account_type"\s*:\s*"([^"\\]+)"', text)
    if match:
        selected_account_type = match.group(1)
        selected_tokens = _code_records(aligned, text, match.start(1), match.end(1))
    container = _choices_container(raw_response)
    choices = _get(container, "choices") or []
    finish_reason = _get(choices[0], "finish_reason") if choices else None
    response_model = model or _get(raw_response, "model") or _get(raw_response, "model_id")
    resolved_route = route or provider
    request_id = request_id or _request_id(raw_response)
    usage = _response_usage(
        raw_response,
        model=response_model,
        provider=provider,
        route=resolved_route,
        request_id=request_id,
        attempt=attempt,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )
    return NormalizedProviderResponse(
        text=text,
        selected_tokens=tuple(selected_tokens),
        candidate_logprobs=_explicit_candidate_scores(raw_response),
        selected_account_type=selected_account_type,
        provider=provider,
        route=resolved_route,
        model=response_model,
        finish_reason=finish_reason,
        evidence_scope_hint="selected_only",
        requires_completion_retry=str(finish_reason or "").lower() == "length",
        usage=usage,
        request_id=request_id,
        attempt=attempt,
        latency_ms=latency_ms,
    )
