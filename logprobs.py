"""Provider-neutral parsing for token log probabilities.

The OCI OpenAI-compatible Responses API and OCI Chat APIs expose different
response shapes.  This module deliberately works with both dicts and SDK
objects and only scores the JSON ``account_type`` value.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterator

try:
    from .provider_response import NormalizedProviderResponse
except ImportError:
    from provider_response import NormalizedProviderResponse


@dataclass(frozen=True)
class LogprobAnalysis:
    status: str
    evidence_scope: str = "selected_only"
    selected_account_type: str | None = None
    selected_loglikelihood: float | None = None
    selected_mean_logprob: float | None = None
    selected_probability_proxy: float | None = None
    runner_up_account_type: str | None = None
    runner_up_loglikelihood: float | None = None
    runner_up_mean_logprob: float | None = None
    logprob_margin: float | None = None
    token_count: int = 0
    saturated: bool = False
    saturated_token_count: int = 0
    saturation_reason: str | None = None
    candidate_scores: tuple[dict[str, Any], ...] = ()
    tokens: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_text_span(raw_response: Any, label: str) -> tuple[float, int] | None:
    text = _response_text(raw_response)
    records = _find_logprob_records(raw_response)
    if not text or not records:
        return None
    start = text.find(label)
    if start < 0:
        return None
    end = start + len(label)
    values: list[float] = []
    offset = 0
    for record in records:
        token = str(_get(record, "token", ""))
        next_offset = offset + len(token)
        if offset < end and next_offset > start:
            try:
                value = float(_get(record, "logprob"))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
        offset = next_offset
    return (sum(values) / len(values), len(values)) if values else None


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _records(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _walk(value: Any, seen: set[int] | None = None) -> Iterator[Any]:
    """Walk SDK objects without traversing arbitrary class internals."""
    seen = seen or set()
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    if isinstance(value, dict):
        for child in value.values():
            yield value
            yield from _walk(child, seen)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child, seen)
        return
    for name in ("output", "choices", "content", "message", "logprobs", "top_logprobs"):
        try:
            child = getattr(value, name, None)
        except Exception:
            child = None
        if child is not None:
            yield value
            yield from _walk(child, seen)


def _find_logprob_records(raw_response: Any) -> list[Any]:
    candidates: list[list[Any]] = []
    for node in _walk(raw_response):
        value = _get(node, "logprobs")
        if isinstance(value, (list, tuple)):
            rows = [row for row in value if _get(row, "token") is not None and _get(row, "logprob") is not None]
            if rows:
                candidates.append(rows)
        content = _get(value, "content")
        if isinstance(content, (list, tuple)):
            rows = [row for row in content if _get(row, "token") is not None and _get(row, "logprob") is not None]
            if rows:
                candidates.append(rows)
    return max(candidates, key=len, default=[])


def _response_text(raw_response: Any) -> str:
    if isinstance(raw_response, NormalizedProviderResponse):
        return raw_response.text
    output_text = _get(raw_response, "output_text")
    if output_text:
        return str(output_text)
    choices = _get(raw_response, "choices") or []
    if choices:
        message = _get(choices[0], "message")
        content = _get(message, "content")
        if isinstance(content, (list, tuple)):
            content = "".join(str(_get(item, "text", "")) for item in content)
        if content:
            return str(content)
    parts: list[str] = []
    for item in _get(raw_response, "output") or []:
        for content in _get(item, "content") or []:
            text = _get(content, "text")
            if text:
                parts.append(str(text))
    return "".join(parts)


def _analysis_from_normalized(
    response: NormalizedProviderResponse,
    allowed_account_types: list[str] | None,
    saturation_threshold: float,
) -> LogprobAnalysis:
    """Analyze provider-neutral evidence already aligned to final output."""
    selected = response.selected_account_type
    records = list(response.selected_tokens)
    if not selected or not records:
        return LogprobAnalysis(status="missing", evidence_scope="missing")

    values = [record.logprob for record in records if math.isfinite(record.logprob)]
    if not values:
        return LogprobAnalysis(
            status="invalid",
            evidence_scope="invalid",
            selected_account_type=selected,
        )

    token_details: list[dict[str, Any]] = []
    for record in records:
        alternatives: list[dict[str, Any]] = []
        for top_map in record.top_logprobs:
            allowed = {str(value).strip() for value in (allowed_account_types or [])}
            allowed.add("Unknown")
            for token, logprob in top_map.items():
                if token.strip() not in allowed or not math.isfinite(logprob):
                    continue
                alternatives.append(
                    {
                        "token": token.strip(),
                        "logprob": logprob,
                        "probability": math.exp(logprob) if logprob <= 0 else None,
                    }
                )
        token_details.append(
            {
                "token": record.token,
                "logprob": record.logprob,
                "probability": math.exp(record.logprob) if record.logprob <= 0 else None,
                "selected": True,
                "top_alternatives": alternatives,
            }
        )

    candidate_rows = list(
        _candidate_score_rows(
            response.candidate_logprobs,
            selected=selected,
            allowed_account_types=allowed_account_types,
        )
    )
    if response.evidence_scope_hint == "candidate_set" and candidate_rows:
        maximum = max(float(row["mean_logprob"]) for row in candidate_rows)
        denominator = sum(
            math.exp(float(row["mean_logprob"]) - maximum)
            for row in candidate_rows
        )
        if denominator > 0:
            for row in candidate_rows:
                row["normalized_share"] = math.exp(
                    float(row["mean_logprob"]) - maximum
                ) / denominator
    alternatives = []
    if response.evidence_scope_hint == "candidate_set":
        alternatives = [
            (str(row["account_type"]), float(row["mean_logprob"]))
            for row in candidate_rows
            if str(row["account_type"]).strip() != selected.strip()
        ]
    runner_up_type, runner_up_score = (max(alternatives, key=lambda item: item[1]) if alternatives else (None, None))

    probability_values = [math.exp(value) for value in values if value <= 0]
    saturated_values = [value for value in probability_values if value >= saturation_threshold]
    saturated_token_count = len(saturated_values)
    saturated = bool(probability_values) and (
        len(saturated_values) == len(probability_values)
        or (len(probability_values) >= 2 and saturated_token_count >= 2)
    )
    saturation_reason = None
    if saturated:
        saturation_reason = "multiple selected tokens have probability proxies at or above the saturation threshold"

    selected_sum = sum(values)
    selected_mean = selected_sum / len(values)
    margin = selected_mean - runner_up_score if runner_up_score is not None else None
    evidence_scope = "saturated" if saturated else response.evidence_scope_hint
    return LogprobAnalysis(
        status="saturated" if saturated else "available",
        evidence_scope=evidence_scope,
        selected_account_type=selected,
        selected_loglikelihood=selected_sum,
        selected_mean_logprob=selected_mean,
        selected_probability_proxy=math.exp(selected_mean) if selected_mean <= 0 else None,
        runner_up_account_type=runner_up_type,
        runner_up_loglikelihood=runner_up_score,
        runner_up_mean_logprob=runner_up_score,
        logprob_margin=margin,
        token_count=len(values),
        saturated=saturated,
        saturated_token_count=saturated_token_count,
        saturation_reason=saturation_reason,
        candidate_scores=tuple(candidate_rows),
        tokens=tuple(token_details),
    )


def _account_value_span(text: str) -> tuple[str, int, int] | None:
    marker = re.search(r'"account_type"\s*:\s*', text)
    if not marker:
        return None
    start_quote = text.find('"', marker.end())
    if start_quote < 0:
        return None
    try:
        value, consumed = json.JSONDecoder().raw_decode(text[start_quote:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, str):
        return None
    value_start = start_quote + 1
    value_end = start_quote + consumed - 1
    return value, value_start, value_end


def _candidate_scores(raw_response: Any) -> dict[str, float]:
    """Read optional provider-specific candidate scores when present."""
    for node in _walk(raw_response):
        scores = _get(node, "candidate_logprobs")
        if isinstance(scores, dict):
            result: dict[str, float] = {}
            for key, value in scores.items():
                try:
                    result[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if result:
                return result
    return {}


def _candidate_score_rows(
    scores: dict[str, float],
    *,
    selected: str,
    allowed_account_types: list[str] | None,
) -> tuple[dict[str, Any], ...]:
    """Return sanitized explicit candidate scores.

    ``candidate_logprobs`` is an optional provider/application extension.  It
    is deliberately treated as evidence only when the provider explicitly
    supplies it; missing taxonomy values are not assigned zero probability.
    """
    allowed = None
    if allowed_account_types is not None:
        allowed = {str(value).strip() for value in allowed_account_types}
        allowed.add("Unknown")
    rows: list[dict[str, Any]] = []
    for name, score in scores.items():
        normalized_name = str(name).strip()
        if allowed is not None and normalized_name not in allowed:
            continue
        if not math.isfinite(score):
            continue
        rows.append(
            {
                "account_type": normalized_name,
                "mean_logprob": score,
                "probability_proxy": math.exp(score) if score <= 0 else None,
                "status": "selected" if normalized_name == selected.strip() else "scored",
            }
        )
    return tuple(rows)


def _allowed_alternatives(record: Any, allowed_account_types: list[str] | None) -> list[dict[str, Any]]:
    allowed = {str(value).strip() for value in (allowed_account_types or [])}
    allowed.add("Unknown")
    alternatives: list[dict[str, Any]] = []
    for item in _get(record, "top_logprobs", None) or []:
        token = str(_get(item, "token", ""))
        if token.strip() not in allowed:
            continue
        try:
            logprob = float(_get(item, "logprob"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(logprob):
            alternatives.append(
                {
                    "token": token,
                    "logprob": logprob,
                    "probability": math.exp(logprob) if logprob <= 0 else None,
                }
            )
    return alternatives


def analyze_logprobs(
    raw_response: Any,
    *,
    allowed_account_types: list[str] | None = None,
    saturation_threshold: float = 0.999,
) -> LogprobAnalysis:
    """Score only the selected ``account_type`` JSON value.

    ``top_logprobs`` generally does not contain complete multi-token scores
    for every alternative label.  A runner-up is therefore reported only when
    the provider supplies explicit candidate scores; absent alternatives are
    never converted to a probability of zero.
    """
    if isinstance(raw_response, NormalizedProviderResponse):
        return _analysis_from_normalized(
            raw_response,
            allowed_account_types,
            saturation_threshold,
        )

    text = _response_text(raw_response)
    records = _find_logprob_records(raw_response)
    span = _account_value_span(text)
    if not records or span is None:
        return LogprobAnalysis(status="missing", evidence_scope="missing")

    selected, value_start, value_end = span
    token_text = "".join(str(_get(record, "token", "")) for record in records)
    using_token_text = value_start > len(token_text)
    source_text = token_text if using_token_text else text
    if using_token_text:
        local_span = _account_value_span(token_text)
        if local_span is None:
            return LogprobAnalysis(status="invalid", evidence_scope="invalid")
        _, value_start, value_end = local_span

    selected_records: list[Any] = []
    offset = 0
    for record in records:
        token = str(_get(record, "token", ""))
        next_offset = offset + len(token)
        if offset < value_end and next_offset > value_start:
            selected_records.append(record)
        offset = next_offset

    valid_records: list[tuple[Any, float]] = []
    for record in selected_records:
        try:
            logprob = float(_get(record, "logprob"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(logprob):
            valid_records.append((record, logprob))
        values = [logprob for _, logprob in valid_records]
        if not values:
            return LogprobAnalysis(status="invalid", evidence_scope="invalid", selected_account_type=selected)

    token_details: list[dict[str, Any]] = []
    for record, logprob in valid_records:
        token = str(_get(record, "token", ""))
        token_details.append(
            {
                "token": token,
                "logprob": logprob,
                "probability": math.exp(logprob) if logprob <= 0 else None,
                "selected": True,
                "top_alternatives": _allowed_alternatives(record, allowed_account_types),
            }
        )

    scores = _candidate_scores(raw_response)
    candidate_rows = _candidate_score_rows(
        scores,
        selected=selected,
        allowed_account_types=allowed_account_types,
    )
    runner_up_type = None
    runner_up_score = None
    if candidate_rows:
        normalized_selected = selected.strip()
        alternatives = [
            (str(row["account_type"]), float(row["mean_logprob"]))
            for row in candidate_rows
            if str(row["account_type"]).strip() != normalized_selected
        ]
        if alternatives:
            runner_up_type, runner_up_score = max(alternatives, key=lambda item: item[1])

    probability_values = [math.exp(value) for value in values if value <= 0]
    saturated_values = [value for value in probability_values if value >= saturation_threshold]
    saturated_token_count = len(saturated_values)
    # A provider returning exact/near-zero logprobs for multiple continuation
    # tokens is commonly a constrained-decoding or display-precision signal.
    # Keep it review-only instead of presenting a false certainty signal.
    saturated = bool(probability_values) and (
        len(saturated_values) == len(probability_values)
        or (len(probability_values) >= 2 and saturated_token_count >= 2)
    )
    saturation_reason = None
    if saturated:
        saturation_reason = "multiple selected tokens have probability proxies at or above the saturation threshold"
    selected_sum = sum(values)
    selected_mean = selected_sum / len(values)
    margin = selected_mean - runner_up_score if runner_up_score is not None else None
    selected_probability_proxy = math.exp(selected_mean) if selected_mean <= 0 else None
    evidence_scope = "saturated" if saturated else "selected_only"
    if candidate_rows and not saturated:
        evidence_scope = "candidate_set"
        allowed = None
        if allowed_account_types is not None:
            allowed = {str(value).strip() for value in allowed_account_types}
            allowed.add("Unknown")
        if allowed is not None and {row["account_type"] for row in candidate_rows} >= allowed:
            evidence_scope = "full_taxonomy"
    return LogprobAnalysis(
        status="saturated" if saturated else "available",
        evidence_scope=evidence_scope,
        selected_account_type=selected,
        selected_loglikelihood=selected_sum,
        selected_mean_logprob=selected_mean,
        selected_probability_proxy=selected_probability_proxy,
        runner_up_account_type=runner_up_type,
        runner_up_loglikelihood=runner_up_score,
        runner_up_mean_logprob=runner_up_score,
        logprob_margin=margin,
        token_count=len(values),
        saturated=saturated,
        saturated_token_count=saturated_token_count,
        saturation_reason=saturation_reason,
        candidate_scores=candidate_rows,
        tokens=tuple(token_details),
    )


def score_label_logprobs(raw_response: Any, label: str) -> float | None:
    """Return the mean token logprob for an exact label-only response."""
    scored = _score_text_span(raw_response, label)
    return scored[0] if scored else None


def analyze_json_field_logprobs(
    raw_response: Any,
    field_name: str,
    allowed_values: list[str],
) -> dict[str, Any]:
    """Probe logprob coverage for a compact JSON enum field.

    This intentionally returns only capability metadata.  It does not expose
    the probe text or token values, making it safe for cached diagnostics.
    """
    text = _response_text(raw_response)
    records = _find_logprob_records(raw_response)
    if not text or not records:
        return {"supported": False, "evidence_scope": "missing"}
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)"', text)
    if not match:
        return {"supported": False, "evidence_scope": "invalid"}
    selected = match.group(1)
    allowed = {str(value).strip() for value in allowed_values}
    if selected not in allowed:
        return {"supported": False, "evidence_scope": "invalid"}
    value_start, value_end = match.start(1), match.end(1)
    offset = 0
    selected_count = 0
    observed = {selected}
    for record in records:
        token = str(_get(record, "token", ""))
        next_offset = offset + len(token)
        if offset < value_end and next_offset > value_start:
            try:
                value = float(_get(record, "logprob"))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                selected_count += 1
        for alternative in _get(record, "top_logprobs", None) or []:
            candidate = str(_get(alternative, "token", "")).strip()
            if candidate in allowed:
                observed.add(candidate)
        offset = next_offset
    if selected_count == 0:
        return {"supported": False, "evidence_scope": "invalid"}
    scope = "full_taxonomy" if observed >= allowed else "candidate_set"
    return {
        "supported": True,
        "evidence_scope": scope,
        "observed_values": len(observed),
        "allowed_values": len(allowed),
    }


def apply_candidate_scores(
    analysis: LogprobAnalysis,
    candidate_scores: dict[str, float],
) -> LogprobAnalysis:
    """Attach an explicit candidate runner-up when the provider supplied one."""
    if analysis.selected_account_type is None or not candidate_scores:
        return analysis
    selected = analysis.selected_account_type.strip()
    rows = _candidate_score_rows(
        candidate_scores,
        selected=selected,
        allowed_account_types=None,
    )
    alternatives = [
        (str(row["account_type"]), float(row["mean_logprob"]))
        for row in rows
        if str(row["account_type"]).strip() != selected
    ]
    if not alternatives:
        return analysis
    runner_name, runner_score = max(alternatives, key=lambda item: item[1])
    margin = None
    if analysis.selected_mean_logprob is not None:
        margin = analysis.selected_mean_logprob - runner_score
    merged_rows = {str(row["account_type"]): dict(row) for row in analysis.candidate_scores}
    merged_rows.update({str(row["account_type"]): dict(row) for row in rows})
    return LogprobAnalysis(
        status=analysis.status,
        evidence_scope="candidate_set",
        selected_account_type=analysis.selected_account_type,
        selected_loglikelihood=analysis.selected_loglikelihood,
        selected_mean_logprob=analysis.selected_mean_logprob,
        selected_probability_proxy=analysis.selected_probability_proxy,
        runner_up_account_type=runner_name,
        runner_up_loglikelihood=runner_score,
        runner_up_mean_logprob=runner_score,
        logprob_margin=margin,
        token_count=analysis.token_count,
        saturated=analysis.saturated,
        saturated_token_count=analysis.saturated_token_count,
        saturation_reason=analysis.saturation_reason,
        candidate_scores=tuple(merged_rows.values()),
        tokens=analysis.tokens,
    )
