"""Provider-neutral token usage accounting for Segment 3 classification.

The OCI Responses and native Chat APIs expose different usage object shapes.
This module normalizes the common counters without fabricating values when a
provider omits a category.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


USAGE_NORMALIZER_VERSION = "gl-provider-usage-v1"


def _get(value: Any, name: str) -> Any:
    if value is None:
        return None
    aliases = (name, name.split("_")[0] + "".join(part.title() for part in name.split("_")[1:]))
    if isinstance(value, dict):
        for candidate in aliases:
            if candidate in value:
                return value[candidate]
        return None
    for candidate in aliases:
        result = getattr(value, candidate, None)
        if result is not None:
            return result
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            serialized = to_dict()
        except Exception:
            return None
        if isinstance(serialized, dict):
            for candidate in aliases:
                if candidate in serialized:
                    return serialized[candidate]
    return None


def _token_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _field_names(value: Any, names: Iterable[str]) -> list[str]:
    return [name for name in names if _get(value, name) is not None]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    reported: bool = False
    model: str | None = None
    provider: str | None = None
    route: str | None = None
    request_id: str | None = None
    call_type: str = "classification"
    attempt: int = 1
    finish_reason: str | None = None
    latency_ms: float | None = None
    latency: float | None = None
    usage_location: str | None = None
    reasoning_tokens_status: str = "provider_unavailable"
    reasoning_tokens_reported: bool = False
    usage_fields_present: tuple[str, ...] = ()
    nested_detail_fields_present: tuple[str, ...] = ()
    output_tokens_semantics: str = "provider_reported_output_total"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["usage_fields_present"] = list(self.usage_fields_present)
        value["nested_detail_fields_present"] = list(self.nested_detail_fields_present)
        return value


@dataclass
class UsageSummary:
    calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    reported_calls: int = 0
    unknown_calls: int = 0
    missing_categories: dict[str, int] = field(default_factory=dict)
    reasoning_tokens_reported: bool = False
    reasoning_tokens_status: str = "not_applicable"
    output_tokens_semantics: str = "provider_reported_output_total"
    total_latency_ms: float | None = None
    model: str | None = None
    route: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    attempt: int | None = None

    def add_call(self, usage: TokenUsage) -> None:
        self.calls += 1
        self.reported_calls += int(usage.reported)
        self.unknown_calls += int(not usage.reported)
        self.reasoning_tokens_reported = self.reasoning_tokens_reported or usage.reasoning_tokens_reported
        if usage.model:
            self.model = usage.model if self.model in {None, usage.model} else "multiple"
        if usage.route:
            self.route = usage.route if self.route in {None, usage.route} else "multiple"
        if usage.request_id:
            self.request_id = usage.request_id
        if usage.finish_reason:
            self.finish_reason = usage.finish_reason
        self.attempt = max(self.attempt or 0, usage.attempt)
        if usage.output_tokens_semantics == "may_include_reasoning_tokens":
            self.output_tokens_semantics = "may_include_reasoning_tokens"
        if usage.reasoning_tokens_reported:
            self.reasoning_tokens_status = "reported"
        elif usage.reasoning_tokens_status in {"provider_unavailable", "unsupported_response_shape"}:
            if self.reasoning_tokens_status != "reported":
                self.reasoning_tokens_status = usage.reasoning_tokens_status
        observed_latency = usage.latency_ms if usage.latency_ms is not None else usage.latency
        if observed_latency is not None:
            self.total_latency_ms = (self.total_latency_ms or 0.0) + observed_latency
        for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens"):
            value = getattr(usage, name)
            if value is None:
                self.missing_categories[name] = self.missing_categories.get(name, 0) + 1
                setattr(self, name, None)
            elif not self.missing_categories.get(name):
                current = getattr(self, name)
                setattr(self, name, value if current is None else current + value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "reported_calls": self.reported_calls,
            "unknown_calls": self.unknown_calls,
            "missing_categories": dict(self.missing_categories),
            "reasoning_tokens_reported": self.reasoning_tokens_reported,
            "reasoning_tokens_status": self.reasoning_tokens_status,
            "output_tokens_semantics": self.output_tokens_semantics,
            "total_latency_ms": self.total_latency_ms,
            "model": self.model,
            "route": self.route,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "attempt": self.attempt,
            "latency": self.total_latency_ms,
        }


def summarize_usage(usages: Iterable[TokenUsage | None]) -> UsageSummary:
    result = UsageSummary()
    for usage in usages:
        if usage is not None:
            result.add_call(usage)
    return result


def normalize_provider_usage(
    usage: Any,
    *,
    model: str | None = None,
    provider: str | None = None,
    route: str | None = None,
    request_id: str | None = None,
    call_type: str = "classification",
    attempt: int = 1,
    finish_reason: str | None = None,
    latency_ms: float | None = None,
    usage_location: str | None = None,
) -> TokenUsage:
    """Normalize OCI-native and OpenAI-compatible usage without guessing."""
    input_tokens = _token_value(_first_not_none(_get(usage, "input_tokens"), _get(usage, "prompt_tokens")))
    output_tokens = _token_value(_first_not_none(_get(usage, "output_tokens"), _get(usage, "completion_tokens")))
    total_tokens = _token_value(_get(usage, "total_tokens"))
    input_details = _first_not_none(_get(usage, "input_tokens_details"), _get(usage, "prompt_tokens_details"))
    output_details = _first_not_none(_get(usage, "output_tokens_details"), _get(usage, "completion_tokens_details"))
    cached_tokens = _token_value(_first_not_none(
        _get(usage, "cached_tokens"),
        _get(input_details, "cached_tokens"),
        _get(input_details, "cache_read_input_tokens"),
    ))

    reasoning_sources = (
        ("reasoning_tokens", _get(usage, "reasoning_tokens")),
        ("output_tokens_details.reasoning_tokens", _get(output_details, "reasoning_tokens")),
    )
    reasoning_tokens = None
    reasoning_location = None
    reasoning_present = False
    for location, value in reasoning_sources:
        reasoning_present = reasoning_present or value is not None
        numeric = _token_value(value)
        if numeric is not None:
            reasoning_tokens = numeric
            reasoning_location = location
            break
    if reasoning_tokens is not None:
        reasoning_status = "reported"
    elif reasoning_present:
        reasoning_status = "unsupported_response_shape"
    else:
        reasoning_status = "provider_unavailable"

    present = _field_names(usage, (
        "input_tokens", "prompt_tokens", "output_tokens", "completion_tokens",
        "total_tokens", "cached_tokens", "reasoning_tokens",
        "input_tokens_details", "prompt_tokens_details",
        "output_tokens_details", "completion_tokens_details",
    ))
    nested = _field_names(input_details, ("cached_tokens", "cache_read_input_tokens"))
    nested += _field_names(output_details, ("reasoning_tokens",))
    values = (input_tokens, output_tokens, cached_tokens, reasoning_tokens, total_tokens)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        reported=usage is not None and any(value is not None for value in values),
        model=model,
        provider=provider,
        route=route,
        request_id=request_id,
        call_type=call_type,
        attempt=attempt,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        latency=latency_ms,
        usage_location=usage_location,
        reasoning_tokens_status=reasoning_status,
        reasoning_tokens_reported=reasoning_tokens is not None,
        usage_fields_present=tuple(present),
        nested_detail_fields_present=tuple(nested),
        output_tokens_semantics=(
            "may_include_reasoning_tokens"
            if model and "gpt-oss" in model.lower()
            else "provider_reported_output_total"
        ),
    )
