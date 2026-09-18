"""Cached, non-sensitive provider capability probing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any

try:
    from .logprobs import analyze_logprobs
    from .config import CAPABILITY_PROBE_CACHE_VERSION, LLM_CAPABILITY_PROBE_CACHE
except ImportError:
    from logprobs import analyze_logprobs
    from config import CAPABILITY_PROBE_CACHE_VERSION, LLM_CAPABILITY_PROBE_CACHE


_CACHE: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
_CACHE_LOCK = Lock()


def _sdk_version() -> str:
    """Return a non-sensitive SDK version for the probe cache key."""
    try:
        return version("oci-openai")
    except PackageNotFoundError:
        return "unknown"


def _run_probe(identity: tuple[str, str, str, str, str, str], client: Any) -> dict[str, Any]:
    response = client.probe_logprobs()
    analysis = analyze_logprobs(response, allowed_account_types=["Supplies"])
    result: dict[str, Any] = {
        "supported": analysis.status in {"available", "saturated"},
        "selected_token_evidence": analysis.status in {"available", "saturated"},
        "candidate_evidence": bool(analysis.candidate_scores),
        "evidence_scope": analysis.evidence_scope,
        "status": analysis.status,
        "token_count": analysis.token_count,
        "saturated_token_count": analysis.saturated_token_count,
        "provider": getattr(response, "provider", None) or getattr(client, "provider", "unknown"),
        "route": getattr(response, "route", None) or getattr(client, "request_mode", "unknown"),
        "request_mode": getattr(client, "request_mode", "unknown"),
        "model": getattr(response, "model", None) or getattr(client, "model", "unknown"),
        "sdk_version": identity[5],
    }
    compact_probe = getattr(client, "probe_compact_codes", None)
    if compact_probe is not None:
        try:
            compact = compact_probe()
            result["compact_code_supported"] = bool(compact.get("supported"))
            result["compact_code_evidence_scope"] = compact.get("evidence_scope")
        except Exception:
            # Capability probing must never expose provider errors or secrets.
            result["compact_code_supported"] = False
            result["compact_code_evidence_scope"] = "probe_failed"
    else:
        result["compact_code_supported"] = False
        result["compact_code_evidence_scope"] = "not_probed"
    return result


def probe_logprobs_capability(client: Any) -> dict[str, Any]:
    """Probe once per endpoint/model/request mode tuple.

    The client itself is intentionally excluded from the cache key because SDK
    clients are not hashable; the identity tuple is the stable cache boundary.
    """
    identity = (
        CAPABILITY_PROBE_CACHE_VERSION,
        str(getattr(client, "provider", "")),
        str(getattr(client, "service_endpoint", "")),
        str(getattr(client, "model", "")),
        str(getattr(client, "request_mode", "")),
        _sdk_version(),
        str(getattr(client, "logprobs_mode", "")),
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(identity) if LLM_CAPABILITY_PROBE_CACHE else None
    if cached is not None:
        return dict(cached)
    result = _run_probe(identity, client)
    if LLM_CAPABILITY_PROBE_CACHE:
        with _CACHE_LOCK:
            _CACHE[identity] = dict(result)
    return result


def clear_probe_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
