from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import replace
from typing import Any, Callable, Mapping

try:
    from . import db_utils
    from .config import (
        AUTO_CONFIDENCE_THRESHOLD,
        AUTO_DEFAULT_ENABLED,
        AUTO_MIN_ACCEPTED_VALIDATION,
        AUTO_PRECISION_TARGET,
        AUTO_REQUIRE_CALIBRATION,
        CANDIDATE_TOP_K,
        validate_classifier_configuration,
        DEFAULT_TOP_K,
        DENSE_TOP_K,
        IDENTITY_MIN_REAL_POSTS,
        LLM_LOGPROBS_SATURATION_THRESHOLD,
        LLM_LOGPROBS_CALIBRATION_PATH,
        MIN_SIMILARITY_SCORE,
        REVIEW_CONFIDENCE_THRESHOLD,
        RRF_K,
        SPARSE_TOP_K,
        VENDOR_DOMINANCE_THRESHOLD,
        VENDOR_MIN_REAL_POSTS,
    )
    from .embeddings import embed_query
    from .feature_pack import build_invoice_features
    from .llm_client import OCIResponsesClient
    from .capability_probe import probe_logprobs_capability
    from .calibration import load_calibrator
    from .logprobs import analyze_logprobs, apply_candidate_scores
    from .prompt_builder import build_decision_prompt
    from .quality_gate import assess_line_quality
    from .result_contract import apply_result_contract
    from .provider_response import COMPACT_CODE_MAP_VERSION, normalize_provider_response
    from .response_parser import parse_account_type_response
    from .retrieval import apply_soft_boosts, one_case_per_account_type, retrieval_summary
    from .usage import TokenUsage, normalize_provider_usage, summarize_usage
except ImportError:  # Supports running the files directly from this folder.
    import db_utils
    from config import (
        AUTO_CONFIDENCE_THRESHOLD,
        AUTO_DEFAULT_ENABLED,
        AUTO_MIN_ACCEPTED_VALIDATION,
        AUTO_PRECISION_TARGET,
        AUTO_REQUIRE_CALIBRATION,
        CANDIDATE_TOP_K,
        validate_classifier_configuration,
        DEFAULT_TOP_K,
        DENSE_TOP_K,
        IDENTITY_MIN_REAL_POSTS,
        LLM_LOGPROBS_SATURATION_THRESHOLD,
        LLM_LOGPROBS_CALIBRATION_PATH,
        MIN_SIMILARITY_SCORE,
        REVIEW_CONFIDENCE_THRESHOLD,
        RRF_K,
        SPARSE_TOP_K,
        VENDOR_DOMINANCE_THRESHOLD,
        VENDOR_MIN_REAL_POSTS,
    )
    from embeddings import embed_query
    from feature_pack import build_invoice_features
    from llm_client import OCIResponsesClient
    from capability_probe import probe_logprobs_capability
    from calibration import load_calibrator
    from logprobs import analyze_logprobs, apply_candidate_scores
    from prompt_builder import build_decision_prompt
    from quality_gate import assess_line_quality
    from result_contract import apply_result_contract
    from provider_response import COMPACT_CODE_MAP_VERSION, normalize_provider_response
    from response_parser import parse_account_type_response
    from retrieval import apply_soft_boosts, one_case_per_account_type, retrieval_summary
    from usage import TokenUsage, normalize_provider_usage, summarize_usage


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
    "Operating Lease Expense - Electricity": "63613",
    "Project Labor Cost": "59110",
    "Supplies": "60520",
    "Withholding Tax Payable": "25400",
}
TAXONOMY = tuple(ACCOUNT_TYPE_TO_SEGMENT3)
REVIEW_ONLY_TYPES = frozenset({"Asset Clearing", "Miscellaneous"})

PENALTY_WEIGHTS = {
    "identity_conflict": (0.20, "Identity history contains conflicting account types."),
    "interleaved_account_types": (0.10, "Top retrieved cases contain multiple account types."),
    "synthetic_only_evidence": (0.10, "Retrieval evidence is synthetic-only."),
    "short_description": (0.12, "The line description is unusually short."),
    "unmapped_software": (0.40, "Software-like nature has no approved taxonomy mapping."),
    "unsupported_tax": (0.40, "Tax nature remains unsupported by Finance policy."),
    "asset_misc_ambiguity": (0.20, "Asset Clearing and Miscellaneous evidence is ambiguous."),
    "numeric_inconsistency": (0.08, "Amount, unit price, and quantity are inconsistent."),
}


def normalize_account_type(account_type: str) -> str:
    value = unicodedata.normalize("NFKC", str(account_type or ""))
    value = value.replace("\ufffd", "-").replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"\s*[-]\s*", " - ", value)
    return re.sub(r"\s+", " ", value).strip()


def _empty_confidence_breakdown(final_confidence: float = 0.0, *, mode: str = "standard") -> dict[str, Any]:
    return {
        "retrieval_margin": 0.0,
        "retrieval_weight": 0.40,
        "retrieval_contribution": 0.0,
        "identity_strength": 0.0,
        "identity_weight": 0.25,
        "identity_contribution": 0.0,
        "vendor_agreement": 0.0,
        "vendor_weight": 0.20,
        "vendor_contribution": 0.0,
        "llm_signal": 0.0,
        "llm_weight": 0.15,
        "llm_contribution": 0.0,
        "penalties": [],
        "penalty_total": 0.0,
        "unclamped_score": float(final_confidence),
        "final_confidence": float(final_confidence),
        "formula": "clamp(0.40R + 0.25I + 0.20V + 0.15L - penalties, 0, 1)",
        "calculation_mode": mode,
    }


def _empty_logprob_explanation(status: str = "not_called") -> dict[str, Any]:
    evidence_scope = status if status in {"not_called", "missing", "invalid", "saturated"} else "selected_only"
    return {
        "status": status,
        "evidence_scope": evidence_scope,
        "selected_account_type": None,
        "selected_token_count": 0,
        "selected_loglikelihood": None,
        "selected_mean_logprob": None,
        "selected_probability_proxy": None,
        "runner_up_account_type": None,
        "runner_up_mean_logprob": None,
        "logprob_margin": None,
        "calibrated_signal": None,
        "confidence_source": "none",
        "used_in_composite": False,
        "candidate_scores": [],
        "requested_candidate_types": [],
        "tokens": [],
        "calibration_loaded": False,
        "calibration_status": "not_configured",
        "provider": None,
        "request_mode": None,
        "model": None,
        "finish_reason": None,
        "candidate_coverage": None,
        "saturated_token_count": 0,
        "saturation_reason": None,
        "usage": None,
        "usage_summary": None,
        "usage_records": [],
        "llm_call_count": 0,
    }


def _logprob_explanation(llm_result: dict[str, Any] | None) -> dict[str, Any]:
    if not llm_result:
        return _empty_logprob_explanation()
    mean_logprob = llm_result.get("selected_mean_logprob")
    probability_proxy = None
    if mean_logprob is not None:
        try:
            probability_proxy = max(0.0, min(1.0, float(math.exp(float(mean_logprob)))))
        except (TypeError, ValueError, OverflowError):
            probability_proxy = None
    return {
        "status": llm_result.get("logprob_status", "missing"),
        "evidence_scope": llm_result.get("logprob_evidence_scope", "selected_only"),
        "selected_account_type": llm_result.get("account_type"),
        "selected_token_count": llm_result.get("logprob_token_count", 0),
        "selected_loglikelihood": llm_result.get("selected_loglikelihood"),
        "selected_mean_logprob": mean_logprob,
        "selected_probability_proxy": llm_result.get("selected_probability_proxy", probability_proxy),
        "runner_up_account_type": llm_result.get("runner_up_account_type"),
        "runner_up_mean_logprob": llm_result.get("runner_up_mean_logprob"),
        "logprob_margin": llm_result.get("logprob_margin"),
        "calibrated_signal": llm_result.get("logprob_signal"),
        "confidence_source": llm_result.get("confidence_source", "none"),
        "used_in_composite": llm_result.get("confidence_source") == "calibrated_logprob",
        "candidate_scores": list(llm_result.get("logprob_candidate_scores") or []),
        "requested_candidate_types": list(llm_result.get("logprob_candidate_types") or []),
        "tokens": list(llm_result.get("logprob_tokens") or []),
        "calibration_loaded": llm_result.get("confidence_source") == "calibrated_logprob",
        "calibration_status": llm_result.get("calibration_status", "not_configured"),
        "provider": llm_result.get("provider"),
        "request_mode": llm_result.get("request_mode"),
        "model": llm_result.get("model"),
        "finish_reason": llm_result.get("finish_reason"),
        "candidate_coverage": llm_result.get("logprob_candidate_coverage"),
        "saturated_token_count": llm_result.get("saturated_token_count", 0),
        "saturation_reason": llm_result.get("saturation_reason"),
        "usage": llm_result.get("usage"),
        "usage_summary": llm_result.get("usage_summary"),
        "usage_records": list(llm_result.get("usage_records") or []),
        "llm_call_count": llm_result.get("llm_call_count", 0),
    }


def map_account_type_to_segments(account_type: str) -> dict[str, Any] | None:
    normalized_type = normalize_account_type(account_type)
    normalized_mapping = {normalize_account_type(key): value for key, value in ACCOUNT_TYPE_TO_SEGMENT3.items()}
    segment3 = normalized_mapping.get(normalized_type)
    if not segment3:
        return None
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


def _blank_result(feature: dict[str, Any], reason: str, inferred: str | None = None) -> dict[str, Any]:
    return {
        "line_index": feature["line_index"],
        "line_description": feature["line_description"],
        "vendor_name_norm": feature.get("vendor_name_norm"),
        "line_type_norm": feature.get("line_type_norm"),
        "line_amount": feature.get("line_amount"),
        "unit_price": feature.get("unit_price"),
        "quantity_invoiced": feature.get("quantity_invoiced"),
        "account_type": "Unknown",
        "inferred_account_type": inferred,
        "reason": reason,
        "confidence": 0.0,
        "heuristic_confidence": 0.0,
        "calibrated_confidence": None,
        "calibrated_correctness_probability": None,
        "confidence_source": "none",
        "calibration_auto_ready": False,
        "decision_band": "BLANK",
        "decision": "REVIEW",
        "gl_code": None,
        "segment1": None,
        "segment2": None,
        "segment3": None,
        "segment4": None,
        "segment5": None,
        "segment6": None,
        "mapping_source": "unmapped",
        "mapping_row_count": 0,
        "retrieved_cases": [],
        "candidate_types": [],
        "conflict_flags": [],
        "llm_called": False,
        "llm_skipped": True,
        "llm_usage": None,
        "llm_usage_summary": None,
        "llm_usage_records": [],
        "llm_call_count": 0,
        "identity_match": None,
        "vendor_prior": None,
        "confidence_breakdown": _empty_confidence_breakdown(),
        "logprob_explanation": _empty_logprob_explanation(),
        "input_quality": dict(feature.get("input_quality") or {}),
        "explainability_version": "v2",
    }


def _add_mapping(result: dict[str, Any], account_type: str) -> None:
    mapping = map_account_type_to_segments(account_type)
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


def _call_llm(
    client: Any,
    prompt: str,
    allowed_types: list[str],
    line_description: str,
    calibrator: Any | None = None,
) -> dict[str, Any]:
    def parse(raw: Any) -> dict[str, Any]:
        raw = normalize_provider_response(
            raw,
            allowed_account_types=allowed_types,
            provider=getattr(client, "provider", "unknown"),
            route=getattr(client, "request_mode", "structured_json"),
            model=getattr(client, "model", None),
        )
        parsed = parse_account_type_response(raw, line_description, allowed_account_types=allowed_types)
        analysis = analyze_logprobs(
            raw,
            allowed_account_types=allowed_types,
            saturation_threshold=LLM_LOGPROBS_SATURATION_THRESHOLD,
        )
        if (
            getattr(client, "candidate_scoring", False)
            and getattr(client, "candidate_scoring_verified", False)
            and hasattr(client, "score_candidate")
            and analysis.status in {"available", "saturated"}
        ):
            candidate_scores: dict[str, float] = {}
            for candidate in list(dict.fromkeys([*allowed_types, "Unknown"]))[:4]:
                if candidate == analysis.selected_account_type:
                    continue
                try:
                    score = client.score_candidate(prompt, candidate)
                except Exception:
                    score = None
                if score is not None:
                    candidate_scores[candidate] = float(score)
            analysis = apply_candidate_scores(analysis, candidate_scores)
        calibrated = None
        calibration_status = "not_configured"
        calibration_approved = False
        artifact_compatible = False
        if calibrator is not None:
            artifact_compatible = bool(
                getattr(calibrator, "is_compatible", lambda **_: False)(
                    model=getattr(raw, "model", None) or getattr(client, "model", None),
                    route=getattr(raw, "route", None) or getattr(client, "provider", None),
                    request_mode=getattr(client, "request_mode", None),
                    code_map_version=COMPACT_CODE_MAP_VERSION,
                )
            )
            calibration_status = "compatible" if artifact_compatible else "incompatible_artifact"
        if (
            artifact_compatible
            and analysis.status == "available"
            and analysis.evidence_scope in {"candidate_set", "full_taxonomy"}
            and analysis.logprob_margin is not None
        ):
            calibrated = calibrator.score(analysis.selected_mean_logprob, analysis.logprob_margin)
            calibration_status = "loaded_and_used" if calibrated is not None else "loaded_but_insufficient_evidence"
            calibration_approved = bool(getattr(calibrator, "finance_approved", False))
        parsed.update(
            {
                "self_reported_confidence": float(parsed["confidence"]),
                "logprob_signal": calibrated,
                "selected_loglikelihood": analysis.selected_loglikelihood,
                "selected_mean_logprob": analysis.selected_mean_logprob,
                "selected_probability_proxy": analysis.selected_probability_proxy,
                "runner_up_account_type": analysis.runner_up_account_type,
                "runner_up_loglikelihood": analysis.runner_up_loglikelihood,
                "runner_up_mean_logprob": analysis.runner_up_mean_logprob,
                "logprob_margin": analysis.logprob_margin,
                "logprob_status": analysis.status,
                "logprob_evidence_scope": analysis.evidence_scope,
                "logprob_token_count": analysis.token_count,
                "logprob_tokens": list(analysis.tokens),
                "logprob_candidate_scores": list(analysis.candidate_scores),
                "logprob_candidate_types": list(dict.fromkeys([*allowed_types, "Unknown"])),
                "logprob_candidate_coverage": {
                    "observed": len(analysis.candidate_scores),
                    "requested": len(set([*allowed_types, "Unknown"])),
                },
                "saturated_token_count": analysis.saturated_token_count,
                "saturation_reason": analysis.saturation_reason,
                "confidence_source": "calibrated_logprob" if calibrated is not None else "diagnostic_only",
                "calibrated_confidence": calibrated,
                "calibration_status": calibration_status,
                "calibration_approved": calibration_approved,
                "calibration_auto_ready": bool(
                    artifact_compatible
                    and calibrated is not None
                    and getattr(calibrator, "is_auto_ready", lambda **_: False)(
                        target_precision=AUTO_PRECISION_TARGET,
                        minimum_samples=AUTO_MIN_ACCEPTED_VALIDATION,
                    )
                ),
                "calibration_artifact_compatible": artifact_compatible,
                "provider": getattr(raw, "provider", None) or getattr(client, "provider", "unknown"),
                "request_mode": getattr(raw, "route", None) or getattr(client, "request_mode", "structured_json"),
                "model": getattr(raw, "model", None) or getattr(client, "model", None),
                "finish_reason": getattr(raw, "finish_reason", None),
            }
        )
        return parsed

    def is_capability_error(exc: Exception) -> bool:
        text = str(exc).lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return (status in {400, 404, 422} or status is None) and any(
            marker in text for marker in ("logprob", "top_logprobs", "include", "unsupported", "unknown field")
        )

    last_error: Exception | None = None
    result: dict[str, Any] | None = None
    for attempt in range(2):
        try:
            try:
                raw = client.call(prompt, allowed_account_types=allowed_types)
            except TypeError:
                raw = client.call(prompt)
            if getattr(raw, "requires_completion_retry", False):
                retry = getattr(client, "call_native_chat_fallback", None)
                if retry is not None:
                    raw = retry(
                        prompt,
                        allowed_account_types=allowed_types,
                        max_output_tokens=max(512, int(getattr(client, "native_max_output_tokens", 256) * 2)),
                    )
            result = parse(raw)
            if (
                result["logprob_status"] in {"missing", "unsupported", "invalid"}
                and getattr(client, "logprobs_mode", "off") in {"auto", "required"}
            ):
                fallback_errors: list[Exception] = []
                for method_name, provider_name in (
                    ("call_chat_fallback", "oci_chat"),
                    ("call_native_chat_fallback", "oci_native_chat"),
                ):
                    method = getattr(client, method_name, None)
                    if method is None:
                        continue
                    try:
                        fallback_raw = method(prompt, allowed_account_types=allowed_types)
                        fallback_result = parse(fallback_raw)
                        fallback_result["provider"] = getattr(fallback_raw, "provider", None) or provider_name
                        fallback_result["request_mode"] = getattr(fallback_raw, "route", None) or "structured_json"
                        result = fallback_result
                        if result["logprob_status"] in {"available", "saturated"}:
                            break
                    except Exception as fallback_exc:
                        fallback_errors.append(fallback_exc)
                if getattr(client, "logprobs_required", False) and result["logprob_status"] not in {"available"}:
                    detail = fallback_errors[-1] if fallback_errors else "no fallback provider configured"
                    raise ValueError(f"OCI logprobs were unavailable through configured fallbacks: {detail}")
                if not getattr(client, "logprobs_required", False):
                    return result
            if getattr(client, "logprobs_required", False) and result["logprob_status"] not in {"available"}:
                raise ValueError("OCI returned no usable token logprobs for account_type")
            return result
        except Exception as exc:
            if (
                attempt == 0
                and getattr(client, "logprobs_mode", "off") in {"auto", "required"}
                and is_capability_error(exc)
            ):
                fallback_errors: list[Exception] = []
                for method_name, provider_name in (
                    ("call_chat_fallback", "oci_chat"),
                    ("call_native_chat_fallback", "oci_native_chat"),
                ):
                    method = getattr(client, method_name, None)
                    if method is None:
                        continue
                    try:
                        raw = method(prompt, allowed_account_types=allowed_types)
                        result = parse(raw)
                        result["provider"] = getattr(raw, "provider", None) or provider_name
                        result["request_mode"] = getattr(raw, "route", None) or "structured_json"
                        if result["logprob_status"] in {"available", "saturated"}:
                            return result
                    except Exception as fallback_exc:
                        fallback_errors.append(fallback_exc)
                last_error = fallback_errors[-1] if fallback_errors else exc
                if not getattr(client, "logprobs_required", False) and result is not None:
                    return result
                continue
            if isinstance(exc, (ValueError, TypeError)):
                last_error = exc
            else:
                raise
        if attempt == 0:
            prompt += "\nReturn the exact JSON schema and taxonomy values; repair any invalid response."
    raise ValueError("LLM response failed validation after one repair retry") from last_error


def _call_llm_with_usage(
    client: Any,
    prompt: str,
    allowed_types: list[str],
    line_description: str,
    calibrator: Any | None = None,
) -> dict[str, Any]:
    """Call the configured classifier while preserving every provider attempt."""
    usage_records = []
    request_count = 0

    def invoke(method: Any, *args: Any, **kwargs: Any) -> tuple[Any, float, int]:
        nonlocal request_count
        started = time.perf_counter()
        raw = method(*args, **kwargs)
        request_count += 1
        return raw, (time.perf_counter() - started) * 1000.0, request_count

    def normalize(raw: Any, latency_ms: float, attempt: int) -> Any:
        normalized = normalize_provider_response(
            raw,
            allowed_account_types=allowed_types,
            provider=getattr(client, "provider", "unknown"),
            route=getattr(client, "request_mode", "structured_json"),
            model=getattr(client, "model", None),
            request_id=getattr(raw, "request_id", None),
            attempt=attempt,
            latency_ms=latency_ms,
        )
        usage = getattr(normalized, "usage", None)
        if usage is None:
            usage = normalize_provider_usage(
                None,
                model=getattr(normalized, "model", None) or getattr(client, "model", None),
                provider=getattr(normalized, "provider", None) or getattr(client, "provider", None),
                route=getattr(normalized, "route", None) or getattr(client, "request_mode", None),
                request_id=getattr(normalized, "request_id", None),
                attempt=attempt,
                latency_ms=latency_ms,
                finish_reason=getattr(normalized, "finish_reason", None),
            )
        else:
                usage = replace(
                    usage,
                    attempt=attempt,
                    latency_ms=latency_ms if latency_ms is not None else usage.latency_ms,
                    latency=latency_ms if latency_ms is not None else usage.latency,
                )
        normalized = replace(
            normalized,
            usage=usage,
            attempt=attempt,
            latency_ms=latency_ms,
        )
        usage_records.append(usage)
        return normalized

    def attach_usage(result: dict[str, Any]) -> dict[str, Any]:
        result["usage_records"] = [item.as_dict() for item in usage_records]
        result["usage_summary"] = summarize_usage(usage_records).as_dict()
        result["usage"] = usage_records[-1].as_dict() if usage_records else None
        result["llm_call_count"] = len(usage_records)
        return result

    def parse_normalized(raw: Any) -> dict[str, Any]:
        parsed = parse_account_type_response(raw, line_description, allowed_account_types=allowed_types)
        analysis = analyze_logprobs(
            raw,
            allowed_account_types=allowed_types,
            saturation_threshold=LLM_LOGPROBS_SATURATION_THRESHOLD,
        )
        if (
            getattr(client, "candidate_scoring", False)
            and getattr(client, "candidate_scoring_verified", False)
            and hasattr(client, "score_candidate")
            and analysis.status in {"available", "saturated"}
        ):
            candidate_scores: dict[str, float] = {}
            for candidate in list(dict.fromkeys([*allowed_types, "Unknown"]))[:4]:
                if candidate == analysis.selected_account_type:
                    continue
                try:
                    score = client.score_candidate(prompt, candidate)
                except Exception:
                    score = None
                if score is not None:
                    candidate_scores[candidate] = float(score)
            analysis = apply_candidate_scores(analysis, candidate_scores)
        calibrated = None
        calibration_status = "not_configured"
        calibration_approved = False
        artifact_compatible = False
        if calibrator is not None:
            artifact_compatible = bool(
                getattr(calibrator, "is_compatible", lambda **_: False)(
                    model=getattr(raw, "model", None) or getattr(client, "model", None),
                    route=getattr(raw, "route", None) or getattr(client, "provider", None),
                    request_mode=getattr(client, "request_mode", None),
                    code_map_version=COMPACT_CODE_MAP_VERSION,
                )
            )
            calibration_status = "compatible" if artifact_compatible else "incompatible_artifact"
        if (
            artifact_compatible
            and analysis.status == "available"
            and analysis.evidence_scope in {"candidate_set", "full_taxonomy"}
            and analysis.logprob_margin is not None
        ):
            calibrated = calibrator.score(analysis.selected_mean_logprob, analysis.logprob_margin)
            calibration_status = "loaded_and_used" if calibrated is not None else "loaded_but_insufficient_evidence"
            calibration_approved = bool(getattr(calibrator, "finance_approved", False))
        parsed.update(
            {
                "self_reported_confidence": float(parsed["confidence"]),
                "logprob_signal": calibrated,
                "selected_loglikelihood": analysis.selected_loglikelihood,
                "selected_mean_logprob": analysis.selected_mean_logprob,
                "selected_probability_proxy": analysis.selected_probability_proxy,
                "runner_up_account_type": analysis.runner_up_account_type,
                "runner_up_loglikelihood": analysis.runner_up_loglikelihood,
                "runner_up_mean_logprob": analysis.runner_up_mean_logprob,
                "logprob_margin": analysis.logprob_margin,
                "logprob_status": analysis.status,
                "logprob_evidence_scope": analysis.evidence_scope,
                "logprob_token_count": analysis.token_count,
                "logprob_tokens": list(analysis.tokens),
                "logprob_candidate_scores": list(analysis.candidate_scores),
                "logprob_candidate_coverage": {
                    "observed": len(analysis.candidate_scores),
                    "requested": len(set([*allowed_types, "Unknown"])),
                },
                "saturated_token_count": analysis.saturated_token_count,
                "saturation_reason": analysis.saturation_reason,
                "confidence_source": "calibrated_logprob" if calibrated is not None else "diagnostic_only",
                "calibrated_confidence": calibrated,
                "calibration_status": calibration_status,
                "calibration_approved": calibration_approved,
                "calibration_auto_ready": bool(
                    artifact_compatible
                    and calibrated is not None
                    and getattr(calibrator, "is_auto_ready", lambda **_: False)(
                        target_precision=AUTO_PRECISION_TARGET,
                        minimum_samples=AUTO_MIN_ACCEPTED_VALIDATION,
                    )
                ),
                "calibration_artifact_compatible": artifact_compatible,
                "provider": getattr(raw, "provider", None) or getattr(client, "provider", "unknown"),
                "request_mode": getattr(raw, "route", None) or getattr(client, "request_mode", "structured_json"),
                "model": getattr(raw, "model", None) or getattr(client, "model", None),
                "finish_reason": getattr(raw, "finish_reason", None),
                "request_id": getattr(raw, "request_id", None),
                "attempt": getattr(raw, "attempt", 1),
                "latency_ms": getattr(raw, "latency_ms", None),
            }
        )
        return attach_usage(parsed)

    def fallback_methods() -> list[tuple[str, Any]]:
        if "gpt-oss" in str(getattr(client, "model", "")).lower():
            return [("oci_native_chat", getattr(client, "call_native_chat_fallback", None))]
        return [
            ("oci_chat", getattr(client, "call_chat_fallback", None)),
            ("oci_native_chat", getattr(client, "call_native_chat_fallback", None)),
        ]

    def is_capability_error(exc: Exception) -> bool:
        text = str(exc).lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return (status in {400, 404, 422} or status is None) and any(
            marker in text for marker in ("logprob", "top_logprobs", "include", "unsupported", "unknown field")
        )

    last_error: Exception | None = None
    result: dict[str, Any] | None = None
    for attempt in range(2):
        try:
            try:
                raw, latency_ms, call_attempt = invoke(
                    client.call,
                    prompt,
                    allowed_account_types=allowed_types,
                )
            except TypeError:
                raw, latency_ms, call_attempt = invoke(client.call, prompt)
            normalized = normalize(raw, latency_ms, call_attempt)
            if getattr(normalized, "requires_completion_retry", False):
                retry = getattr(client, "call_native_chat_fallback", None)
                if retry is not None:
                    retry_raw, retry_latency, retry_attempt = invoke(
                        retry,
                        prompt,
                        allowed_account_types=allowed_types,
                        max_output_tokens=max(512, int(getattr(client, "native_max_output_tokens", 256) * 2)),
                    )
                    normalized = normalize(retry_raw, retry_latency, retry_attempt)
            result = parse_normalized(normalized)
            if (
                result["logprob_status"] in {"missing", "unsupported", "invalid"}
                and getattr(client, "logprobs_mode", "off") in {"auto", "required"}
            ):
                fallback_errors: list[Exception] = []
                for provider_name, method in fallback_methods():
                    if method is None:
                        continue
                    try:
                        fallback_raw, fallback_latency, fallback_attempt = invoke(
                            method,
                            prompt,
                            allowed_account_types=allowed_types,
                        )
                        fallback_result = parse_normalized(
                            normalize(fallback_raw, fallback_latency, fallback_attempt)
                        )
                        fallback_result["provider"] = getattr(fallback_raw, "provider", None) or provider_name
                        fallback_result["request_mode"] = getattr(fallback_raw, "route", None) or "structured_json"
                        result = attach_usage(fallback_result)
                        if result["logprob_status"] in {"available", "saturated"}:
                            break
                    except Exception as fallback_exc:
                        fallback_errors.append(fallback_exc)
                if getattr(client, "logprobs_required", False) and result["logprob_status"] not in {"available"}:
                    detail = fallback_errors[-1] if fallback_errors else "no fallback provider configured"
                    raise ValueError(f"OCI logprobs were unavailable through configured fallbacks: {detail}")
                if not getattr(client, "logprobs_required", False):
                    return attach_usage(result)
            if getattr(client, "logprobs_required", False) and result["logprob_status"] not in {"available"}:
                raise ValueError("OCI returned no usable token logprobs for account_type")
            return attach_usage(result)
        except Exception as exc:
            if (
                attempt == 0
                and getattr(client, "logprobs_mode", "off") in {"auto", "required"}
                and is_capability_error(exc)
            ):
                fallback_errors: list[Exception] = []
                for provider_name, method in fallback_methods():
                    if method is None:
                        continue
                    try:
                        fallback_raw, fallback_latency, fallback_attempt = invoke(
                            method,
                            prompt,
                            allowed_account_types=allowed_types,
                        )
                        result = parse_normalized(
                            normalize(fallback_raw, fallback_latency, fallback_attempt)
                        )
                        result["provider"] = getattr(fallback_raw, "provider", None) or provider_name
                        result["request_mode"] = getattr(fallback_raw, "route", None) or "structured_json"
                        if result["logprob_status"] in {"available", "saturated"}:
                            return attach_usage(result)
                    except Exception as fallback_exc:
                        fallback_errors.append(fallback_exc)
                last_error = fallback_errors[-1] if fallback_errors else exc
                if not getattr(client, "logprobs_required", False) and result is not None:
                    return attach_usage(result)
                continue
            if isinstance(exc, (ValueError, TypeError)):
                last_error = exc
            else:
                raise
        if attempt == 0:
            prompt += "\nReturn the exact JSON schema and taxonomy values; repair any invalid response."
    raise ValueError("LLM response failed validation after one repair retry") from last_error


_call_llm = _call_llm_with_usage


def calculate_confidence_breakdown(
    summary: dict[str, Any],
    identity: dict[str, Any] | None,
    vendor_prior: dict[str, Any] | None,
    selected_type: str,
    llm_signal: float,
    flags: list[str],
) -> dict[str, Any]:
    retrieval_margin = float(summary.get("retrieval_margin") or 0.0)
    identity_strength = 1.0 if identity and not identity.get("conflict") and identity.get("real_post_count", 0) >= 2 else 0.0
    vendor_agreement = 1.0 if vendor_prior and vendor_prior.get("eligible") and vendor_prior.get("dominant_account_type") == selected_type else 0.0
    retrieval_contribution = 0.40 * retrieval_margin
    identity_contribution = 0.25 * identity_strength
    vendor_contribution = 0.20 * vendor_agreement
    llm_contribution = 0.15 * float(llm_signal)
    penalty_details = [
        {"flag": flag, "weight": PENALTY_WEIGHTS[flag][0], "description": PENALTY_WEIGHTS[flag][1]}
        for flag in flags
        if flag in PENALTY_WEIGHTS
    ]
    penalty_total = sum(item["weight"] for item in penalty_details)
    unclamped_score = retrieval_contribution + identity_contribution + vendor_contribution + llm_contribution - penalty_total
    return {
        "retrieval_margin": retrieval_margin,
        "retrieval_weight": 0.40,
        "retrieval_contribution": retrieval_contribution,
        "identity_strength": identity_strength,
        "identity_weight": 0.25,
        "identity_contribution": identity_contribution,
        "vendor_agreement": vendor_agreement,
        "vendor_weight": 0.20,
        "vendor_contribution": vendor_contribution,
        "llm_signal": float(llm_signal),
        "llm_weight": 0.15,
        "llm_contribution": llm_contribution,
        "penalties": penalty_details,
        "penalty_total": penalty_total,
        "unclamped_score": unclamped_score,
        "final_confidence": max(0.0, min(1.0, unclamped_score)),
        "formula": "clamp(0.40R + 0.25I + 0.20V + 0.15L - penalties, 0, 1)",
        "calculation_mode": "standard",
    }


def _confidence(
    summary: dict[str, Any],
    identity: dict[str, Any] | None,
    vendor_prior: dict[str, Any] | None,
    selected_type: str,
    llm_signal: float,
    flags: list[str],
) -> float:
    """Backward-compatible scalar confidence wrapper."""
    return float(
        calculate_confidence_breakdown(summary, identity, vendor_prior, selected_type, llm_signal, flags)[
            "final_confidence"
        ]
    )


def _decision_band(confidence: float) -> str:
    if confidence >= AUTO_CONFIDENCE_THRESHOLD:
        return "AUTO_ELIGIBLE"
    if confidence >= REVIEW_CONFIDENCE_THRESHOLD:
        return "REVIEW"
    return "BLANK"


def classify_invoice(
    payload: Mapping[str, Any],
    embedder: object,
    settings: object | None = None,
    classification_context: Mapping[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_similarity_score: float = MIN_SIMILARITY_SCORE,
    stage_callback: Callable[[str], None] | None = None,
    db_module: Any = db_utils,
    llm_client_factory: Callable[[], Any] = OCIResponsesClient,
) -> dict[str, Any]:
    """Classify every invoice line using the six-field cascade contract."""
    del settings
    if not 0.0 <= min_similarity_score <= 1.0:
        raise ValueError("min_similarity_score must be between 0.0 and 1.0")
    if not 1 <= top_k <= CANDIDATE_TOP_K:
        raise ValueError(f"top_k must be between 1 and {CANDIDATE_TOP_K}")
    if llm_client_factory is OCIResponsesClient:
        validate_classifier_configuration()
    features = build_invoice_features(payload, classification_context=classification_context)
    calibrator = load_calibrator(LLM_LOGPROBS_CALIBRATION_PATH)
    classification_context = classification_context or {}
    retrieval_dataset_types = classification_context.get("retrieval_dataset_types", ("HISTORY",))
    exclude_invoice_distribution_ids = classification_context.get("exclude_invoice_distribution_ids")
    exclude_source_invoice_distribution_ids = classification_context.get("exclude_source_invoice_distribution_ids")
    exclude_ids = classification_context.get("exclude_history_ids")
    results: list[dict[str, Any]] = []
    default_classifier_client: Any | None = None
    classifier_capability: dict[str, Any] | None = None

    def get_default_classifier_client() -> Any:
        nonlocal default_classifier_client, classifier_capability
        if default_classifier_client is None:
            default_classifier_client = llm_client_factory()
            try:
                classifier_capability = probe_logprobs_capability(default_classifier_client)
            except Exception as exc:
                raise RuntimeError(
                    "Classifier capability/provisioning check failed for the configured OCI endpoint."
                ) from exc
            if not classifier_capability.get("supported"):
                raise RuntimeError(
                    "Classifier capability/provisioning check did not confirm native gpt-oss logprobs."
                )
        return default_classifier_client
    for feature in features:
        if stage_callback:
            stage_callback(f"line_{feature['line_index']}_started")
        description = feature["line_description"]
        input_quality = assess_line_quality(feature)
        feature["input_quality"] = input_quality
        if not description:
            results.append(_blank_result(feature, "LineDescription is empty; Segment 3 remains blank."))
            continue
        if input_quality["blocks_classification"]:
            reasons = ", ".join(input_quality["review_reasons"])
            result = _blank_result(
                feature,
                f"Line cannot be assigned one natural account safely; human review is required ({reasons}).",
            )
            result.update({"decision_band": "REVIEW_REQUIRED", "decision": "REVIEW_REQUIRED"})
            results.append(result)
            continue
        if feature["line_type_norm"] not in {"ITEM", "TAX", "FREIGHT", "MISC", "UNKNOWN"}:
            results.append(
                _blank_result(
                    feature,
                    f"Unsupported LineType '{feature['line_type']}'; Segment 3 remains blank for review.",
                )
            )
            continue

        identity = None
        if hasattr(db_module, "get_identity_match"):
            identity = db_module.get_identity_match(feature["identity_key"])
        vendor_prior = None
        if hasattr(db_module, "get_vendor_prior"):
            vendor_prior = db_module.get_vendor_prior(feature["vendor_name_norm"])
        flags: list[str] = []
        if len(description) < 8:
            flags.append("short_description")
        numeric = feature.get("numeric_consistency") or {}
        if numeric.get("available") and numeric.get("consistent") is False:
            flags.append("numeric_inconsistency")
        if feature["line_type_norm"] == "TAX":
            flags.append("unsupported_tax")
        if identity and identity.get("conflict"):
            flags.append("identity_conflict")

        identity_type = normalize_account_type(identity.get("account_type")) if identity else ""
        if (
            identity
            and not identity.get("conflict")
            and identity.get("real_post_count", 0) >= IDENTITY_MIN_REAL_POSTS
            and identity_type in ACCOUNT_TYPE_TO_SEGMENT3
            and not feature["is_software_like"]
        ):
            selected = identity_type
            result = _blank_result(feature, "")
            identity_breakdown = calculate_confidence_breakdown(
                {},
                identity,
                vendor_prior,
                selected,
                0.0,
                flags,
            )
            identity_breakdown.update(
                {
                    "unclamped_score": 0.99,
                    "final_confidence": 0.99,
                    "calculation_mode": "identity_cache_override",
                    "override_reason": "Consistent real identity history is a deterministic cascade short-circuit.",
                }
            )
            result.update(
                {
                    "account_type": selected,
                    "inferred_account_type": None,
                    "reason": "Exact normalized vendor, description, and line-type identity has consistent real posting history.",
                    "confidence": 0.99,
                    "heuristic_confidence": 0.99,
                    "calibrated_confidence": None,
                    "confidence_source": "deterministic_identity",
                    "calibration_auto_ready": False,
                    "decision_band": "REVIEW"
                    if selected in REVIEW_ONLY_TYPES or not AUTO_DEFAULT_ENABLED or AUTO_REQUIRE_CALIBRATION
                    else "AUTO_ELIGIBLE",
                    "decision": "REVIEW"
                    if selected in REVIEW_ONLY_TYPES or not AUTO_DEFAULT_ENABLED or AUTO_REQUIRE_CALIBRATION
                    else "AUTO_DEFAULT",
                    "llm_called": False,
                    "llm_skipped": True,
                    "identity_match": identity,
                    "vendor_prior": vendor_prior,
                    "vendor_name_norm": feature.get("vendor_name_norm"),
                    "line_type_norm": feature.get("line_type_norm"),
                    "conflict_flags": flags,
                    "confidence_breakdown": identity_breakdown,
                    "explainability_version": "v2",
                }
            )
            _add_mapping(result, selected)
            results.append(result)
            continue

        if stage_callback:
            stage_callback(f"line_{feature['line_index']}_embedding_running")
        query_embedding = embed_query(feature["retrieval_text"], embedder)
        if stage_callback:
            stage_callback(f"line_{feature['line_index']}_retrieval_running")
        if hasattr(db_module, "search_hybrid_history"):
            cases = db_module.search_hybrid_history(
                feature,
                query_embedding,
                dense_top_k=DENSE_TOP_K,
                sparse_top_k=SPARSE_TOP_K,
                rrf_k=RRF_K,
                dataset_types=retrieval_dataset_types,
                exclude_ids=exclude_ids,
                exclude_invoice_distribution_ids=exclude_invoice_distribution_ids,
                exclude_source_invoice_distribution_ids=exclude_source_invoice_distribution_ids,
            )
        else:
            cases = db_module.search_similar_history(
                query_embedding,
                top_k=CANDIDATE_TOP_K,
                target_accuracy=95,
                dataset_types=retrieval_dataset_types,
                exclude_ids=exclude_ids,
                exclude_invoice_distribution_ids=exclude_invoice_distribution_ids,
                exclude_source_invoice_distribution_ids=exclude_source_invoice_distribution_ids,
            )
        cases = list(cases or [])[:CANDIDATE_TOP_K]
        cases = apply_soft_boosts(
            cases,
            vendor_name_norm=feature.get("vendor_name_norm"),
            line_type_norm=feature.get("line_type_norm"),
            vendor_prior=vendor_prior,
        )
        summary = retrieval_summary(cases)
        candidate_types = [normalize_account_type(value) for value in summary["candidate_types"] if value]
        identity_types = [normalize_account_type(value) for value in ((identity or {}).get("account_types") or [])]
        allowed_types = [value for value in dict.fromkeys([*candidate_types, *identity_types]) if value in ACCOUNT_TYPE_TO_SEGMENT3]
        if not allowed_types and vendor_prior and vendor_prior.get("eligible"):
            prior_type = normalize_account_type(vendor_prior.get("dominant_account_type"))
            if prior_type in ACCOUNT_TYPE_TO_SEGMENT3:
                allowed_types = [prior_type]

        top_type = normalize_account_type(summary.get("top_account_type") or "")
        if len(set(candidate_types[:3])) > 1:
            flags.append("interleaved_account_types")
        if cases and all(case.get("is_synthetic") == "Y" for case in cases):
            flags.append("synthetic_only_evidence")
        force_llm = bool(identity and identity.get("conflict")) or feature["is_software_like"] or feature["line_type_norm"] == "TAX" or not cases
        if top_type in REVIEW_ONLY_TYPES and len(set(candidate_types[:3])) > 1:
            flags.append("asset_misc_ambiguity")
            force_llm = True
        vendor_agrees = bool(
            vendor_prior
            and vendor_prior.get("eligible")
            and normalize_account_type(vendor_prior.get("dominant_account_type")) == top_type
        )
        strong_retrieval = bool(summary.get("top3_unanimous")) and bool(top_type) and float(summary.get("retrieval_margin") or 0) >= 0.6
        skip_llm = not force_llm and (strong_retrieval or vendor_agrees) and top_type in ACCOUNT_TYPE_TO_SEGMENT3

        llm_result: dict[str, Any] | None = None
        confidence_source = "retrieval_only"
        calibrated_confidence: float | None = None
        calibration_auto_ready = False
        if skip_llm:
            selected_type = top_type
            reason = "Hybrid retrieval produced a unanimous, well-separated account-type result; LLM was skipped."
            # Retrieval agreement is useful evidence, but it is not a calibrated
            # probability and must never be promoted to an LLM confidence score.
            llm_signal = 0.0
        elif not allowed_types:
            result = _blank_result(feature, "No approved taxonomy precedent was retrieved; Segment 3 remains blank.")
            result.update(
                {
                    "candidate_types": candidate_types,
                    "retrieved_cases": cases,
                    "identity_match": identity,
                    "vendor_prior": vendor_prior,
                    "conflict_flags": flags,
                    "llm_skipped": True,
                }
            )
            results.append(result)
            continue
        else:
            if feature["is_software_like"]:
                flags.append("unmapped_software")
                allowed_types = list(dict.fromkeys([*allowed_types, "Unknown"]))
            llm_cases = one_case_per_account_type(cases, limit=top_k)
            prompt = build_decision_prompt(feature, llm_cases, allowed_types, "qualified" if cases else "weak")
            if stage_callback:
                stage_callback(f"line_{feature['line_index']}_llm_running")
            try:
                client = (
                    get_default_classifier_client()
                    if llm_client_factory is OCIResponsesClient
                    else llm_client_factory()
                )
                llm_result = _call_llm(client, prompt, allowed_types, description, calibrator)
                selected_type = normalize_account_type(llm_result["account_type"])
                reason = llm_result["reason"]
                llm_signal = float(llm_result.get("logprob_signal") or 0.0)
                calibrated_confidence = llm_result.get("calibrated_confidence")
                confidence_source = str(llm_result.get("confidence_source") or "diagnostic_only")
                calibration_auto_ready = bool(llm_result.get("calibration_auto_ready"))
            except ValueError as exc:
                result = _blank_result(feature, f"LLM decision was invalid or unavailable: {exc}")
                result.update(
                    {
                        "candidate_types": candidate_types,
                        "retrieved_cases": cases,
                        "identity_match": identity,
                        "vendor_prior": vendor_prior,
                        "conflict_flags": flags,
                        "llm_called": True,
                        "llm_skipped": False,
                    }
                )
                results.append(result)
                continue
            if feature["is_software_like"]:
                selected_type = "Unknown"
                reason = "Software-like line has no approved software taxonomy mapping; review is required."
                llm_signal = min(llm_signal, 0.40)
            elif feature["line_type_norm"] == "TAX":
                selected_type = "Unknown"
                reason = "Tax line types remain review-only until Finance approves the tax-nature mapping policy."
                llm_signal = min(llm_signal, 0.40)

        if not selected_type or selected_type == "Unknown" or selected_type not in ACCOUNT_TYPE_TO_SEGMENT3:
            result = _blank_result(feature, reason, (llm_result or {}).get("inferred_account_type") if llm_result else None)
            unknown_breakdown = calculate_confidence_breakdown(
                summary,
                identity,
                vendor_prior,
                "Unknown",
                llm_signal,
                flags,
            )
            unknown_breakdown["final_confidence"] = 0.0
            unknown_breakdown["unclamped_score"] = 0.0
            result.update(
                {
                    "confidence": 0.0,
                    "candidate_types": candidate_types,
                    "retrieved_cases": cases,
                    "identity_match": identity,
                    "vendor_prior": vendor_prior,
                    "conflict_flags": flags,
                    "llm_called": llm_result is not None,
                    "llm_skipped": llm_result is None,
                    "llm_usage": (llm_result or {}).get("usage"),
                    "llm_usage_summary": (llm_result or {}).get("usage_summary"),
                    "llm_usage_records": list((llm_result or {}).get("usage_records") or []),
                    "llm_call_count": (llm_result or {}).get("llm_call_count", 0),
                    "logprob_explanation": _logprob_explanation(llm_result),
                    "confidence_breakdown": unknown_breakdown,
                    "logprob_status": (llm_result or {}).get("logprob_status", "not_called"),
                    "logprob_evidence_scope": (llm_result or {}).get("logprob_evidence_scope", "not_called"),
                    "logprob_candidate_scores": (llm_result or {}).get("logprob_candidate_scores", []),
                    "confidence_source": "calibrated_logprob" if calibrated_confidence is not None else "heuristic_composite",
                    "logprob_confidence_source": (llm_result or {}).get("confidence_source", confidence_source),
                    "calibrated_confidence": (llm_result or {}).get("calibrated_confidence", calibrated_confidence),
                    "calibration_auto_ready": (llm_result or {}).get("calibration_auto_ready", calibration_auto_ready),
                    "llm_provider": (llm_result or {}).get("provider"),
                    "llm_request_mode": (llm_result or {}).get("request_mode"),
                    "llm_finish_reason": (llm_result or {}).get("finish_reason"),
                    "llm_request_id": (llm_result or {}).get("request_id"),
                    "llm_attempt": (llm_result or {}).get("attempt", 1),
                    "explainability_version": "v2",
                }
            )
            results.append(result)
            continue

        confidence_breakdown = calculate_confidence_breakdown(
            summary,
            identity,
            vendor_prior,
            selected_type,
            llm_signal,
            flags,
        )
        heuristic_confidence = float(confidence_breakdown["final_confidence"])
        confidence = float(calibrated_confidence) if calibrated_confidence is not None else heuristic_confidence
        band = _decision_band(confidence)
        if band == "AUTO_ELIGIBLE" and not AUTO_DEFAULT_ENABLED:
            band = "REVIEW"
        if band == "AUTO_ELIGIBLE" and AUTO_REQUIRE_CALIBRATION and confidence_source != "calibrated_logprob":
            band = "REVIEW"
        if band == "AUTO_ELIGIBLE" and AUTO_REQUIRE_CALIBRATION and not calibration_auto_ready:
            band = "REVIEW"
        if selected_type in REVIEW_ONLY_TYPES:
            band = "REVIEW"
        result = {
            "line_index": feature["line_index"],
            "line_description": description,
            "account_type": selected_type,
            "inferred_account_type": None,
            "reason": reason,
            "confidence": confidence,
            "heuristic_confidence": heuristic_confidence,
            "calibrated_confidence": calibrated_confidence,
            "decision_band": band,
            "decision": "AUTO_DEFAULT" if band == "AUTO_ELIGIBLE" else band,
            "candidate_types": candidate_types,
            "retrieved_cases": cases,
            "identity_match": identity,
            "vendor_prior": vendor_prior,
            "vendor_name_norm": feature.get("vendor_name_norm"),
            "line_type_norm": feature.get("line_type_norm"),
            "line_amount": feature.get("line_amount"),
            "unit_price": feature.get("unit_price"),
            "quantity_invoiced": feature.get("quantity_invoiced"),
            "conflict_flags": flags,
            "llm_called": llm_result is not None,
            "llm_skipped": llm_result is None,
            "llm_usage": (llm_result or {}).get("usage"),
            "llm_usage_summary": (llm_result or {}).get("usage_summary"),
            "llm_usage_records": list((llm_result or {}).get("usage_records") or []),
            "llm_call_count": (llm_result or {}).get("llm_call_count", 0),
            "self_reported_confidence": (llm_result or {}).get("self_reported_confidence"),
            "logprob_signal": (llm_result or {}).get("logprob_signal"),
            "selected_loglikelihood": (llm_result or {}).get("selected_loglikelihood"),
            "selected_mean_logprob": (llm_result or {}).get("selected_mean_logprob"),
            "runner_up_account_type": (llm_result or {}).get("runner_up_account_type"),
            "runner_up_loglikelihood": (llm_result or {}).get("runner_up_loglikelihood"),
            "runner_up_mean_logprob": (llm_result or {}).get("runner_up_mean_logprob"),
            "logprob_margin": (llm_result or {}).get("logprob_margin"),
            "logprob_status": (llm_result or {}).get("logprob_status", "not_called"),
            "logprob_evidence_scope": (llm_result or {}).get("logprob_evidence_scope", "not_called"),
            "logprob_candidate_scores": (llm_result or {}).get("logprob_candidate_scores", []),
            "confidence_source": "calibrated_logprob" if calibrated_confidence is not None else "heuristic_composite",
            "logprob_confidence_source": (llm_result or {}).get("confidence_source", confidence_source),
            "calibration_auto_ready": (llm_result or {}).get("calibration_auto_ready", calibration_auto_ready),
            "llm_provider": (llm_result or {}).get("provider"),
            "llm_request_mode": (llm_result or {}).get("request_mode"),
            "llm_finish_reason": (llm_result or {}).get("finish_reason"),
            "llm_request_id": (llm_result or {}).get("request_id"),
            "llm_attempt": (llm_result or {}).get("attempt", 1),
            "confidence_breakdown": confidence_breakdown,
            "logprob_explanation": _logprob_explanation(llm_result),
            "explainability_version": "v2",
            "candidate_case_count": len(cases),
            "qualifying_case_count": sum(
                1 for case in cases if float(case.get("similarity_score") or 0.0) >= min_similarity_score
            ),
        }
        if band == "BLANK":
            result.update(
                {
                    "segment1": None,
                    "segment2": None,
                    "segment3": None,
                    "segment4": None,
                    "segment5": None,
                    "segment6": None,
                    "gl_code": None,
                    "mapping_source": "confidence_gate_blank",
                    "mapping_row_count": 0,
                }
            )
        else:
            _add_mapping(result, selected_type)
        results.append(result)
        if stage_callback:
            stage_callback(f"line_{feature['line_index']}_complete")

    results = [
        apply_result_contract(result, input_quality=result.get("input_quality"))
        for result in results
    ]
    usage_records = [
        TokenUsage(**record)
        for line in results
        for record in (line.get("llm_usage_records") or [])
    ]
    return {
        "line_count": len(results),
        "lines": results,
        "approved_fields": ["LineDescription", "LineType", "VendorName/PayeeName", "LineAmount", "UnitPrice", "QuantityInvoiced"],
        "classifier_version": "segment3-cascade-v2-logprob-evidence",
        "classifier_capability": classifier_capability,
        "classification_usage": summarize_usage(usage_records).as_dict(),
    }


def classify_line(
    line_description: str,
    embedder: object,
    settings: object,
    top_k: int = DEFAULT_TOP_K,
    min_similarity_score: float = MIN_SIMILARITY_SCORE,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Backward-compatible single-description CLI wrapper."""
    payload = {"LineItems": [{"LineDescription": line_description}]}
    return classify_invoice(
        payload,
        embedder,
        settings=settings,
        top_k=top_k,
        min_similarity_score=min_similarity_score,
        stage_callback=stage_callback,
    )["lines"][0]
