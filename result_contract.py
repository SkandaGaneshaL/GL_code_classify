"""Stable public result fields that do not confuse confidence with accuracy."""

from __future__ import annotations

import math
from typing import Any, Mapping


POLICY_VERSION = "review-first-v1"
_CALIBRATED_SOURCES = frozenset({"calibrated_ranker"})


def _probability(value: Any) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return probability if math.isfinite(probability) and 0.0 <= probability <= 1.0 else None


def apply_result_contract(
    result: Mapping[str, Any], *, input_quality: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Add honest, versioned response fields without discarding legacy fields."""
    output = dict(result)
    source = str(output.get("confidence_source") or "").lower()
    calibrated = _probability(output.get("calibrated_confidence")) if source in _CALIBRATED_SOURCES else None
    logprob_status = str(output.get("logprob_status") or "not_called")
    if calibrated is not None:
        confidence_status = "calibrated"
    elif logprob_status not in {"not_called", "missing", "invalid"} or output.get("confidence_source") not in {None, "none"}:
        confidence_status = "diagnostic_only"
    else:
        confidence_status = "unavailable"

    review_reasons = list((input_quality or output.get("input_quality") or {}).get("review_reasons") or [])
    coa_validation = output.get("coa_validation") or {}
    if coa_validation.get("status") in {"invalid", "unavailable"} and coa_validation.get("reason") == "finance_coa_artifact_required":
        review_reasons.append("active_coa_artifact_required")
        for field in ("segment1", "segment2", "segment3", "segment4", "segment5", "segment6", "gl_code"):
            output[field] = None
        output["mapping_source"] = "finance_artifact_required"
        output["mapping_row_count"] = 0
    if coa_validation.get("status") == "invalid":
        reason = str(coa_validation.get("reason") or "active_coa_validation_failed")
        if reason not in review_reasons:
            review_reasons.append(reason)
        for field in ("segment1", "segment2", "segment3", "segment4", "segment5", "segment6", "gl_code"):
            output[field] = None
        output["mapping_source"] = "coa_validation_rejected"
        output["mapping_row_count"] = 0
    decision_band = str(output.get("decision_band") or "BLANK").upper()
    # This release is review-only. A future approved release gate may change
    # this policy, but no runtime input may bypass it today.
    if coa_validation.get("status") == "invalid":
        review_band = "BLANK"
    elif decision_band == "REVIEW_REQUIRED":
        review_band = "REVIEW_REQUIRED"
    elif output.get("segment3") and str(output.get("account_type") or "Unknown") != "Unknown":
        review_band = "REVIEW"
    else:
        review_band = "BLANK"
    decision = "REVIEW_REQUIRED"

    output.update(
        {
            "prediction": {
                "account_type": output.get("account_type") or "Unknown",
                "segment3": output.get("segment3"),
            },
            "calibrated_correctness_probability": calibrated,
            "confidence_status": confidence_status,
            "review_reasons": review_reasons,
            "decision_band": review_band,
            "policy_version": POLICY_VERSION,
            "model_version": (
                output.get("llm_model")
                or (output.get("logprob_explanation") or {}).get("model")
                or output.get("llm_provider")
                or "unavailable"
            ),
            "auto_post_eligible": False,
            "decision": decision,
        }
    )
    return output
