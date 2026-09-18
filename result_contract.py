"""Stable public result fields that do not confuse confidence with accuracy."""

from __future__ import annotations

import math
from typing import Any, Mapping


POLICY_VERSION = "review-first-v1"


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
    calibrated = _probability(output.get("calibrated_confidence"))
    logprob_status = str(output.get("logprob_status") or "not_called")
    if calibrated is not None and output.get("confidence_source") == "calibrated_logprob":
        confidence_status = "calibrated"
    elif logprob_status not in {"not_called", "missing", "invalid"} or output.get("confidence_source") not in {None, "none"}:
        confidence_status = "diagnostic_only"
    else:
        confidence_status = "unavailable"

    review_reasons = list((input_quality or output.get("input_quality") or {}).get("review_reasons") or [])
    decision_band = str(output.get("decision_band") or "BLANK").upper()
    if review_reasons or decision_band != "AUTO_ELIGIBLE":
        decision = "REVIEW_REQUIRED"
    else:
        # AUTO_ELIGIBLE is retained for a future Finance-approved policy.
        decision = "AUTO_ELIGIBLE"

    output.update(
        {
            "prediction": {
                "account_type": output.get("account_type") or "Unknown",
                "segment3": output.get("segment3"),
            },
            "calibrated_correctness_probability": calibrated,
            "confidence_status": confidence_status,
            "review_reasons": review_reasons,
            "policy_version": POLICY_VERSION,
            "model_version": (
                output.get("llm_model")
                or (output.get("logprob_explanation") or {}).get("model")
                or output.get("llm_provider")
                or "unavailable"
            ),
            "decision": decision,
        }
    )
    return output
