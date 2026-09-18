"""Finance release gates for a calibrated automatic-default policy."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


DEFAULT_TARGET_PRECISION = 0.98
DEFAULT_MIN_REAL_ACCEPTED = 200


def wilson_lower_bound(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / total) + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - margin)


def evaluate_release_gate(
    records: Iterable[Mapping[str, Any]],
    *,
    finance_approved: bool,
    review_first: bool,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    minimum_real_accepted: int = DEFAULT_MIN_REAL_ACCEPTED,
) -> dict[str, Any]:
    """Assess only approved, real, accepted held-out predictions."""
    accepted = [
        record
        for record in records
        if bool(record.get("accepted")) and not bool(record.get("is_synthetic"))
    ]
    correct = sum(bool(record.get("correct")) for record in accepted)
    count = len(accepted)
    precision = correct / count if count else None
    lower_bound = wilson_lower_bound(correct, count)
    reasons: list[str] = []
    if review_first:
        reasons.append("review_first_policy")
    if not finance_approved:
        reasons.append("finance_approval_required")
    if count < minimum_real_accepted:
        reasons.append("insufficient_real_accepted_evidence")
    if lower_bound < target_precision:
        reasons.append("precision_lower_bound_below_target")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "accepted_real_count": count,
        "accepted_real_correct": correct,
        "accepted_precision": precision,
        "accepted_precision_lower_bound": lower_bound,
        "target_precision": target_precision,
        "minimum_real_accepted": minimum_real_accepted,
    }
