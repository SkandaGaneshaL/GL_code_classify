"""Dependency-free drift checks for daily production monitoring."""

from __future__ import annotations

import math
from typing import Mapping


def population_stability_index(
    baseline: Mapping[str, int | float], current: Mapping[str, int | float], *, epsilon: float = 1e-6
) -> float:
    """Calculate PSI over categorical distributions; safe for new categories."""
    baseline_total = sum(max(0.0, float(value)) for value in baseline.values())
    current_total = sum(max(0.0, float(value)) for value in current.values())
    if baseline_total <= 0 or current_total <= 0:
        raise ValueError("PSI requires non-empty baseline and current distributions")
    score = 0.0
    for category in set(baseline) | set(current):
        expected = max(epsilon, max(0.0, float(baseline.get(category, 0))) / baseline_total)
        observed = max(epsilon, max(0.0, float(current.get(category, 0))) / current_total)
        score += (observed - expected) * math.log(observed / expected)
    return score


def drift_status(psi: float, *, review_threshold: float = 0.10, block_threshold: float = 0.25) -> str:
    if psi >= block_threshold:
        return "block_auto_defaulting"
    if psi >= review_threshold:
        return "review_required"
    return "stable"
