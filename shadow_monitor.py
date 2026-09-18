"""Weekly review-first shadow-mode metrics and drift checks."""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any, Iterable, Mapping

try:
    from .drift import population_stability_index
    from .evaluation import _rate_metric
except ImportError:
    from drift import population_stability_index
    from evaluation import _rate_metric


def summarize_shadow_week(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_vendor_distribution: Mapping[str, int] | None = None,
    baseline_line_type_distribution: Mapping[str, int] | None = None,
    week_start: str | date | None = None,
) -> dict[str, Any]:
    """Aggregate clerk-reviewed predictions without enabling auto-posting."""
    rows = [dict(record) for record in records]
    def is_synthetic(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().upper() in {"Y", "YES", "TRUE", "1"}
        return bool(value)

    evaluated = [row for row in rows if not is_synthetic(row.get("is_synthetic"))]
    filled = [row for row in evaluated if row.get("filled") or row.get("segment3")]
    correct = [row for row in filled if bool(row.get("correct"))]
    overrides = [row for row in filled if bool(row.get("override")) or row.get("corrected_account_type")]
    invalid = [row for row in evaluated if bool(row.get("invalid_code"))]
    vendor_counts = Counter(str(row.get("vendor_name") or "UNKNOWN") for row in evaluated)
    line_type_counts = Counter(str(row.get("line_type") or "UNKNOWN") for row in evaluated)
    return {
        "week_start": str(week_start) if week_start is not None else None,
        "evaluated_real_count": len(evaluated),
        "coverage": _rate_metric(len(filled), len(evaluated)),
        "filled_cell_accuracy": _rate_metric(len(correct), len(filled)),
        "override_rate": _rate_metric(len(overrides), len(filled)),
        "invalid_code_rate": _rate_metric(len(invalid), len(evaluated)),
        "vendor_distribution": dict(vendor_counts),
        "line_type_distribution": dict(line_type_counts),
        "vendor_psi": (
            population_stability_index(baseline_vendor_distribution, vendor_counts)
            if baseline_vendor_distribution and vendor_counts
            else None
        ),
        "line_type_psi": (
            population_stability_index(baseline_line_type_distribution, line_type_counts)
            if baseline_line_type_distribution and line_type_counts
            else None
        ),
        "auto_post_enabled": False,
    }
