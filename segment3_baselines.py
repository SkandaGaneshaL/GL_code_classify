"""Leakage-safe baselines used before promoting a Segment 3 ranker."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

try:
    from .evaluation import EvaluationRow, evaluate_predictions
    from .segment3_taxonomy import ACCOUNT_TYPE_TO_SEGMENT3, normalize_account_type
except ImportError:
    from evaluation import EvaluationRow, evaluate_predictions
    from segment3_taxonomy import ACCOUNT_TYPE_TO_SEGMENT3, normalize_account_type


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _prediction(account_type: str | None, *, decision_band: str = "REVIEW") -> dict[str, Any]:
    normalized = normalize_account_type(account_type or "Unknown")
    return {
        "account_type": normalized,
        "segment3": ACCOUNT_TYPE_TO_SEGMENT3.get(normalized),
        "decision_band": decision_band if normalized in ACCOUNT_TYPE_TO_SEGMENT3 else "BLANK",
        "confidence_source": "baseline",
    }


def vendor_majority_predictions(
    rows: Sequence[EvaluationRow], history_rows: Sequence[EvaluationRow]
) -> list[dict[str, Any]]:
    """Predict the dominant posted account for each vendor, then global mode."""
    by_vendor: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in history_rows:
        if row.is_synthetic:
            continue
        label = normalize_account_type(row.account_type)
        global_counts[label] += 1
        if row.vendor_name:
            by_vendor[_norm(row.vendor_name)][label] += 1
    fallback = global_counts.most_common(1)[0][0] if global_counts else "Unknown"
    predictions = []
    for row in rows:
        counts = by_vendor.get(_norm(row.vendor_name), Counter()) if row.vendor_name else Counter()
        predictions.append(_prediction(counts.most_common(1)[0][0] if counts else fallback))
    return predictions


def exact_identity_predictions(
    rows: Sequence[EvaluationRow], history_rows: Sequence[EvaluationRow]
) -> list[dict[str, Any]]:
    """Use only exact vendor/description/line-type matches from HISTORY."""
    identities: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in history_rows:
        if row.is_synthetic:
            continue
        key = (_norm(row.vendor_name), _norm(row.description), _norm(row.line_type))
        identities[key][normalize_account_type(row.account_type)] += 1
    predictions = []
    for row in rows:
        key = (_norm(row.vendor_name), _norm(row.description), _norm(row.line_type))
        counts = identities.get(key, Counter())
        predictions.append(_prediction(counts.most_common(1)[0][0] if counts else "Unknown"))
    return predictions


def retrieval_predictions(
    rows: Sequence[EvaluationRow], candidates_by_row: Sequence[Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Use the first valid retrieved account type without LLM calls."""
    if len(rows) != len(candidates_by_row):
        raise ValueError("rows and candidates_by_row must have equal length")
    predictions = []
    for candidates in candidates_by_row:
        selected = next(
            (
                normalize_account_type(str(candidate.get("account_type") or "Unknown"))
                for candidate in candidates
                if normalize_account_type(str(candidate.get("account_type") or "Unknown"))
                in ACCOUNT_TYPE_TO_SEGMENT3
            ),
            "Unknown",
        )
        predictions.append(_prediction(selected))
    return predictions


def compare_baselines(
    rows: Sequence[EvaluationRow],
    predictions_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return the same metric contract for every baseline."""
    return {
        name: evaluate_predictions(list(rows), list(predictions))
        for name, predictions in predictions_by_name.items()
    }

