from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _case_key(case: dict[str, Any]) -> str:
    if case.get("id") is not None:
        return str(case["id"])
    return "|".join(
        str(case.get(key) or "")
        for key in ("line_description", "account_type", "invoice_num")
    )


def reciprocal_rank_fusion(
    dense_cases: Iterable[dict[str, Any]],
    sparse_cases: Iterable[dict[str, Any]],
    k: int = 60,
    missing_rank: int = 16,
) -> list[dict[str, Any]]:
    if k <= 0 or missing_rank <= 0:
        raise ValueError("RRF parameters must be positive")
    merged: dict[str, dict[str, Any]] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for channel, cases in (("dense", dense_cases), ("sparse", sparse_cases)):
        for rank, case in enumerate(cases, start=1):
            key = _case_key(case)
            merged.setdefault(key, {}).update(case)
            ranks[key][channel] = rank
    for key, case in merged.items():
        dense_rank = ranks[key].get("dense", missing_rank)
        sparse_rank = ranks[key].get("sparse", missing_rank)
        case["dense_rank"] = ranks[key].get("dense")
        case["sparse_rank"] = ranks[key].get("sparse")
        case["rrf_score"] = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)


def one_case_per_account_type(cases: Iterable[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        account_type = str(case.get("account_type") or "Unknown")
        if account_type in seen:
            continue
        seen.add(account_type)
        selected.append(case)
        if len(selected) >= limit:
            break
    return selected


def retrieval_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    types = [str(case.get("account_type") or "Unknown") for case in cases]
    unique_types = list(dict.fromkeys(types))
    top3_types = types[:3]
    top = cases[0] if cases else {}
    second = next((case for case in cases if case.get("account_type") != top.get("account_type")), None)
    top_score = float(top.get("rrf_score") or 0.0)
    second_score = float(second.get("rrf_score") or 0.0) if second else 0.0
    margin = 1.0 if len(set(types[:3])) == 1 and types else 0.2
    if second and top_score > 0:
        gap = max(0.0, min(1.0, (top_score - second_score) / top_score))
        margin = max(margin, 0.6 if gap >= 0.12 else 0.2)
    return {
        "top_account_type": top.get("account_type"),
        "candidate_types": unique_types,
        "top3_types": top3_types,
        "top3_unanimous": bool(top3_types) and len(set(top3_types)) == 1,
        "retrieval_margin": margin,
        "top_score": top_score,
        "second_score": second_score,
    }


def apply_soft_boosts(
    cases: Iterable[dict[str, Any]],
    *,
    vendor_name_norm: str | None = None,
    line_type_norm: str | None = None,
    vendor_prior: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply small ranking boosts without excluding cross-vendor/type evidence."""
    vendor_enabled = bool(vendor_name_norm) and not (
        vendor_prior and float(vendor_prior.get("entropy") or 0.0) > 0.90
    )
    ranked = []
    for case in cases:
        item = dict(case)
        boost = 0.0
        if vendor_enabled and case.get("vendor_name_norm") == vendor_name_norm:
            boost += 0.002
        if line_type_norm and case.get("line_type_norm") == line_type_norm:
            boost += 0.001
        item["soft_boost"] = boost
        item["rrf_score"] = float(case.get("rrf_score") or 0.0) + boost
        ranked.append(item)
    return sorted(ranked, key=lambda item: item.get("rrf_score", 0.0), reverse=True)
