"""Clerk-controlled split suggestion acceptance helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def accept_split_suggestions(
    suggestions: Mapping[str, Any],
    approved_children: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an explicit review decision without inventing Segment 3 values."""
    if suggestions.get("status") != "split_suggested":
        raise ValueError("Only split_suggested results can be accepted")
    source_children = list(suggestions.get("children") or [])
    approved = []
    for child in approved_children:
        description = str(child.get("description") or "").strip()
        if not description:
            raise ValueError("Approved split children require a description")
        approved.append(
            {
                "description": description,
                "account_nature": child.get("account_nature"),
                "account_type": None,
                "segment3": None,
                "decision": "REVIEW_REQUIRED",
                "amount": child.get("amount"),
                "amount_allocation": "REVIEW_REQUIRED",
            }
        )
    if not approved:
        raise ValueError("At least one approved split child is required")
    return {
        "status": "split_accepted_for_coding",
        "requires_review": True,
        "children": approved,
        "source_child_count": len(source_children),
        "amount_allocation": "REVIEW_REQUIRED",
        "note": "Each child still requires an independent constrained account suggestion.",
    }


def merge_split_suggestions(children: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge clerk-selected children back into one review-only description."""
    values = [str(child.get("description") or "").strip() for child in children]
    values = [value for value in values if value]
    if len(values) < 2:
        raise ValueError("At least two children are required to merge a split")
    return {
        "status": "split_merged_for_review",
        "requires_review": True,
        "description": ", ".join(values),
        "account_type": None,
        "segment3": None,
        "decision": "REVIEW_REQUIRED",
    }


def persist_split_decision(
    db_module: Any,
    *,
    source_line_id: str,
    decision: Mapping[str, Any],
    reviewer_id: str | None = None,
) -> bool:
    """Persist a clerk split decision when the configured audit sink exists."""
    writer = getattr(db_module, "record_split_review_decision", None)
    if not callable(writer):
        return False
    writer(source_line_id=source_line_id, decision=dict(decision), reviewer_id=reviewer_id)
    return True
