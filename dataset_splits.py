"""Leakage-safe, temporal split construction for Finance-labelled history."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
from typing import Any, Iterable, Mapping


def _parse_date(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Real evaluation rows require invoice_date")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid invoice_date: {text!r}") from exc


def split_real_rows_by_time(
    rows: Iterable[Mapping[str, Any]],
    *,
    development_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Split real labels chronologically while keeping every source group intact."""
    if not 0 < development_fraction < 1 or not 0 < calibration_fraction < 1 or development_fraction + calibration_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a locked test fraction")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_dates: dict[str, datetime] = {}
    for source in rows:
        synthetic = source.get("is_synthetic")
        if isinstance(synthetic, str):
            synthetic = synthetic.strip().upper() in {"Y", "YES", "TRUE", "1"}
        if bool(synthetic):
            continue
        row = dict(source)
        group = str(row.get("source_group_id") or "").strip()
        if not group:
            raise ValueError("Real evaluation rows require source_group_id")
        row_date = _parse_date(row.get("invoice_date"))
        groups[group].append(row)
        group_dates[group] = min(group_dates.get(group, row_date), row_date)
    ordered_groups = sorted(groups, key=lambda group: (group_dates[group], group))
    if len(ordered_groups) < 3:
        raise ValueError("At least three real source groups are required for development, calibration, and test splits")
    development_count = max(1, int(len(ordered_groups) * development_fraction))
    calibration_count = max(1, int(len(ordered_groups) * calibration_fraction))
    if development_count + calibration_count >= len(ordered_groups):
        calibration_count = len(ordered_groups) - development_count - 1
    boundaries = {
        "development": ordered_groups[:development_count],
        "calibration": ordered_groups[development_count : development_count + calibration_count],
        "test": ordered_groups[development_count + calibration_count :],
    }
    return {name: [row for group in selected for row in groups[group]] for name, selected in boundaries.items()}


def build_dataset_splits(
    rows: Iterable[Mapping[str, Any]],
    *,
    development_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
    dataset_version: str = "unversioned",
) -> dict[str, Any]:
    """Build immutable chronological splits plus a reproducibility manifest."""
    splits = split_real_rows_by_time(
        rows,
        development_fraction=development_fraction,
        calibration_fraction=calibration_fraction,
    )
    groups = {
        name: sorted({str(row.get("source_group_id")) for row in values})
        for name, values in splits.items()
    }
    payload = {
        "dataset_version": dataset_version,
        "development_fraction": development_fraction,
        "calibration_fraction": calibration_fraction,
        "groups": groups,
        "row_ids": {name: sorted(str(row.get("row_id") or row.get("invoice_distribution_id")) for row in values) for name, values in splits.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return {**splits, "manifest": payload}
