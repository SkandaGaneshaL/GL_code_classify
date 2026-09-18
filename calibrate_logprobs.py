"""Fit a versioned Segment 3 logprob calibration artifact from held-out records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibration import fit_logprob_calibrator


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("Calibration input JSON must be an array of objects")
        return value
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on calibration input line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Calibration input line {line_number} is not an object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Held-out prediction JSON or JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Calibration artifact JSON path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint-route", required=True)
    parser.add_argument("--request-mode", required=True)
    parser.add_argument("--code-map-version", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--auto-threshold", type=float, default=0.90)
    parser.add_argument("--finance-approved", action="store_true", help="Record approval; does not bypass readiness checks")
    args = parser.parse_args()

    records = _load_records(args.input)
    calibrator = fit_logprob_calibrator(
        records,
        model=args.model,
        endpoint_route=args.endpoint_route,
        request_mode=args.request_mode,
        code_map_version=args.code_map_version,
        dataset_version=args.dataset_version,
        auto_threshold=args.auto_threshold,
        finance_approved=args.finance_approved,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.output)
    print(json.dumps({
        "output": str(args.output),
        "sample_count": calibrator.sample_count,
        "validation_metrics": calibrator.validation_metrics,
        "auto_ready": calibrator.is_auto_ready(target_precision=0.98, minimum_samples=200),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
