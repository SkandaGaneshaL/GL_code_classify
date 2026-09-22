"""Train a deterministic Segment 3 candidate ranker from Finance-labelled JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from segment3_ranker import calibrate_ranker_artifact, fit_ranker_artifact


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL with candidates and gold_account_type")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranker-version", required=True)
    parser.add_argument("--feature-schema-version", default="segment3-features-v2")
    parser.add_argument("--coa-version", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--calibration-version", required=True)
    parser.add_argument("--model-route", default="segment3_ranker")
    parser.add_argument("--calibration-input", type=Path, help="Held-out real calibration JSONL")
    parser.add_argument("--register", action="store_true", help="Register the artifact in Oracle model registry")
    parser.add_argument("--finance-approved", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    candidates = [row.get("candidates") or row.get("retrieved_candidates") or [] for row in rows]
    gold = [str(row.get("gold_account_type") or row.get("account_type") or "") for row in rows]
    synthetic = [str(row.get("is_synthetic") or "").strip().upper() in {"Y", "YES", "TRUE", "1"} for row in rows]
    artifact = fit_ranker_artifact(
        candidates,
        gold,
        ranker_version=args.ranker_version,
        feature_schema_version=args.feature_schema_version,
        coa_version=args.coa_version,
        dataset_version=args.dataset_version,
        calibration_version=args.calibration_version,
        model_route=args.model_route,
        is_synthetic=synthetic,
    )
    if args.calibration_input:
        calibration_rows = load_jsonl(args.calibration_input)
        calibration_candidates = [row.get("candidates") or row.get("retrieved_candidates") or [] for row in calibration_rows]
        calibration_gold = [str(row.get("gold_account_type") or row.get("account_type") or "") for row in calibration_rows]
        calibration_synthetic = [str(row.get("is_synthetic") or "").strip().upper() in {"Y", "YES", "TRUE", "1"} for row in calibration_rows]
        artifact = calibrate_ranker_artifact(
            artifact,
            calibration_candidates,
            calibration_gold,
            calibration_version=args.calibration_version,
            is_synthetic=calibration_synthetic,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = replace(artifact, finance_approved=args.finance_approved)
    if args.finance_approved and not args.calibration_input:
        raise SystemExit("Finance approval requires an independent real calibration JSONL")
    if args.finance_approved and not artifact.release_ready():
        raise SystemExit("Finance approval requires calibrated evidence, at least 200 accepted rows, and the Wilson target")
    artifact.save(args.output)
    artifact_hash = artifact.artifact_sha256
    if args.register:
        from db_utils import register_model_artifact

        register_model_artifact(
            model_version=artifact.ranker_version,
            artifact_type="segment3_ranker_calibration",
            artifact_hash=artifact_hash,
            dataset_version=artifact.dataset_version,
            finance_approved=artifact.finance_approved,
            metrics={
                "sample_count": artifact.sample_count,
                "feature_schema_version": artifact.feature_schema_version,
                "calibration_version": artifact.calibration_version,
            },
        )
    print(json.dumps({"output": str(args.output), **artifact.to_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
