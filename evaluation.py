from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

try:
    from .classification import ACCOUNT_TYPE_TO_SEGMENT3, normalize_account_type
    from .release_gate import evaluate_release_gate
except ImportError:  # Supports running the file directly from this folder.
    from classification import ACCOUNT_TYPE_TO_SEGMENT3, normalize_account_type
    from release_gate import evaluate_release_gate


TAXONOMY = tuple(ACCOUNT_TYPE_TO_SEGMENT3)
ABSTENTION_TYPES = frozenset({"", "Unknown", "UNKNOWN", "None", "NULL"})
AUTO_BANDS = frozenset({"AUTO", "AUTO_DEFAULT", "AUTO_ELIGIBLE"})
METRIC_CONTRACT_VERSION = "segment3-evaluation-v2"


@dataclass(frozen=True)
class EvaluationRow:
    row_id: str
    description: str
    account_type: str
    is_synthetic: bool
    vendor_name: str | None = None
    line_type: str | None = None
    dataset_type: str = "TEST"
    source_group_id: str | None = None
    source_invoice_distribution_id: str | None = None


def load_evaluation_rows(
    excel_path: Path,
    sheet_name: str = "POC Dataset",
    dataset_types: Iterable[str] | None = ("TEST",),
) -> list[EvaluationRow]:
    """Load labeled rows for evaluation without mixing HISTORY into TEST."""
    frame = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=object, keep_default_na=False)
    required = {"INVOICE_DISTRIBUTION_ID", "LINE_DESCRIPTION", "SEGMENT3_DESCRIPTION", "IS_SYNTHETIC"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation worksheet is missing columns: {missing}")
    requested_types = None if dataset_types is None else {str(value).strip().upper() for value in dataset_types}
    rows: list[EvaluationRow] = []
    seen_ids: set[str] = set()
    for index, values in frame.iterrows():
        description = str(values["LINE_DESCRIPTION"]).strip()
        gold = normalize_account_type(str(values["SEGMENT3_DESCRIPTION"]).strip())
        if not description or not gold or gold == "Unknown":
            continue
        dataset_type = str(values.get("DATASET_TYPE") or "").strip().upper() or "HISTORY"
        if requested_types is not None and dataset_type not in requested_types:
            continue
        row_id = str(values["INVOICE_DISTRIBUTION_ID"] or index).strip()
        if row_id in seen_ids:
            raise ValueError(f"Evaluation worksheet contains duplicate row id: {row_id}")
        seen_ids.add(row_id)
        source_invoice_distribution_id = str(values.get("SOURCE_INVOICE_DISTRIBUTION_ID") or "").strip() or None
        # Use the distribution lineage as the split group. For real rows the
        # row itself is the group; synthetic paraphrases point back to it.
        source_group_id = source_invoice_distribution_id or row_id
        rows.append(
            EvaluationRow(
                row_id=row_id,
                description=description,
                account_type=gold,
                is_synthetic=str(values["IS_SYNTHETIC"]).upper() == "Y",
                vendor_name=str(values.get("VENDOR_NAME_NORM") or values.get("VENDOR_NAME") or "").strip() or None,
                line_type=str(values.get("LINE_TYPE") or "").strip() or None,
                dataset_type=dataset_type,
                source_group_id=source_group_id,
                source_invoice_distribution_id=source_invoice_distribution_id,
            )
        )
    validate_split_groups(rows)
    return rows


def validate_split_groups(rows: Sequence[EvaluationRow]) -> None:
    """Reject duplicate row IDs inside one evaluation split."""
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Evaluation rows contain duplicate row IDs")


def validate_no_cross_split_leakage(*splits: Sequence[EvaluationRow]) -> None:
    """Fail when a source group appears in more than one evaluation split."""
    seen: dict[str, int] = {}
    for split_index, rows in enumerate(splits):
        for row in rows:
            group = row.source_group_id or row.row_id
            previous = seen.get(group)
            if previous is not None and previous != split_index:
                raise ValueError(f"Evaluation source group {group!r} crosses dataset splits")
            seen[group] = split_index


def confusion_matrix(
    expected: Iterable[str], predicted: Iterable[str], labels: Iterable[str] = TAXONOMY
) -> dict[str, dict[str, int]]:
    labels = list(labels)
    matrix = {actual: {guess: 0 for guess in [*labels, "Unknown"]} for actual in labels}
    for actual, guess in zip(expected, predicted):
        actual = normalize_account_type(actual)
        guess = normalize_account_type(guess) if guess else "Unknown"
        matrix.setdefault(actual, {label: 0 for label in [*labels, "Unknown"]})
        matrix[actual][guess] = matrix[actual].get(guess, 0) + 1
    return matrix


def format_percentage(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "Unavailable"
    return f"{float(value) * 100:.{digits}f}%"


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = correct / total
    denominator = 1.0 + (z * z / total)
    centre = (proportion + (z * z / (2.0 * total))) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _rate_metric(correct: int, total: int) -> dict[str, Any]:
    value = correct / total if total else None
    interval = _wilson_interval(correct, total)
    return {
        "correct": int(correct),
        "total": int(total),
        "value": value,
        "percent": format_percentage(value),
        "ci95": None
        if interval is None
        else {"lower": interval[0], "upper": interval[1], "percent": f"{interval[0] * 100:.2f}%–{interval[1] * 100:.2f}%"},
    }


def _is_filled(prediction: Mapping[str, Any], normalized_type: str) -> bool:
    return normalized_type not in ABSTENTION_TYPES and bool(prediction.get("segment3"))


def _is_auto(prediction: Mapping[str, Any]) -> bool:
    return str(prediction.get("decision_band") or prediction.get("decision") or "").upper() in AUTO_BANDS


def _classification_metrics(rows: Sequence[EvaluationRow], predictions: Sequence[str]) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    supported: list[dict[str, Any]] = []
    weighted_f1_numerator = 0.0
    total_support = 0
    for label in TAXONOMY:
        tp = sum(row.account_type == label and guess == label for row, guess in zip(rows, predictions))
        support = sum(row.account_type == label for row in rows)
        predicted = sum(guess == label for guess in predictions)
        fn = support - tp
        fp = predicted - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics = {
            "support": support,
            "predicted": predicted,
            "correct": tp,
            "precision": precision,
            "precision_percent": format_percentage(precision),
            "recall": recall,
            "recall_percent": format_percentage(recall),
            "f1": f1,
            "f1_percent": format_percentage(f1),
            "false_positive": fp,
            "false_negative": fn,
        }
        per_class[label] = metrics
        if support:
            supported.append(metrics)
            weighted_f1_numerator += f1 * support
            total_support += support
    macro_precision = sum(item["precision"] for item in supported) / len(supported) if supported else 0.0
    macro_recall = sum(item["recall"] for item in supported) / len(supported) if supported else 0.0
    macro_f1 = sum(item["f1"] for item in supported) / len(supported) if supported else 0.0
    weighted_f1 = weighted_f1_numerator / total_support if total_support else 0.0
    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_precision_percent": format_percentage(macro_precision),
        "macro_recall": macro_recall,
        "macro_recall_percent": format_percentage(macro_recall),
        "macro_f1": macro_f1,
        "macro_f1_percent": format_percentage(macro_f1),
        "weighted_f1": weighted_f1,
        "weighted_f1_percent": format_percentage(weighted_f1),
        "balanced_accuracy": macro_recall,
        "balanced_accuracy_percent": format_percentage(macro_recall),
    }


def _calibrated_confidence(prediction: Mapping[str, Any]) -> float | None:
    source = str(prediction.get("confidence_source") or "").lower()
    if source not in {"calibrated_logprob", "calibrated", "calibrated_probability"}:
        return None
    value = prediction.get("calibrated_confidence")
    if value is None:
        value = prediction.get("logprob_signal")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _calibration_metrics(rows: Sequence[EvaluationRow], predictions: Sequence[Mapping[str, Any]], normalized: Sequence[str]) -> dict[str, Any]:
    scored: list[tuple[float, int]] = []
    for row, item, guess in zip(rows, predictions, normalized):
        confidence = _calibrated_confidence(item)
        if confidence is None or not _is_filled(item, guess):
            continue
        scored.append((confidence, int(guess == row.account_type)))
    if not scored:
        return {"evidence_count": 0, "brier_score": None, "log_loss": None, "ece": None, "mce": None, "reliability_bins": [], "risk_coverage": [], "aurc": None}

    brier = sum((confidence - correct) ** 2 for confidence, correct in scored) / len(scored)
    log_loss = sum(-math.log(max(1e-15, confidence if correct else 1.0 - confidence)) for confidence, correct in scored) / len(scored)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for bucket in range(10):
        lower = bucket / 10.0
        upper = 1.0 if bucket == 9 else (bucket + 1) / 10.0
        values = [(confidence, correct) for confidence, correct in scored if lower <= confidence <= upper and (bucket == 9 or confidence < upper)]
        if not values:
            continue
        mean_confidence = sum(value[0] for value in values) / len(values)
        observed_accuracy = sum(value[1] for value in values) / len(values)
        gap = abs(mean_confidence - observed_accuracy)
        ece += len(values) / len(scored) * gap
        mce = max(mce, gap)
        bins.append({"lower": lower, "upper": upper, "count": len(values), "mean_confidence": mean_confidence, "accuracy": observed_accuracy, "gap": gap})

    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    risk_coverage: list[dict[str, Any]] = []
    errors = 0
    for index, (_, correct) in enumerate(ordered, start=1):
        errors += 1 - correct
        risk_coverage.append({"accepted": index, "coverage": index / len(rows), "risk": errors / index})
    aurc = sum(point["risk"] for point in risk_coverage) / len(rows)
    return {"evidence_count": len(scored), "brier_score": brier, "log_loss": log_loss, "ece": ece, "mce": mce, "reliability_bins": bins, "risk_coverage": risk_coverage, "aurc": aurc}


def _input_quality_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    insufficient_count = 0
    reason_counts: dict[str, int] = {}
    for prediction in predictions:
        quality = prediction.get("input_quality") or {}
        if quality.get("quality_status") == "insufficient":
            insufficient_count += 1
        for reason in quality.get("review_reasons") or prediction.get("review_reasons") or []:
            text = str(reason).strip()
            if text:
                reason_counts[text] = reason_counts.get(text, 0) + 1
    return {
        "insufficient_count": insufficient_count,
        "insufficient_rate": insufficient_count / len(predictions) if predictions else 0.0,
        "review_reason_counts": reason_counts,
    }


def _retrieval_ranking_metrics(rows: Sequence[EvaluationRow], candidates_by_row: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for row, candidates in zip(rows, candidates_by_row):
        rank = next(
            (
                index
                for index, candidate in enumerate(candidates[:3], start=1)
                if normalize_account_type(str(candidate.get("account_type") or "Unknown")) == row.account_type
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank else 0.0)
    denominator = len(rows)
    return {
        "retrieval_mrr": sum(reciprocal_ranks) / denominator if denominator else 0.0,
        "retrieval_ndcg_at_3": sum(ndcgs) / denominator if denominator else 0.0,
    }


def evaluate_predictions(
    rows: list[EvaluationRow],
    predictions: list[Mapping[str, Any]],
    retrieval_candidates: list[list[Mapping[str, Any]]] | None = None,
    latencies_seconds: list[float] | None = None,
    token_counts: list[int] | None = None,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have equal length")
    normalized_predictions = [normalize_account_type(str(item.get("account_type") or "Unknown")) for item in predictions]
    total = len(rows)
    filled_indices = [index for index, value in enumerate(normalized_predictions) if _is_filled(predictions[index], value)]
    correct_indices = [index for index in filled_indices if normalized_predictions[index] == rows[index].account_type]
    auto_indices = [index for index, item in enumerate(predictions) if _is_auto(item) and _is_filled(item, normalized_predictions[index])]
    auto_correct = [index for index in auto_indices if normalized_predictions[index] == rows[index].account_type]
    real_indices = [index for index, row in enumerate(rows) if not row.is_synthetic]
    synthetic_indices = [index for index, row in enumerate(rows) if row.is_synthetic]
    real_filled_indices = [index for index in filled_indices if index in real_indices]
    synthetic_filled_indices = [index for index in filled_indices if index in synthetic_indices]
    real_correct_indices = [index for index in correct_indices if index in real_indices]
    synthetic_correct_indices = [index for index in correct_indices if index in synthetic_indices]
    review_count = sum(str(item.get("decision_band") or "").upper() == "REVIEW" for item in predictions)
    abstain_count = total - len(filled_indices)
    review_or_blank_count = sum(
        str(item.get("decision_band") or "").upper() in {"REVIEW", "BLANK"}
        or index not in filled_indices
        for index, item in enumerate(predictions)
    )
    skip_count = sum(bool(item.get("llm_skipped")) for item in predictions)

    result: dict[str, Any] = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "row_count": total,
        "real_row_count": len(real_indices),
        "synthetic_row_count": len(synthetic_indices),
        "strict_accuracy": _rate_metric(len(correct_indices), total),
        "selective_accuracy": _rate_metric(len(correct_indices), len(filled_indices)),
        "coverage": _rate_metric(len(filled_indices), total),
        "auto_post_precision": _rate_metric(len(auto_correct), len(auto_indices)),
        "auto_post_recall": _rate_metric(len(auto_correct), total),
        "review_rate": _rate_metric(review_count, total),
        "abstain_rate": _rate_metric(abstain_count, total),
        "filled_cell_accuracy": len(correct_indices) / len(filled_indices) if filled_indices else 0.0,
        "filled_cell_accuracy_metric": _rate_metric(len(correct_indices), len(filled_indices)),
        "filled_cell_count": len(filled_indices),
        "review_or_blank_rate": review_or_blank_count / total if total else 0.0,
        "llm_skip_rate": skip_count / total if total else 0.0,
        "real_filled_cell_accuracy": (
            len(real_correct_indices) / len(real_filled_indices)
            if real_filled_indices
            else 0.0
        ),
        "synthetic_filled_cell_accuracy": (
            len(synthetic_correct_indices) / len(synthetic_filled_indices)
            if synthetic_filled_indices
            else 0.0
        ),
        "real_strict_accuracy": _rate_metric(len(real_correct_indices), len(real_indices)),
        "synthetic_strict_accuracy": _rate_metric(len(synthetic_correct_indices), len(synthetic_indices)),
        "real_selective_accuracy": _rate_metric(len(real_correct_indices), len(real_filled_indices)),
        "synthetic_selective_accuracy": _rate_metric(len(synthetic_correct_indices), len(synthetic_filled_indices)),
        "classification_metrics": _classification_metrics(rows, normalized_predictions),
        "calibration_metrics": _calibration_metrics(rows, predictions, normalized_predictions),
        "input_quality_metrics": _input_quality_metrics(predictions),
        "confusion_matrix": confusion_matrix((row.account_type for row in rows), normalized_predictions),
    }
    if retrieval_candidates is not None:
        recall_hits = 0
        for row, candidates in zip(rows, retrieval_candidates):
            top_types = {normalize_account_type(str(case.get("account_type") or "Unknown")) for case in candidates[:3]}
            recall_hits += row.account_type in top_types
        result["retrieval_type_recall_at_3"] = recall_hits / total if total else 0.0
        result["retrieval_type_recall_at_3_metric"] = _rate_metric(recall_hits, total)
        result.update(_retrieval_ranking_metrics(rows, retrieval_candidates))
    if latencies_seconds is not None and latencies_seconds:
        ordered = sorted(latencies_seconds)
        result["latency_p50_seconds"] = _percentile(ordered, 0.50)
        result["latency_p95_seconds"] = _percentile(ordered, 0.95)
        result["latency_p99_seconds"] = _percentile(ordered, 0.99)
    if token_counts is not None and token_counts:
        result["token_count_total"] = sum(token_counts)
        result["token_count_average"] = sum(token_counts) / len(token_counts)
    return result


def _invoke_evaluation_classifier(classifier: Callable[..., Mapping[str, Any]], row: EvaluationRow) -> Mapping[str, Any]:
    """Pass an evaluation exclusion context when the injected classifier supports it."""
    try:
        parameters = inspect.signature(classifier).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "classification_context" in parameters:
        excluded_distribution_ids = tuple(
            value
            for value in (row.row_id, row.source_invoice_distribution_id)
            if value
        )
        return classifier(
            row,
            classification_context={
                "retrieval_dataset_types": ("HISTORY",),
                "exclude_invoice_distribution_ids": excluded_distribution_ids,
                "exclude_source_invoice_distribution_ids": (
                    (row.source_invoice_distribution_id,)
                    if row.source_invoice_distribution_id
                    else ()
                ),
            },
        )
    return classifier(row)


def run_evaluation(rows: list[EvaluationRow], classifier: Callable[[EvaluationRow], Mapping[str, Any]], name: str) -> dict[str, Any]:
    predictions: list[Mapping[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        started = time.perf_counter()
        predictions.append(_invoke_evaluation_classifier(classifier, row))
        latencies.append(time.perf_counter() - started)
    metrics = evaluate_predictions(rows, predictions, latencies_seconds=latencies)
    return {"name": name, "metrics": metrics, "predictions": predictions}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
