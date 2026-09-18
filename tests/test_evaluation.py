import pytest
from pathlib import Path

from evaluation import (
    EvaluationRow,
    confusion_matrix,
    evaluate_predictions,
    format_percentage,
    validate_no_cross_split_leakage,
    load_evaluation_rows,
)


def test_evaluation_reports_real_synthetic_and_recall_metrics():
    rows = [
        EvaluationRow("r1", "paper", "Supplies", False),
        EvaluationRow("s1", "hotel", "Hotel / Accomodation", True),
    ]
    predictions = [
        {"account_type": "Supplies", "segment3": "60520", "decision_band": "REVIEW", "llm_skipped": True},
        {"account_type": "Unknown", "segment3": None, "decision_band": "BLANK", "llm_skipped": False},
    ]
    metrics = evaluate_predictions(
        rows,
        predictions,
        retrieval_candidates=[[{"account_type": "Supplies"}], [{"account_type": "Hotel / Accomodation"}]],
        latencies_seconds=[1.0, 2.0],
        token_counts=[10, 20],
    )
    assert metrics["real_filled_cell_accuracy"] == 1.0
    assert metrics["synthetic_filled_cell_accuracy"] == 0.0
    assert metrics["retrieval_type_recall_at_3"] == 1.0
    assert metrics["latency_p95_seconds"] == 1.95
    assert metrics["token_count_total"] == 30


def test_confusion_matrix_contains_all_taxonomy_labels():
    matrix = confusion_matrix(["Supplies"], ["Supplies"])
    assert matrix["Supplies"]["Supplies"] == 1
    assert "Airfare" in matrix


def test_strict_accuracy_counts_abstentions_and_reports_denominators():
    rows = [
        EvaluationRow("r1", "paper", "Supplies", False),
        EvaluationRow("r2", "meal", "Meals", False),
        EvaluationRow("r3", "hotel", "Hotel / Accomodation", False),
    ]
    predictions = [
        {"account_type": "Supplies", "segment3": "60520", "decision_band": "AUTO_ELIGIBLE"},
        {"account_type": "Airfare", "segment3": "60512", "decision_band": "REVIEW"},
        {"account_type": "Unknown", "segment3": None, "decision_band": "BLANK"},
    ]
    metrics = evaluate_predictions(rows, predictions)
    assert metrics["strict_accuracy"]["correct"] == 1
    assert metrics["strict_accuracy"]["total"] == 3
    assert metrics["strict_accuracy"]["value"] == pytest.approx(1 / 3)
    assert metrics["strict_accuracy"]["percent"] == "33.33%"
    assert metrics["selective_accuracy"]["correct"] == 1
    assert metrics["selective_accuracy"]["total"] == 2
    assert metrics["coverage"]["percent"] == "66.67%"
    assert metrics["auto_post_precision"]["percent"] == "100.00%"
    assert metrics["abstain_rate"]["percent"] == "33.33%"


def test_calibration_metrics_ignore_non_calibrated_confidence():
    rows = [
        EvaluationRow("r1", "paper", "Supplies", False),
        EvaluationRow("r2", "meal", "Meals", False),
    ]
    predictions = [
        {"account_type": "Supplies", "segment3": "60520", "confidence": 0.99, "confidence_source": "diagnostic_only"},
        {"account_type": "Meals", "segment3": "60521", "calibrated_confidence": 0.8, "confidence_source": "calibrated_logprob"},
    ]
    metrics = evaluate_predictions(rows, predictions)
    assert metrics["calibration_metrics"]["evidence_count"] == 1
    assert metrics["calibration_metrics"]["brier_score"] == pytest.approx(0.04)


def test_split_group_collision_is_rejected():
    history = [EvaluationRow("h1", "paper", "Supplies", False, source_group_id="invoice-1")]
    test = [EvaluationRow("t1", "paper paraphrase", "Supplies", True, source_group_id="invoice-1")]
    with pytest.raises(ValueError, match="crosses dataset splits"):
        validate_no_cross_split_leakage(history, test)


def test_percentage_formatter_is_explicit_and_honest():
    assert format_percentage(0.98) == "98.00%"
    assert format_percentage(None) == "Unavailable"


def test_loader_defaults_to_frozen_test_split():
    workbook = Path(__file__).parents[1] / "gl_account_history_poc_120.xlsx"
    rows = load_evaluation_rows(workbook)
    assert len(rows) == 30
    assert {row.dataset_type for row in rows} == {"TEST"}
    assert len({row.row_id for row in rows}) == len(rows)


def test_evaluation_reports_ranked_retrieval_and_input_quality_metrics():
    rows = [
        EvaluationRow("r1", "paper", "Supplies", False),
        EvaluationRow("r2", "hotel", "Hotel / Accomodation", False),
    ]
    predictions = [
        {
            "account_type": "Supplies",
            "segment3": "60520",
            "decision_band": "REVIEW",
            "input_quality": {"quality_status": "sufficient", "review_reasons": []},
        },
        {
            "account_type": "Unknown",
            "segment3": None,
            "decision_band": "REVIEW_REQUIRED",
            "input_quality": {
                "quality_status": "insufficient",
                "review_reasons": ["missing_vendor", "compound_account_nature"],
            },
        },
    ]
    metrics = evaluate_predictions(
        rows,
        predictions,
        retrieval_candidates=[
            [{"account_type": "Meals"}, {"account_type": "Supplies"}],
            [{"account_type": "Hotel / Accomodation"}],
        ],
    )

    assert metrics["retrieval_mrr"] == pytest.approx(0.75)
    assert metrics["retrieval_ndcg_at_3"] == pytest.approx((1 / 1.584962500721156 + 1) / 2)
    assert metrics["input_quality_metrics"]["insufficient_count"] == 1
    assert metrics["input_quality_metrics"]["review_reason_counts"]["missing_vendor"] == 1
