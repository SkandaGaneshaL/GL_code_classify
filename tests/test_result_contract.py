from result_contract import apply_result_contract


def test_diagnostic_heuristic_is_not_exposed_as_accuracy():
    result = apply_result_contract(
        {
            "account_type": "Supplies",
            "segment3": "60520",
            "confidence": 0.0,
            "heuristic_confidence": 0.0,
            "calibrated_confidence": None,
            "confidence_source": "heuristic_composite",
            "logprob_status": "saturated",
            "decision_band": "REVIEW",
            "reason": "Evidence is mixed.",
        }
    )

    assert result["calibrated_correctness_probability"] is None
    assert result["confidence_status"] == "diagnostic_only"
    assert result["prediction"] == {"account_type": "Supplies", "segment3": "60520"}
    assert result["decision"] == "REVIEW_REQUIRED"
    assert "accuracy" not in result


def test_quality_review_forces_review_even_with_calibrated_probability():
    result = apply_result_contract(
        {
            "account_type": "Supplies",
            "segment3": "60520",
            "calibrated_confidence": 0.98,
            "confidence_source": "calibrated_logprob",
            "decision_band": "AUTO_ELIGIBLE",
            "reason": "Candidate selected.",
        },
        input_quality={"review_reasons": ["missing_vendor"]},
    )

    assert result["confidence_status"] == "calibrated"
    assert result["calibrated_correctness_probability"] == 0.98
    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["review_reasons"] == ["missing_vendor"]
