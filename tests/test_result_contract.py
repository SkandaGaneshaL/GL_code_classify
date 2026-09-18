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
    assert result["decision_band"] == "REVIEW"
    assert "accuracy" not in result


def test_logprob_calibration_remains_diagnostic_even_when_value_exists():
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

    assert result["confidence_status"] == "diagnostic_only"
    assert result["calibrated_correctness_probability"] is None
    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["review_reasons"] == ["missing_vendor"]


def test_only_calibrated_sources_expose_a_correctness_probability():
    calibrated = apply_result_contract(
        {
            "account_type": "Supplies",
            "segment3": "60520",
            "calibrated_confidence": 0.82,
            "confidence_source": "calibrated_ranker",
            "decision_band": "REVIEW",
        }
    )
    diagnostic = apply_result_contract(
        {
            "account_type": "Supplies",
            "segment3": "60520",
            "calibrated_confidence": 0.99,
            "confidence_source": "heuristic_composite",
            "decision_band": "REVIEW",
        }
    )

    assert calibrated["calibrated_correctness_probability"] == 0.82
    assert calibrated["confidence_status"] == "calibrated"
    assert diagnostic["calibrated_correctness_probability"] is None
    assert diagnostic["confidence_status"] == "diagnostic_only"


def test_review_first_policy_never_returns_auto_eligible_decision():
    result = apply_result_contract(
        {
            "account_type": "Supplies",
            "segment3": "60520",
            "calibrated_confidence": 0.99,
            "confidence_source": "calibrated_ranker",
            "decision_band": "AUTO_ELIGIBLE",
        }
    )

    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["decision_band"] == "REVIEW"


def test_invalid_active_coa_mapping_is_blank_and_review_required():
    result = apply_result_contract(
        {
            "account_type": "Supplies",
            "segment3": "99999",
            "gl_code": "101.10.99999.000.000.000",
            "coa_validation": {"status": "invalid", "reason": "segment3_not_in_active_coa"},
            "decision_band": "REVIEW",
        }
    )
    assert result["segment3"] is None
    assert result["decision_band"] == "BLANK"
    assert result["decision"] == "REVIEW_REQUIRED"
