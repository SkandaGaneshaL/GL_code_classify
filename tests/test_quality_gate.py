from quality_gate import assess_line_quality


def test_compound_account_natures_require_review():
    assessment = assess_line_quality(
        {
            "line_description": (
                "Office workstation capital asset acquisition, Miami office equipment "
                "lease charges, and consultant payment withholding tax adjustment."
            ),
            "vendor_name_norm": "",
            "line_type_norm": "UNKNOWN",
        }
    )

    assert assessment["decision"] == "REVIEW_REQUIRED"
    assert assessment["blocks_classification"] is True
    assert {"compound_account_nature", "missing_vendor", "missing_line_type"} <= set(
        assessment["review_reasons"]
    )


def test_atomic_complete_line_can_continue_to_classifier():
    assessment = assess_line_quality(
        {
            "line_description": "Printer toner cartridges",
            "vendor_name_norm": "ACME OFFICE SUPPLIES",
            "line_type_norm": "ITEM",
        }
    )

    assert assessment["decision"] == "CONTINUE"
    assert assessment["blocks_classification"] is False
    assert assessment["llm_eligible"] is True
    assert assessment["singleton_account_eligible"] is True
    assert assessment["segment3_output_eligible"] is True
    assert assessment["review_reasons"] == []
    assert assessment["quality_status"] == "sufficient"


def test_missing_context_requires_review_but_does_not_claim_ambiguity():
    assessment = assess_line_quality(
        {
            "line_description": "Printer toner cartridges",
            "vendor_name_norm": "",
            "line_type_norm": "UNKNOWN",
        }
    )

    assert assessment["decision"] == "REVIEW_REQUIRED"
    assert assessment["blocks_classification"] is True
    assert assessment["review_reasons"] == ["missing_vendor", "missing_line_type"]


def test_low_extraction_quality_blocks_classification():
    assessment = assess_line_quality(
        {
            "line_description": "Printer toner",
            "vendor_name_norm": "acme",
            "line_type_norm": "ITEM",
            "extraction_quality": 0.2,
        }
    )
    assert assessment["blocks_classification"] is True
    assert "low_extraction_quality" in assessment["review_reasons"]
