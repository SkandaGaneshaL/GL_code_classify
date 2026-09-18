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

    assert assessment == {
        "decision": "CONTINUE",
        "blocks_classification": False,
        "review_reasons": [],
        "quality_status": "sufficient",
    }


def test_missing_context_requires_review_but_does_not_claim_ambiguity():
    assessment = assess_line_quality(
        {
            "line_description": "Printer toner cartridges",
            "vendor_name_norm": "",
            "line_type_norm": "UNKNOWN",
        }
    )

    assert assessment["decision"] == "REVIEW_REQUIRED"
    assert assessment["blocks_classification"] is False
    assert assessment["review_reasons"] == ["missing_vendor", "missing_line_type"]
