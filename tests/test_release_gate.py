from release_gate import evaluate_release_gate


def test_review_first_policy_never_promotes_automatic_defaulting():
    gate = evaluate_release_gate(
        [{"correct": True, "is_synthetic": False, "accepted": True}] * 250,
        finance_approved=True,
        review_first=True,
    )

    assert gate["eligible"] is False
    assert gate["reasons"] == ["review_first_policy"]


def test_gate_requires_200_real_accepted_rows_and_wilson_precision():
    gate = evaluate_release_gate(
        [{"correct": True, "is_synthetic": False, "accepted": True}] * 200,
        finance_approved=True,
        review_first=False,
    )

    assert gate["accepted_real_count"] == 200
    assert gate["accepted_precision_lower_bound"] >= 0.98
    assert gate["eligible"] is True


def test_synthetic_rows_do_not_count_toward_release_evidence():
    gate = evaluate_release_gate(
        [{"correct": True, "is_synthetic": True, "accepted": True}] * 250,
        finance_approved=True,
        review_first=False,
    )

    assert gate["accepted_real_count"] == 0
    assert "insufficient_real_accepted_evidence" in gate["reasons"]


def test_excel_n_flag_is_real_release_evidence():
    gate = evaluate_release_gate(
        [{"correct": True, "is_synthetic": "N", "accepted": True}] * 200,
        finance_approved=True,
        review_first=False,
    )
    assert gate["accepted_real_count"] == 200
