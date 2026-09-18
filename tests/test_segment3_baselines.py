from evaluation import EvaluationRow
from segment3_baselines import exact_identity_predictions, vendor_majority_predictions


def test_vendor_majority_uses_history_only():
    history = [EvaluationRow("h1", "paper", "Supplies", False, vendor_name="Acme", line_type="ITEM")]
    rows = [EvaluationRow("t1", "paper", "Supplies", False, vendor_name="Acme", line_type="ITEM")]
    prediction = vendor_majority_predictions(rows, history)[0]
    assert prediction["account_type"] == "Supplies"
    assert prediction["segment3"] == "60520"


def test_exact_identity_abstains_when_history_has_no_match():
    history = [EvaluationRow("h1", "paper", "Supplies", False, vendor_name="Acme", line_type="ITEM")]
    rows = [EvaluationRow("t1", "paper", "Meals", False, vendor_name="Other", line_type="ITEM")]
    assert exact_identity_predictions(rows, history)[0]["segment3"] is None
