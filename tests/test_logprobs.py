from types import SimpleNamespace

from logprobs import analyze_logprobs


def _response(tokens, candidate_logprobs=None):
    content = SimpleNamespace(
        text="".join(token for token, _ in tokens),
        logprobs=[SimpleNamespace(token=token, logprob=logprob) for token, logprob in tokens],
    )
    response = SimpleNamespace(
        output_text="".join(token for token, _ in tokens),
        output=[SimpleNamespace(content=[content])],
    )
    if candidate_logprobs is not None:
        response.candidate_logprobs = candidate_logprobs
    return response


def test_scores_only_account_type_value_and_excludes_reason_tokens():
    text = (
        '{"line_description":"paper","account_type":"Supplies",'
        '"inferred_account_type":null,"reason":"office item","confidence":0.9}'
    )
    tokens = [(text[: text.index("Supplies")], -0.2), ("Supplies", -0.1), (text[text.index("Supplies") + 8 :], -0.8)]
    result = analyze_logprobs(
        _response(tokens, {"Supplies": -0.1, "Airfare": -1.3}),
        allowed_account_types=["Supplies", "Airfare"],
    )
    assert result.status == "available"
    assert result.selected_account_type == "Supplies"
    assert result.selected_loglikelihood == -0.1
    assert result.token_count == 1
    assert result.tokens[0]["token"] == "Supplies"
    assert result.tokens[0]["selected"] is True
    assert "reason" not in result.tokens[0]["token"]
    assert result.runner_up_account_type == "Airfare"
    assert result.logprob_margin == 1.2


def test_supports_multi_token_account_type():
    text = (
        '{"account_type":"Operating Lease Expense - Equipment Rental",'
        '"line_description":"rental","inferred_account_type":null,'
        '"reason":"lease","confidence":0.8}'
    )
    label = "Operating Lease Expense - Equipment Rental"
    tokens = [(text[: text.index(label)], -0.2), (label[:18], -0.1), (label[18:], -0.2), (text[text.index(label) + len(label) :], -0.7)]
    result = analyze_logprobs(_response(tokens))
    assert result.status == "available"
    assert result.selected_account_type == label
    assert result.token_count == 2
    assert result.selected_loglikelihood == -0.30000000000000004


def test_missing_logprobs_is_not_zero_probability():
    response = SimpleNamespace(output_text='{"account_type":"Supplies"}')
    result = analyze_logprobs(response, allowed_account_types=["Supplies"])
    assert result.status == "missing"
    assert result.selected_loglikelihood is None
    assert result.logprob_margin is None


def test_top_alternatives_do_not_become_complete_candidate_scores():
    text = '{"account_type":"Project Labor Cost"}'
    prefix = text[: text.index("Project Labor Cost")]
    response = _response(
        [
            (prefix, -0.2),
            ("Project", -0.3),
            (" Labor", -0.2),
            (" Cost", -0.1),
            ('"}', -0.2),
        ]
    )
    response.output[0].content[0].logprobs[1].top_logprobs = [
        SimpleNamespace(token="Unknown", logprob=-5.0),
    ]
    result = analyze_logprobs(
        response,
        allowed_account_types=["Project Labor Cost", "Contractor Expenses"],
    )
    assert result.status == "available"
    assert result.evidence_scope == "selected_only"
    assert result.runner_up_account_type is None
    assert result.candidate_scores == ()


def test_explicit_candidate_scores_are_sanitized_and_exposed():
    text = '{"account_type":"Supplies"}'
    response = _response(
        [
            (text[: text.index("Supplies")], -0.2),
            ("Supplies", -0.2),
            ('"}', -0.2),
        ],
        {"Supplies": -0.2, "Meals": -0.8, "secret": -0.1},
    )
    result = analyze_logprobs(
        response,
        allowed_account_types=["Supplies", "Meals"],
    )
    assert result.evidence_scope == "candidate_set"
    assert {row["account_type"] for row in result.candidate_scores} == {"Supplies", "Meals"}
    assert all("secret" not in row["account_type"] for row in result.candidate_scores)
    assert result.runner_up_account_type == "Meals"


def test_saturated_values_are_marked_for_diagnostics():
    text = '{"account_type":"Supplies"}'
    result = analyze_logprobs(
        _response(
            [
                (text[: text.index("Supplies")], -0.0001),
                ("Supplies", -0.0001),
                (text[text.index("Supplies") + len("Supplies") :], -0.0001),
            ]
        )
    )
    assert result.status == "saturated"
    assert result.saturated is True
