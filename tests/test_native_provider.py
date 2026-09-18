import json
from types import SimpleNamespace

import pytest

from logprobs import analyze_logprobs
from provider_response import normalize_native_chat_response, normalize_provider_response
from response_parser import extract_response_text, parse_account_type_response


def _native_response(*, complete_candidates=True, finish_reason="stop"):
    final_text = json.dumps(
        {
            "line_description": "Office paper",
            "account_type_code": "A",
            "inferred_account_type": None,
            "reason": "Office paper is a supply.",
            "confidence": 0.9,
        },
        separators=(",", ":"),
    )
    reasoning = "I will compare the allowed candidates."
    tokens = ["<|channel|>", "analysis", reasoning, "<|channel|>", "final"]
    values = [0.0, 0.0, -0.2, 0.0, 0.0]
    tops = [{}, {}, {}, {}, {}]
    for index, char in enumerate(final_text):
        tokens.append(char)
        values.append(-0.01 if char == "A" else -0.2)
        if char == "A":
            alternatives = {"A": -0.01, "B": -1.2, "C": -2.0}
            if not complete_candidates:
                alternatives.pop("C")
            tops.append(alternatives)
        else:
            tops.append({})

    message = SimpleNamespace(
        content=[SimpleNamespace(text=final_text)],
        reasoning_content=reasoning,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        logprobs=SimpleNamespace(
            tokens=tokens,
            token_logprobs=values,
            top_logprobs=tops,
        ),
    )
    return SimpleNamespace(
        chat_response=SimpleNamespace(choices=[choice]),
    )


def test_extract_response_text_reads_nested_native_chat_result():
    raw = _native_response()
    text = extract_response_text(raw)
    assert '"account_type_code":"A"' in text


def test_generic_responses_shape_uses_the_same_normalized_envelope():
    text = '{"account_type":"Supplies"}'
    raw = SimpleNamespace(
        output_text=text,
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        logprobs=[
                            SimpleNamespace(token=text[: text.index("Supplies")], logprob=-0.2),
                            SimpleNamespace(token="Supplies", logprob=-0.1),
                            SimpleNamespace(token=text[text.index("Supplies") + 8 :], logprob=-0.2),
                        ]
                    )
                ]
            )
        ],
    )
    normalized = normalize_provider_response(
        raw,
        allowed_account_types=["Supplies"],
        provider="oci_responses",
        route="structured_json",
        model="openai.gpt-4o",
    )
    analysis = analyze_logprobs(normalized, allowed_account_types=["Supplies"])
    assert normalized.provider == "oci_responses"
    assert analysis.status == "available"
    assert analysis.selected_account_type == "Supplies"


def test_native_normalization_excludes_reasoning_and_maps_candidate_codes():
    raw = _native_response()
    normalized = normalize_native_chat_response(
        raw,
        allowed_account_types=["Supplies", "Meals"],
        code_map={"A": "Supplies", "B": "Meals", "C": "Unknown"},
        provider="oci_native_chat",
        model="openai.gpt-oss-20b",
    )

    assert normalized.text.startswith('{"line_description":"Office paper"')
    assert normalized.selected_account_type == "Supplies"
    assert normalized.finish_reason == "stop"
    assert normalized.reasoning_text == "I will compare the allowed candidates."
    assert normalized.selected_tokens[0].token == "A"
    assert {"Supplies", "Meals", "Unknown"} <= set(normalized.candidate_logprobs)

    analysis = analyze_logprobs(normalized, allowed_account_types=["Supplies", "Meals"])
    assert analysis.status == "available"
    assert analysis.evidence_scope == "candidate_set"
    assert analysis.runner_up_account_type == "Meals"
    assert all("analysis" not in token["token"] for token in analysis.tokens)


def test_native_normalization_keeps_partial_candidates_selected_only():
    normalized = normalize_native_chat_response(
        _native_response(complete_candidates=False),
        allowed_account_types=["Supplies", "Meals"],
        code_map={"A": "Supplies", "B": "Meals", "C": "Unknown"},
        provider="oci_native_chat",
        model="openai.gpt-oss-20b",
    )
    analysis = analyze_logprobs(normalized, allowed_account_types=["Supplies", "Meals"])
    assert analysis.status == "available"
    assert analysis.evidence_scope == "selected_only"
    assert analysis.runner_up_account_type is None


def test_native_normalization_preserves_canonical_five_field_contract():
    normalized = normalize_native_chat_response(
        _native_response(),
        allowed_account_types=["Supplies", "Meals"],
        code_map={"A": "Supplies", "B": "Meals", "C": "Unknown"},
        provider="oci_native_chat",
        model="openai.gpt-oss-20b",
    )
    parsed = parse_account_type_response(
        normalized,
        "Office paper",
        allowed_account_types=["Supplies", "Meals"],
    )
    assert set(parsed) == {
        "line_description",
        "account_type",
        "inferred_account_type",
        "reason",
        "confidence",
    }
    assert parsed["account_type"] == "Supplies"


def test_native_length_response_is_rejected_for_retry():
    raw = _native_response(finish_reason="length")
    normalized = normalize_native_chat_response(
        raw,
        allowed_account_types=["Supplies", "Meals"],
        code_map={"A": "Supplies", "B": "Meals", "C": "Unknown"},
        provider="oci_native_chat",
        model="openai.gpt-oss-20b",
    )
    assert normalized.finish_reason == "length"
    assert normalized.requires_completion_retry is True


@pytest.mark.parametrize("value", [None, "", "not-json"])
def test_native_empty_text_is_rejected(value):
    raw = SimpleNamespace(
        chat_response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=[SimpleNamespace(text=value)]),
                    finish_reason="stop",
                    logprobs=None,
                )
            ]
        )
    )
    if value in (None, ""):
        with pytest.raises(ValueError):
            normalize_native_chat_response(
                raw,
                allowed_account_types=["Supplies"],
                code_map={"A": "Supplies", "B": "Unknown"},
            )
