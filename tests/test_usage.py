from types import SimpleNamespace

from provider_response import normalize_native_chat_response
from usage import normalize_provider_usage, summarize_usage


def test_normalizes_native_nested_usage_and_preserves_reasoning_semantics():
    usage = normalize_provider_usage(
        SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130,
            prompt_tokens_details=SimpleNamespace(cached_tokens=12),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
        ),
        model="openai.gpt-oss-20b",
        provider="oci_native_chat",
        route="oci_native_chat",
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 30
    assert usage.cached_tokens == 12
    assert usage.reasoning_tokens == 8
    assert usage.total_tokens == 130
    assert usage.output_tokens_semantics == "may_include_reasoning_tokens"
    assert usage.reasoning_tokens_status == "reported"


def test_missing_categories_are_not_fabricated_as_zero():
    usage = normalize_provider_usage(
        {"input_tokens": 10, "total_tokens": 10},
        model="openai.gpt-oss-20b",
    )
    summary = summarize_usage([usage])
    assert summary.input_tokens == 10
    assert summary.output_tokens is None
    assert summary.reasoning_tokens is None
    assert summary.missing_categories["output_tokens"] == 1


def test_nested_chat_result_usage_is_attached_to_normalized_response():
    text = '{"line_description":"paper","account_type":"Supplies","inferred_account_type":null,"reason":"supply","confidence":0.9}'
    raw = SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=[SimpleNamespace(text=text)]),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=22, completion_tokens=11, total_tokens=33),
            )
        )
    )
    normalized = normalize_native_chat_response(
        raw,
        allowed_account_types=["Supplies"],
        model="openai.gpt-oss-20b",
    )
    assert normalized.text == text
    assert normalized.usage is not None
    assert normalized.usage.input_tokens == 22
    assert normalized.usage.output_tokens == 11
    assert normalized.usage.total_tokens == 33
