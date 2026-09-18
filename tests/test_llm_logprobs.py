from types import SimpleNamespace

import llm_client
from capability_probe import clear_probe_cache, probe_logprobs_capability


class FakeAuth:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text='{"account_type":"Supplies"}')


class FakeChatCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[])


class FakeOpenAI:
    last = None

    def __init__(self, **kwargs):
        self.responses = FakeResponses()
        self.chat = SimpleNamespace(completions=FakeChatCompletions())
        FakeOpenAI.last = self


def test_responses_request_enables_output_logprobs(monkeypatch, tmp_path):
    config_file = tmp_path / "config"
    config_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "openai.gpt-4o")
    monkeypatch.setattr(llm_client, "OciUserPrincipalAuth", FakeAuth)
    monkeypatch.setattr(llm_client, "OciOpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_client, "LLM_LOGPROBS_MODE", "auto")
    monkeypatch.setattr(llm_client, "LLM_LOGPROBS_TOP_K", 5)

    client = llm_client.OCIResponsesClient(config_path=config_file)
    client.call("prompt", allowed_account_types=["Supplies"])
    kwargs = FakeOpenAI.last.responses.kwargs
    assert kwargs["include"] == ["message.output_text.logprobs"]
    assert kwargs["top_logprobs"] == 5
    assert kwargs["temperature"] == 0


def test_chat_fallback_uses_chat_logprobs(monkeypatch, tmp_path):
    config_file = tmp_path / "config"
    config_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "openai.gpt-oss-20b")
    monkeypatch.setattr(llm_client, "OciUserPrincipalAuth", FakeAuth)
    monkeypatch.setattr(llm_client, "OciOpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_client, "LLM_LOGPROBS_TOP_K", 4)
    client = llm_client.OCIResponsesClient(config_path=config_file)
    client.call_chat_fallback("prompt", allowed_account_types=["Supplies"])
    kwargs = FakeOpenAI.last.chat.completions.kwargs
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 4


def test_native_chat_uses_low_reasoning_strict_schema_and_compact_codes(monkeypatch, tmp_path):
    import oci
    import oci.generative_ai_inference as oci_inference

    config_file = tmp_path / "config"
    config_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "openai.gpt-oss-20b")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "test-compartment")
    monkeypatch.setattr(llm_client, "OciUserPrincipalAuth", FakeAuth)
    monkeypatch.setattr(llm_client, "OciOpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_client, "LLM_LOGPROBS_TOP_K", 5)
    monkeypatch.setattr(llm_client, "LLM_NATIVE_MAX_OUTPUT_TOKENS", 256)
    monkeypatch.setattr(llm_client, "LLM_NATIVE_REASONING_EFFORT", "LOW")
    monkeypatch.setattr(oci.config, "from_file", lambda *args: {"region": "us-chicago-1"})

    final_text = (
        '{"line_description":"paper","account_type_code":"A",'
        '"inferred_account_type":null,"reason":"supply","confidence":0.9}'
    )
    tokens = list(final_text)
    tops = [{} for _ in tokens]
    tops[tokens.index("A")] = {"A": -0.1, "B": -1.0}

    class FakeNativeInference:
        last = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeNativeInference.last = self

        def chat(self, details):
            self.details = details
            raw = SimpleNamespace(
                chat_response=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=[SimpleNamespace(text=final_text)],
                                reasoning_content="",
                            ),
                            finish_reason="stop",
                            logprobs=SimpleNamespace(
                                tokens=tokens,
                                token_logprobs=[-0.1 for _ in tokens],
                                top_logprobs=tops,
                            ),
                        )
                    ]
                )
            )
            return SimpleNamespace(data=raw)

    monkeypatch.setattr(oci_inference, "GenerativeAiInferenceClient", FakeNativeInference)
    client = llm_client.OCIResponsesClient(config_path=config_file)
    result = client.call_native_chat_fallback(
        "prompt",
        allowed_account_types=["Supplies"],
    )

    request = FakeNativeInference.last.details.chat_request
    assert request.reasoning_effort == "LOW"
    assert request.max_tokens == 256
    assert request.log_probs == 5
    assert request.response_format.type == "JSON_SCHEMA"
    assert request.response_format.json_schema.is_strict is True
    assert result.provider == "oci_native_chat"


def test_gpt_oss_logprob_call_routes_directly_to_native_chat(monkeypatch, tmp_path):
    config_file = tmp_path / "config"
    config_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "openai.gpt-oss-20b")
    monkeypatch.setattr(llm_client, "OciUserPrincipalAuth", FakeAuth)
    monkeypatch.setattr(llm_client, "OciOpenAI", FakeOpenAI)
    client = llm_client.OCIResponsesClient(config_path=config_file)
    calls = {}

    def native(prompt, allowed_account_types=None, timeout_seconds=None, max_output_tokens=None):
        calls["args"] = (prompt, allowed_account_types, timeout_seconds, max_output_tokens)
        return SimpleNamespace(provider="oci_native_chat", route="oci_native_chat")

    monkeypatch.setattr(client, "call_native_chat_fallback", native)
    result = client.call("prompt", allowed_account_types=["Supplies"])
    assert result.provider == "oci_native_chat"
    assert calls["args"][1] == ["Supplies"]


def test_capability_probe_is_cached_without_storing_response_payload():
    clear_probe_cache()

    class ProbeClient:
        model = "openai.gpt-oss-20b"
        service_endpoint = "https://example.invalid"
        request_mode = "structured_json"
        logprobs_mode = "auto"
        provider = "oci_responses"
        calls = 0

        def probe_logprobs(self):
            self.calls += 1
            return SimpleNamespace(
                output_text='{"account_type":"Supplies"}',
                output=[
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                logprobs=[SimpleNamespace(token='{"account_type":"', logprob=-0.2), SimpleNamespace(token='Supplies', logprob=-0.1), SimpleNamespace(token='"}', logprob=-0.2)]
                            )
                        ]
                    )
                ],
            )

    client = ProbeClient()
    first = probe_logprobs_capability(client)
    second = probe_logprobs_capability(client)
    assert first["supported"] is True
    assert first["sdk_version"] == "1.1.0"
    assert second == first
    assert client.calls == 1
