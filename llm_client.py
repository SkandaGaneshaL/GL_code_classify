from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from oci_openai import OciOpenAI, OciUserPrincipalAuth

try:
    from .config import (
        CLASSIFIER_PROVIDER,
        LLM_CANDIDATE_SCORING,
        LLM_CANDIDATE_SCORING_MODE,
        LLM_GUIDED_MODE,
        LLM_MAX_OUTPUT_TOKENS,
        LLM_NATIVE_MAX_OUTPUT_TOKENS,
        LLM_NATIVE_REASONING_EFFORT,
        LLM_COMPACT_CODE_EVIDENCE,
        LLM_LOGPROBS_MODE,
        LLM_LOGPROBS_REQUIRE,
        LLM_LOGPROBS_TOP_K,
        LLM_TIMEOUT_SECONDS,
        OCI_CONFIG_PATH,
        OCI_PROFILE,
        USE_STRUCTURED_OUTPUTS,
    )
except ImportError:
    from config import (
        CLASSIFIER_PROVIDER,
        LLM_CANDIDATE_SCORING,
        LLM_CANDIDATE_SCORING_MODE,
        LLM_GUIDED_MODE,
        LLM_MAX_OUTPUT_TOKENS,
        LLM_NATIVE_MAX_OUTPUT_TOKENS,
        LLM_NATIVE_REASONING_EFFORT,
        LLM_COMPACT_CODE_EVIDENCE,
        LLM_LOGPROBS_MODE,
        LLM_LOGPROBS_REQUIRE,
        LLM_LOGPROBS_TOP_K,
        LLM_TIMEOUT_SECONDS,
        OCI_CONFIG_PATH,
        OCI_PROFILE,
        USE_STRUCTURED_OUTPUTS,
    )

try:
    from .provider_response import normalize_native_chat_response
except ImportError:
    from provider_response import normalize_native_chat_response


def _is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _native_code_map(allowed_account_types: list[str] | None) -> dict[str, str] | None:
    if LLM_COMPACT_CODE_EVIDENCE == "off" or not allowed_account_types:
        return None
    values = list(dict.fromkeys([*allowed_account_types, "Unknown"]))
    # One-character codes keep the comparison at a single output position.
    # The provider supports at most 20 top alternatives and one value is
    # reserved for every requested taxonomy type plus Unknown.
    if len(values) > 20:
        return None
    return {chr(ord("A") + index): value for index, value in enumerate(values)}


class OCIResponsesClient:
    """OCI OpenAI-compatible Responses API client."""

    def __init__(self, *, config_path: str | Path | None = None, profile: str | None = None) -> None:
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise ValueError("LLM_MODEL must be set in GL_code_classify/.env")
        path = Path(config_path) if config_path is not None else OCI_CONFIG_PATH
        path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OCI config file was not found at {path}")
        selected_profile = (profile or os.getenv("OCI_PROFILE", OCI_PROFILE)).strip() or OCI_PROFILE
        auth = OciUserPrincipalAuth(config_file=str(path), profile_name=selected_profile)
        self.model = model
        self.provider = "oci_native_chat" if _is_gpt_oss(model) else CLASSIFIER_PROVIDER
        self.request_mode = "guided_json" if LLM_GUIDED_MODE == "guided_json" else "structured_json"
        self.logprobs_mode = LLM_LOGPROBS_MODE
        self.logprobs_required = LLM_LOGPROBS_REQUIRE or LLM_LOGPROBS_MODE == "required"
        self.native_max_output_tokens = LLM_NATIVE_MAX_OUTPUT_TOKENS
        self.native_reasoning_effort = LLM_NATIVE_REASONING_EFFORT
        self.candidate_scoring = LLM_CANDIDATE_SCORING
        # A prompt that says "return exactly this candidate" is not a neutral
        # class scorer.  Keep it disabled unless a provider-specific,
        # teacher-forced implementation is explicitly verified.
        self.candidate_scoring_mode = LLM_CANDIDATE_SCORING_MODE
        # No neutral teacher-forced scorer is implemented for OCI in this
        # version.  This guard prevents the legacy prompt-forcing method from
        # being mistaken for comparable candidate likelihoods.
        self.candidate_scoring_verified = False
        self.config_path = path
        self.profile = selected_profile
        self.service_endpoint = os.getenv("OCI_SERVICE_ENDPOINT")
        self.compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        client_kwargs = {
            "service_endpoint": self.service_endpoint,
            "compartment_id": self.compartment_id,
        }
        try:
            self.client = OciOpenAI(auth=auth, **client_kwargs)
        except TypeError as exc:
            # openai 3.x uses httpx2, while oci-openai's signer subclasses
            # the legacy httpx.Auth type. Adapt the signer when that newer
            # client is present; openai 1.x continues through the fast path.
            if 'Invalid "auth" argument' not in str(exc):
                raise
            import httpx2

            class Httpx2OciAuth(httpx2.Auth):
                def __init__(self, legacy_auth: Any) -> None:
                    self.legacy_auth = legacy_auth

                def auth_flow(self, request: Any):
                    yield from self.legacy_auth.auth_flow(request)

            self.client = OciOpenAI(auth=Httpx2OciAuth(auth), **client_kwargs)

    def call(
        self,
        prompt: str,
        allowed_account_types: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        if _is_gpt_oss(self.model):
            # OCI Responses rejects logprobs for gpt-oss.  Route directly to
            # the native contract so every classification gets one coherent
            # request rather than paying for a guaranteed 400 first.
            return self.call_native_chat_fallback(
                prompt,
                allowed_account_types=allowed_account_types,
                timeout_seconds=timeout_seconds,
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "temperature": 0,
            "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
            "timeout": timeout_seconds or LLM_TIMEOUT_SECONDS,
        }
        if USE_STRUCTURED_OUTPUTS and allowed_account_types:
            account_enum = list(dict.fromkeys([*allowed_account_types, "Unknown"]))
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "account_type_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "line_description": {"type": "string"},
                            "account_type": {"type": "string", "enum": account_enum},
                            "inferred_account_type": {
                                "anyOf": [
                                    {"type": "string", "enum": account_enum},
                                    {"type": "null"},
                                ]
                            },
                            "reason": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": [
                            "line_description",
                            "account_type",
                            "inferred_account_type",
                            "reason",
                            "confidence",
                        ],
                    },
                }
            }
        if self.logprobs_mode in {"auto", "required", "report_only"}:
            # Responses logprobs are opt-in.  OCI may reject these fields for
            # a particular model or endpoint; classification owns fallback.
            kwargs["include"] = ["message.output_text.logprobs"]
            kwargs["top_logprobs"] = LLM_LOGPROBS_TOP_K
        return self.client.responses.create(**kwargs)

    def call_chat_fallback(
        self,
        prompt: str,
        allowed_account_types: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Try the OpenAI-compatible Chat Completions logprob contract."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": LLM_MAX_OUTPUT_TOKENS,
            "timeout": timeout_seconds or LLM_TIMEOUT_SECONDS,
            "logprobs": True,
            "top_logprobs": LLM_LOGPROBS_TOP_K,
        }
        # Chat Completions providers commonly accept response_format rather
        # than the Responses ``text.format`` envelope.  The prompt and parser
        # remain the final validation boundary.
        if USE_STRUCTURED_OUTPUTS and allowed_account_types:
            enum_values = list(dict.fromkeys([*allowed_account_types, "Unknown"]))
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["metadata"] = {"allowed_account_types": ",".join(enum_values)}
        return self.client.chat.completions.create(**kwargs)

    def call_native_chat_fallback(
        self,
        prompt: str,
        allowed_account_types: list[str] | None = None,
        timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> Any:
        """Use OCI's native Chat API, whose request field is ``log_probs``."""
        import oci
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        from oci.generative_ai_inference.models import (
            ChatDetails,
            GenericChatRequest,
            JsonSchemaResponseFormat,
            OnDemandServingMode,
            ResponseJsonSchema,
            TextContent,
            UserMessage,
        )

        code_map = _native_code_map(allowed_account_types)
        enum_values = list(dict.fromkeys([*(allowed_account_types or []), "Unknown"]))
        output_schema: dict[str, Any]
        request_prompt = prompt
        if code_map:
            codes = list(code_map)
            output_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "line_description": {"type": "string"},
                    "account_type_code": {"type": "string", "enum": codes},
                    "inferred_account_type": {
                        "anyOf": [
                            {"type": "string", "enum": enum_values},
                            {"type": "null"},
                        ]
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "line_description",
                    "account_type_code",
                    "inferred_account_type",
                    "reason",
                    "confidence",
                ],
            }
            mapping = ", ".join(f"{code}={name}" for code, name in code_map.items())
            request_prompt = (
                f"{prompt}\n\nTransport requirement: return account_type_code using this exact mapping: {mapping}. "
                "The adapter will convert that field to account_type. Keep inferred_account_type null "
                "unless account_type is Unknown. Return only the JSON object."
            )
            schema_name = "account_type_code_decision"
        else:
            output_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "line_description": {"type": "string"},
                    "account_type": {"type": "string", "enum": enum_values or ["Unknown"]},
                    "inferred_account_type": {
                        "anyOf": [
                            {"type": "string", "enum": enum_values or ["Unknown"]},
                            {"type": "null"},
                        ]
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "line_description",
                    "account_type",
                    "inferred_account_type",
                    "reason",
                    "confidence",
                ],
            }
            schema_name = "account_type_decision"

        oci_config = oci.config.from_file(str(self.config_path), self.profile)
        endpoint = self.service_endpoint
        if not endpoint and oci_config.get("region"):
            endpoint = f"https://inference.generativeai.{oci_config['region']}.oci.oraclecloud.com"
        client = GenerativeAiInferenceClient(
            config=oci_config,
            service_endpoint=endpoint,
            timeout=timeout_seconds or LLM_TIMEOUT_SECONDS,
        )
        request = GenericChatRequest(
            api_format="GENERIC",
            messages=[UserMessage(content=[TextContent(text=request_prompt)])],
            temperature=0,
            max_tokens=max_output_tokens or LLM_NATIVE_MAX_OUTPUT_TOKENS,
            log_probs=min(20, max(LLM_LOGPROBS_TOP_K, len(code_map or {}))),
            reasoning_effort=LLM_NATIVE_REASONING_EFFORT,
            response_format=JsonSchemaResponseFormat(
                json_schema=ResponseJsonSchema(
                    name=schema_name,
                    description="Strict account-type classification response.",
                    schema=output_schema,
                    is_strict=True,
                )
            ),
        )
        details = ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=OnDemandServingMode(model_id=self.model),
            chat_request=request,
        )
        started = time.perf_counter()
        response = client.chat(details)
        raw = getattr(response, "data", response)
        headers = getattr(response, "headers", None)
        request_id = None
        if isinstance(headers, dict):
            request_id = headers.get("opc-request-id") or headers.get("x-request-id")
        return normalize_native_chat_response(
            raw,
            allowed_account_types=allowed_account_types or [],
            code_map=code_map,
            provider="oci_native_chat",
            model=self.model,
            request_id=request_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def probe_logprobs(self) -> Any:
        """Issue a static capability probe without invoice or customer data."""
        return self.call(
            '{"description":"office paper","account_type":"Supplies",'
            '"inferred_account_type":null,"reason":"static capability probe",'
            '"confidence":0.5}',
            allowed_account_types=["Supplies"],
        )

    def probe_compact_codes(self) -> dict[str, Any]:
        """Probe whether the endpoint exposes complete compact-code evidence."""
        codes = list("ABCDEFGHIJKLMNOPQ")
        if _is_gpt_oss(self.model):
            raw = self.call_native_chat_fallback(
                "Return JSON only with one account_type_code from A through Q: "
                '{"account_type_code":"A"}',
                allowed_account_types=codes,
            )
            observed = set(raw.candidate_logprobs)
            expected = set(codes) | {"Unknown"}
            return {
                "supported": bool(raw.candidate_logprobs),
                "evidence_scope": "full_taxonomy" if observed >= expected else "candidate_set",
                "observed_values": len(observed),
                "allowed_values": len(expected),
            }
        try:
            from .logprobs import analyze_json_field_logprobs
        except ImportError:
            from logprobs import analyze_json_field_logprobs

        raw = self.client.responses.create(
            model=self.model,
            input=(
                'Return JSON only with one account_type_code from A through Q: '
                '{"account_type_code":"A"}'
            ),
            temperature=0,
            max_output_tokens=24,
            timeout=LLM_TIMEOUT_SECONDS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "account_type_code_probe",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"account_type_code": {"type": "string", "enum": codes}},
                        "required": ["account_type_code"],
                    },
                }
            },
            include=["message.output_text.logprobs"],
            top_logprobs=max(LLM_LOGPROBS_TOP_K, len(codes)),
        )
        return analyze_json_field_logprobs(raw, "account_type_code", codes)

    def score_candidate(
        self,
        context: str,
        candidate_type: str,
        timeout_seconds: float | None = None,
    ) -> float | None:
        """Score one candidate only for a verified provider implementation.

        The legacy prompt-forcing approach is intentionally rejected because
        it changes the conditioning context for every candidate and cannot be
        interpreted as a class softmax.
        """
        if (
            not self.candidate_scoring_verified
            or self.logprobs_mode not in {"auto", "required", "report_only"}
        ):
            return None
        try:
            from .logprobs import score_label_logprobs
        except ImportError:
            from logprobs import score_label_logprobs

        raw = self.client.responses.create(
            model=self.model,
            input=(
                f"{context}\nReturn exactly this account type and no other text: {candidate_type}"
            ),
            temperature=0,
            max_output_tokens=24,
            timeout=timeout_seconds or LLM_TIMEOUT_SECONDS,
            include=["message.output_text.logprobs"],
            top_logprobs=LLM_LOGPROBS_TOP_K,
        )
        return score_label_logprobs(raw, candidate_type)
