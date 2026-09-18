import pytest

import config


def test_classifier_configuration_rejects_non_gpt_oss(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai.gpt-4o")
    monkeypatch.setenv("OCI_SERVICE_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "compartment")
    with pytest.raises(RuntimeError, match="required 'openai.gpt-oss-20b'"):
        config.validate_classifier_configuration()


def test_classifier_configuration_accepts_required_gpt_oss(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai.gpt-oss-20b")
    monkeypatch.setenv("OCI_SERVICE_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "compartment")
    result = config.validate_classifier_configuration()
    assert result["model"] == "openai.gpt-oss-20b"
    assert result["fallback_allowed"] is False
