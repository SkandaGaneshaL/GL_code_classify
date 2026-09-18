import os
from pathlib import Path

import pytest

import embeddings
import llm_client
from ap_invoice_adapter import _ap_environment
from config import normalize_path, resolve_oci_config_path


AP_ROOT = Path(r"C:\Users\Skanda Ganesha L\Downloads\AP_INVOICE")


def test_normalize_path_expands_variables_and_resolves_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OCI_DIR", "oci")
    assert normalize_path("$TEST_OCI_DIR/config", tmp_path) == (tmp_path / "oci/config").resolve()


def test_normalize_path_preserves_absolute_windows_path():
    path = normalize_path(r"C:\Users\Skanda Ganesha L\.oci\config")
    assert path == Path(r"C:\Users\Skanda Ganesha L\.oci\config").resolve()


def test_resolve_oci_config_path_uses_gl_override(tmp_path, monkeypatch):
    configured = tmp_path / "custom-oci-config"
    monkeypatch.setenv("GL_OCI_CONFIG_PATH", str(configured))
    assert resolve_oci_config_path() == configured.resolve()


def test_embedding_settings_use_resolved_path_and_profile(tmp_path, monkeypatch):
    configured = tmp_path / "config"
    configured.write_text("placeholder", encoding="utf-8")
    calls = {}

    def fake_from_file(path, profile):
        calls["path"] = path
        calls["profile"] = profile
        return {"region": "us-chicago-1"}

    monkeypatch.setenv("GL_OCI_CONFIG_PATH", str(configured))
    monkeypatch.setenv("OCI_PROFILE", "ALT")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "compartment-for-test")
    monkeypatch.setattr(embeddings.oci.config, "from_file", fake_from_file)
    settings = embeddings.load_oci_settings()
    assert settings.config_path == configured.resolve()
    assert settings.profile == "ALT"
    assert calls == {"path": str(configured.resolve()), "profile": "ALT"}


def test_ap_environment_bridges_config_and_restores_process_environment(monkeypatch):
    original_file = os.environ.get("OCI_CONFIG_FILE")
    original_profile = os.environ.get("OCI_PROFILE")
    monkeypatch.setenv("OCI_CONFIG_FILE", "original-config")
    monkeypatch.setenv("OCI_PROFILE", "TEST_PROFILE")
    with _ap_environment(AP_ROOT):
        assert os.environ["OCI_CONFIG_FILE"]
        assert Path(os.environ["OCI_CONFIG_FILE"]).is_absolute()
        assert os.environ["OCI_PROFILE"] == "TEST_PROFILE"
    assert os.environ.get("OCI_CONFIG_FILE") == "original-config"
    assert os.environ.get("OCI_PROFILE") == "TEST_PROFILE"
    assert original_file != "original-config" or original_profile == "TEST_PROFILE"


def test_classifier_auth_uses_explicit_config_path_and_profile(tmp_path, monkeypatch):
    config_file = tmp_path / "oci-config"
    config_file.write_text("placeholder", encoding="utf-8")
    calls = {}

    class FakeAuth:
        def __init__(self, *, config_file, profile_name):
            calls["auth"] = (config_file, profile_name)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_client, "OciUserPrincipalAuth", FakeAuth)
    monkeypatch.setattr(llm_client, "OciOpenAI", FakeOpenAI)
    llm_client.OCIResponsesClient(config_path=config_file, profile="ALT")
    assert calls["auth"] == (str(config_file.resolve()), "ALT")


def test_classifier_missing_config_fails_before_client_creation(tmp_path, monkeypatch):
    secret_name = "private-secret-marker"
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with pytest.raises(FileNotFoundError, match="OCI config file was not found") as exc_info:
        llm_client.OCIResponsesClient(config_path=tmp_path / secret_name)
    assert secret_name in str(exc_info.value)
    assert "DB_PASSWORD" not in str(exc_info.value)
