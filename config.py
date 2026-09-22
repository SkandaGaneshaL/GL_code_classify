from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
POC_DIR = BASE_DIR.parent
load_dotenv(POC_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)


def normalize_path(path: str | Path | None, base_dir: Path = BASE_DIR) -> Path | None:
    """Expand and resolve a configured path without requiring it to exist."""
    if path is None or not str(path).strip():
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path).strip()))
    value = Path(expanded)
    if not value.is_absolute():
        value = base_dir / value
    return value.resolve()


def resolve_oci_config_path(config_path: str | Path | None = None) -> Path:
    """Return the one canonical OCI config path used by every OCI client."""
    configured = config_path if config_path is not None else os.getenv("GL_OCI_CONFIG_PATH")
    return normalize_path(configured, BASE_DIR) or (Path.home() / ".oci" / "config").resolve()

DEFAULT_XML_PATH = Path(
    r"C:\Users\tchand\Downloads\ad06361d-d790-42b9-8338-202b81aadb92.xml"
)
XML_PATH = Path(os.getenv("GL_XML_PATH", str(DEFAULT_XML_PATH)))
OCI_CONFIG_PATH = resolve_oci_config_path()
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT").strip() or "DEFAULT"
DB_TABLE = "AG_GL_ACCOUNT_HISTORY_CASES"
DEFAULT_AP_INVOICE_ROOT = Path(r"C:\Users\Skanda Ganesha L\Downloads\AP_INVOICE")
AP_INVOICE_ROOT = normalize_path(os.getenv("AP_INVOICE_ROOT"), BASE_DIR) or DEFAULT_AP_INVOICE_ROOT
AP_EXTRACTION_MODEL_ID = os.getenv("AP_EXTRACTION_MODEL_ID", "")
AP_EXTRACTION_MAX_BYTES = int(os.getenv("AP_EXTRACTION_MAX_BYTES", str(20 * 1024 * 1024)))
AP_EXTRACTION_MAX_PAGES = int(os.getenv("AP_EXTRACTION_MAX_PAGES", "20"))
AP_EXTRACTION_SPLIT_PASSES = os.getenv("AP_EXTRACTION_SPLIT_PASSES", "").strip()
CLASSIFIER_MODEL_ID = os.getenv("LLM_MODEL", "").strip()
REQUIRED_CLASSIFIER_MODEL = os.getenv("GL_REQUIRED_CLASSIFIER_MODEL", "openai.gpt-oss-20b").strip()
ENFORCE_REQUIRED_CLASSIFIER_MODEL = os.getenv("GL_ENFORCE_REQUIRED_CLASSIFIER_MODEL", "1").lower() not in {"0", "false", "no"}
ALLOW_CLASSIFIER_MODEL_FALLBACK = os.getenv("GL_ALLOW_CLASSIFIER_MODEL_FALLBACK", "0").lower() in {"1", "true", "yes"}
CAPABILITY_PROBE_CACHE_VERSION = os.getenv("GL_CAPABILITY_PROBE_CACHE_VERSION", "1").strip() or "1"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("GL_EMBEDDING_MODEL", "cohere.embed-v4.0"),
)
EXPECTED_EMBEDDING_DIMENSION = 1024
EMBEDDING_OUTPUT_DIMENSIONS = int(
    os.getenv("GL_EMBEDDING_OUTPUT_DIMENSIONS", str(EXPECTED_EMBEDDING_DIMENSION))
)
DEFAULT_TOP_K = int(os.getenv("GL_TOP_K", "3"))
DENSE_TOP_K = int(os.getenv("GL_DENSE_TOP_K", "15"))
SPARSE_TOP_K = int(os.getenv("GL_SPARSE_TOP_K", "15"))
CANDIDATE_TOP_K = DENSE_TOP_K
RRF_K = int(os.getenv("GL_RRF_K", "60"))
MIN_SIMILARITY_SCORE = float(os.getenv("GL_MIN_SIMILARITY_SCORE", "0.60"))
IDENTITY_MIN_REAL_POSTS = int(os.getenv("GL_IDENTITY_MIN_REAL_POSTS", "2"))
VENDOR_MIN_REAL_POSTS = int(os.getenv("GL_VENDOR_MIN_REAL_POSTS", "5"))
VENDOR_DOMINANCE_THRESHOLD = float(os.getenv("GL_VENDOR_DOMINANCE_THRESHOLD", "0.90"))
AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("GL_AUTO_CONFIDENCE_THRESHOLD", "0.90"))
REVIEW_CONFIDENCE_THRESHOLD = float(os.getenv("GL_REVIEW_CONFIDENCE_THRESHOLD", "0.70"))
AUTO_DEFAULT_ENABLED = os.getenv("GL_AUTO_DEFAULT_ENABLED", "0").lower() in {"1", "true", "yes"}
AUTO_PRECISION_TARGET = float(os.getenv("GL_AUTO_PRECISION_TARGET", "0.98"))
AUTO_MIN_ACCEPTED_VALIDATION = int(os.getenv("GL_AUTO_MIN_ACCEPTED_VALIDATION", "200"))
AUTO_REQUIRE_CALIBRATION = os.getenv("GL_AUTO_REQUIRE_CALIBRATION", "1").lower() not in {"0", "false", "no"}
LLM_TIMEOUT_SECONDS = float(
    os.getenv(
        "GL_LLM_TIMEOUT_SECONDS",
        os.getenv("GL_CLASSIFICATION_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "90")),
    )
)
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("GL_LLM_MAX_OUTPUT_TOKENS", "180"))
LLM_NATIVE_MAX_OUTPUT_TOKENS = int(os.getenv("GL_NATIVE_MAX_OUTPUT_TOKENS", "256"))
LLM_NATIVE_REASONING_EFFORT = (
    os.getenv("GL_NATIVE_REASONING_EFFORT", "LOW").strip().upper() or "LOW"
)
USE_STRUCTURED_OUTPUTS = os.getenv("GL_USE_STRUCTURED_OUTPUTS", "1").lower() not in {"0", "false", "no"}
CLASSIFIER_PROVIDER = os.getenv("CLASSIFIER_PROVIDER", "oci_responses").strip() or "oci_responses"
LLM_LOGPROBS_MODE = os.getenv("LLM_LOGPROBS_MODE", "auto").strip().lower() or "auto"
LLM_LOGPROBS_TOP_K = max(1, min(20, int(os.getenv("LLM_LOGPROBS_TOP_K", "5"))))
LLM_LOGPROBS_REQUIRE = os.getenv("LLM_LOGPROBS_REQUIRE", "0").lower() in {"1", "true", "yes"}
LLM_LOGPROBS_CALIBRATION_PATH = normalize_path(os.getenv("LLM_LOGPROBS_CALIBRATION_PATH"), BASE_DIR)
SEGMENT3_RANKER_CALIBRATION_PATH = normalize_path(os.getenv("SEGMENT3_RANKER_CALIBRATION_PATH"), BASE_DIR)
COA_VALUE_SET_PATH = normalize_path(os.getenv("GL_COA_VALUE_SET_PATH"), BASE_DIR)
COA_COMBINATION_RULES_PATH = normalize_path(os.getenv("GL_COA_COMBINATION_RULES_PATH"), BASE_DIR)
SEGMENT3_TAXONOMY_PATH = normalize_path(os.getenv("GL_SEGMENT3_TAXONOMY_PATH"), BASE_DIR)
COA_VERSION = os.getenv("GL_COA_VERSION", "segment3-demo-16-v1").strip() or "segment3-demo-16-v1"
STRICT_FINANCE_MODE = os.getenv("GL_STRICT_FINANCE_MODE", "0").lower() in {"1", "true", "yes"}
MIN_EXTRACTION_QUALITY = float(os.getenv("GL_MIN_EXTRACTION_QUALITY", "0.80"))
LLM_ROUTING_MODE = os.getenv("GL_LLM_ROUTING_MODE", "always").strip().lower() or "always"
if LLM_ROUTING_MODE not in {"always", "uncertain_only", "never"}:
    raise ValueError("GL_LLM_ROUTING_MODE must be always, uncertain_only, or never")
LLM_CANDIDATE_SCORING = os.getenv("LLM_CANDIDATE_SCORING", "off").strip().lower() in {"1", "true", "yes", "on"}
LLM_CANDIDATE_SCORING_MODE = os.getenv("LLM_CANDIDATE_SCORING_MODE", "off").strip().lower() or "off"
LLM_GUIDED_MODE = os.getenv("LLM_GUIDED_MODE", "off").strip().lower() or "off"
LLM_COMPACT_CODE_EVIDENCE = os.getenv("LLM_COMPACT_CODE_EVIDENCE", "auto").strip().lower() or "auto"
LLM_CAPABILITY_PROBE_CACHE = os.getenv("LLM_CAPABILITY_PROBE_CACHE", "1").lower() not in {"0", "false", "no"}
LLM_LOGPROBS_SATURATION_THRESHOLD = float(os.getenv("LLM_LOGPROBS_SATURATION_THRESHOLD", "0.999"))


def validate_classifier_configuration() -> dict[str, str | bool]:
    """Validate the production classifier model before any LLM call."""
    model = os.getenv("LLM_MODEL", CLASSIFIER_MODEL_ID).strip()
    required = os.getenv("GL_REQUIRED_CLASSIFIER_MODEL", REQUIRED_CLASSIFIER_MODEL).strip()
    endpoint = os.getenv("OCI_SERVICE_ENDPOINT", "").strip()
    compartment = os.getenv("OCI_COMPARTMENT_ID", "").strip()
    if ENFORCE_REQUIRED_CLASSIFIER_MODEL and model != required:
        raise RuntimeError(
            "Classifier configuration error: "
            f"LLM_MODEL={model or '<missing>'!r}; required {required!r}."
        )
    if not model:
        raise RuntimeError("Classifier configuration error: LLM_MODEL is required.")
    if not endpoint:
        raise RuntimeError("Classifier configuration error: OCI_SERVICE_ENDPOINT is required.")
    if not compartment:
        raise RuntimeError("Classifier configuration error: OCI_COMPARTMENT_ID is required.")
    return {
        "model": model,
        "required_model": required,
        "endpoint": endpoint,
        "fallback_allowed": ALLOW_CLASSIFIER_MODEL_FALLBACK,
        "enforced": ENFORCE_REQUIRED_CLASSIFIER_MODEL,
    }


def resolve_path(path: str | None, base_dir: Path = POC_DIR) -> str | None:
    """Resolve a path against the existing POC folder, preserving its legacy API."""
    value = normalize_path(path, base_dir)
    return str(value) if value else None
