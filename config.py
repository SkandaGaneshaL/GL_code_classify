from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
POC_DIR = BASE_DIR.parent
load_dotenv(POC_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

DEFAULT_XML_PATH = Path(
    r"C:\Users\tchand\Downloads\ad06361d-d790-42b9-8338-202b81aadb92.xml"
)
XML_PATH = Path(os.getenv("GL_XML_PATH", str(DEFAULT_XML_PATH)))
OCI_CONFIG_PATH = Path(
    os.getenv("GL_OCI_CONFIG_PATH", str(POC_DIR / ".oci" / "config"))
)
DB_TABLE = "AG_GL_ACCOUNT_HISTORY_CASES"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("GL_EMBEDDING_MODEL", "cohere.embed-english-v3.0"),
)
EXPECTED_EMBEDDING_DIMENSION = 1024
DEFAULT_TOP_K = int(os.getenv("GL_TOP_K", "3"))
CANDIDATE_TOP_K = 5
MIN_SIMILARITY_SCORE = float(os.getenv("GL_MIN_SIMILARITY_SCORE", "0.60"))


def resolve_path(path: str | None, base_dir: Path = POC_DIR) -> str | None:
    """Resolve a relative path against the existing POC folder."""
    if not path:
        return None
    value = Path(path)
    return str(value if value.is_absolute() else base_dir / value)
