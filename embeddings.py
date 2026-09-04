from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import oci
from langchain_oci import OCIGenAIEmbeddings

try:
    from .config import EMBEDDING_MODEL, EXPECTED_EMBEDDING_DIMENSION, POC_DIR
except ImportError:  # Supports running the files directly from this folder.
    from config import EMBEDDING_MODEL, EXPECTED_EMBEDDING_DIMENSION, POC_DIR


LOGGER = logging.getLogger(__name__)
EMBEDDING_INPUT_TYPE = "SEARCH_DOCUMENT"


@dataclass(frozen=True)
class OCIEmbeddingSettings:
    config_path: Path
    profile: str
    compartment_id: str
    service_endpoint: str
    model_id: str = EMBEDDING_MODEL


def load_oci_settings(
    config_path: Path | None = None,
    profile: str | None = None,
    compartment_id: str | None = None,
    service_endpoint: str | None = None,
) -> OCIEmbeddingSettings:
    path = config_path or Path(os.getenv("GL_OCI_CONFIG_PATH", str(POC_DIR / ".oci" / "config")))
    selected_profile = profile or os.getenv("OCI_PROFILE", "DEFAULT")
    if not path.is_file():
        raise FileNotFoundError(f"OCI config file was not found: {path}")
    config = oci.config.from_file(str(path), selected_profile)
    resolved_compartment = compartment_id or os.getenv("OCI_COMPARTMENT_ID")
    if not resolved_compartment:
        raise ValueError("OCI_COMPARTMENT_ID must be set")
    endpoint = service_endpoint or os.getenv("OCI_SERVICE_ENDPOINT")
    if not endpoint:
        region = config.get("region")
        if not region:
            raise ValueError("OCI_SERVICE_ENDPOINT is required when OCI config has no region")
        endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    return OCIEmbeddingSettings(path, selected_profile, resolved_compartment, endpoint)


def create_document_embedder(settings: OCIEmbeddingSettings) -> OCIGenAIEmbeddings:
    return OCIGenAIEmbeddings(
        model_id=settings.model_id,
        service_endpoint=settings.service_endpoint,
        compartment_id=settings.compartment_id,
        auth_profile=settings.profile,
        auth_file_location=str(settings.config_path),
        input_type=EMBEDDING_INPUT_TYPE,
        truncate="END",
        batch_size=96,
    )


def embed_documents(records: list[dict[str, object]], embedder: OCIGenAIEmbeddings) -> list[list[float]]:
    LOGGER.info("Generating %d embeddings with input_type=%s", len(records), EMBEDDING_INPUT_TYPE)
    descriptions = []
    for row in records:
        description = row.get("line_description") or row.get("LINE_DESCRIPTION")
        if not description:
            raise ValueError("Each embedding record must contain a non-empty line description")
        descriptions.append(str(description))
    vectors = embedder.embed_documents(descriptions)
    if len(vectors) != len(records):
        raise RuntimeError(f"OCI returned {len(vectors)} embeddings for {len(records)} records")
    for vector in vectors:
        if len(vector) != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, received {len(vector)}"
            )
    return vectors


def embed_query(line_description: str, embedder: OCIGenAIEmbeddings) -> list[float]:
    vector = embedder.embed_query(line_description)
    if len(vector) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected query embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, received {len(vector)}"
        )
    return vector
