from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from . import db_utils
    from .embeddings import create_document_embedder, embed_documents, load_oci_settings
    from .xml_dataset import read_gl_history
except ImportError:  # Supports running the file directly from this folder.
    import db_utils
    from embeddings import create_document_embedder, embed_documents, load_oci_settings
    from xml_dataset import read_gl_history


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and load GL account history into Oracle")
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--compartment-id", default=None)
    parser.add_argument("--service-endpoint", default=None)
    parser.add_argument("--skip-empty-descriptions", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        from .config import XML_PATH
    except ImportError:  # Supports running the file directly from this folder.
        from config import XML_PATH

    xml_path = args.xml_path or XML_PATH
    records = read_gl_history(xml_path, skip_empty_descriptions=args.skip_empty_descriptions)
    settings = load_oci_settings(
        profile=args.profile,
        compartment_id=args.compartment_id,
        service_endpoint=args.service_endpoint,
    )
    embedder = create_document_embedder(settings)
    raw_records = [record.as_dict() for record in records]
    vectors = embed_documents(raw_records, embedder)
    rows = ({**record, "embedding": vector} for record, vector in zip(raw_records, vectors))
    count = db_utils.bulk_upsert_history(rows)
    print(f"Loaded {count} GL history rows into {db_utils.TABLE_NAME}")


if __name__ == "__main__":
    main()
