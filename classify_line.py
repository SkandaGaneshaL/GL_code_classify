from __future__ import annotations

import argparse
import json
try:
    from .classification import classify_line
    from .embeddings import create_document_embedder, load_oci_settings
except ImportError:  # Supports running the file directly from this folder.
    from classification import classify_line
    from embeddings import create_document_embedder, load_oci_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an invoice line account type and map it to a GL code")
    parser.add_argument("line_description", nargs="?")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--compartment-id", default=None)
    parser.add_argument("--service-endpoint", default=None)
    args = parser.parse_args()
    line_description = args.line_description or input("Enter an invoice line description: ").strip()
    settings = load_oci_settings(
        profile=args.profile,
        compartment_id=args.compartment_id,
        service_endpoint=args.service_endpoint,
    )
    result = classify_line(
        line_description,
        create_document_embedder(settings),
        settings,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
