from __future__ import annotations

import argparse
import json
try:
    from .classification import classify_line
    from .embeddings import create_query_embedder, load_oci_settings
except ImportError:  # Supports running the file directly from this folder.
    from classification import classify_line
    from embeddings import create_query_embedder, load_oci_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an invoice line account type and map it to a GL code")
    parser.add_argument("line_description", nargs="?")
    parser.add_argument("--vendor", default=None)
    parser.add_argument("--line-type", default=None, choices=["ITEM", "TAX", "FREIGHT", "MISC"])
    parser.add_argument("--line-amount", default=None)
    parser.add_argument("--unit-price", default=None)
    parser.add_argument("--quantity", dest="quantity_invoiced", default=None)
    parser.add_argument("--force-llm", action="store_true")
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
        create_query_embedder(settings),
        settings,
        vendor=args.vendor,
        line_type=args.line_type,
        line_amount=args.line_amount,
        unit_price=args.unit_price,
        quantity_invoiced=args.quantity_invoiced,
        force_llm=args.force_llm,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
