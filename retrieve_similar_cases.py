from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

try:
    from . import db_utils
    from .embeddings import create_query_embedder, embed_query, load_oci_settings
    from .feature_pack import build_line_feature
except ImportError:  # Supports running the file directly from this folder.
    import db_utils
    from embeddings import create_query_embedder, embed_query, load_oci_settings
    from feature_pack import build_line_feature


LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_K = 3


def retrieve_cases(line_description: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Embed one query and return the nearest Oracle history cases."""
    text = line_description.strip()
    if not text:
        raise ValueError("Line description cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    settings = load_oci_settings()
    feature = build_line_feature({"LineDescription": text}, vendor="", line_index=0)
    embedder = create_query_embedder(settings)
    query_embedding = embed_query(feature["retrieval_text"], embedder)
    cases = db_utils.search_similar_history(query_embedding, top_k=top_k, target_accuracy=95)

    # Recalculate explicitly so the terminal output documents the requested formula.
    for case in cases:
        distance = case.get("cosine_distance")
        case["similarity_score"] = None if distance is None else 1.0 - float(distance)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve the top similar GL account history cases for a line description"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of cases to retrieve; defaults to 3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    line_description = input("Paste invoice line description: ").strip()

    try:
        cases = retrieve_cases(line_description, top_k=args.top_k)
        print(f"\nTOP {args.top_k} SIMILAR CASES")
        print("Similarity score = 1 - cosine distance\n")
        if not cases:
            print("No historical cases found.")
            return

        for rank, case in enumerate(cases, start=1):
            distance = case.get("cosine_distance")
            similarity = case.get("similarity_score")
            print(f"{rank}. Similarity score: {similarity:.6f}")
            print(f"   Cosine distance:  {distance:.6f}")
            print(f"   Line description: {case.get('line_description')}")
            print(f"   Account type:     {case.get('account_type')}")
            print(f"   GL code:          {case.get('gl_code')}")
            print()
    except Exception as exc:
        LOGGER.exception("Similar-case retrieval failed")
        print(f"Retrieval failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
