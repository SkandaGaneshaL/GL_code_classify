from __future__ import annotations

import argparse
from pathlib import Path

import db_utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the GL account history table and vector index")
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("schema.sql"))
    args = parser.parse_args()
    statements = [part.strip() for part in args.schema.read_text(encoding="utf-8").split(";") if part.strip()]
    conn = db_utils.get_db_connection()
    try:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
        conn.commit()
    finally:
        conn.close()
    print(f"Created GL history schema from {args.schema}")


if __name__ == "__main__":
    main()

